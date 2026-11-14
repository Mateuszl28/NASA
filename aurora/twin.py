"""The twin as a running object: estimate, forecast, counterfactual.

Three operations, and the distinction between the first two is the whole idea.

**Estimate** (`run_filter`). An extended Kalman filter over the seven latent
variables. It consumes every measurement as it arrives and maintains a best
guess at the astronaut's hidden state with an honest covariance. This is what
the dashboard shows and what attribution reasons over.

**Forecast** (`open_loop`). The same dynamics, run forward from a state anchored
some days *back*, driven only by the logged flight-plan inputs, with the
intervening measurements deliberately withheld. This answers the question that
matters: given what we asked this person to do, what should they look like
today?

The gap between the two is the finding. A filter on its own cannot produce one,
because a filter's job is to explain the data -- give it a crew member whose HRV
is collapsing and it will dutifully raise its autonomic-load estimate and
report small, well-behaved residuals. It will be right about the state and
silent about the event. Only by refusing to look at the recent measurements can
the twin say *this is not what the plan predicted*, and only then does
`divergence.py` have something to score.

That also disposes of the hardest false positive in this domain. A crew member
given twelve days of short sleep and heavy workload deteriorates exactly like a
crew member who is becoming ill. A baseline monitor flags both. The open-loop
forecast was given the short sleep as an input, predicts the deterioration, and
raises nothing -- while still forecasting forward, because "expected" is not
the same as "fine".

**Counterfactual** (`counterfactual`, `ensemble`). Forward Monte Carlo over
parameter and process uncertainty, under a modified input schedule. This is the
operational payoff of having a model rather than a chart: it answers "if we
protect his sleep for five days and halve the training block, where does his
HRV be on Friday, and with what confidence" in a form a flight surgeon can act
on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    FIT_BOUNDS, INPUT_RANGE, NOMINAL_INPUT, N_STATE, PROCESS_SD, STATE, UIDX,
    Params, _clip_state, load_terms, settle, step,
)
from .observe import as_vector, jacobian, obs_sd, predict
from .signals import MONITORED

N_CH = len(MONITORED)

#: Added to every covariance diagonal before inversion. An EKF on a clipped,
#: bounded state will occasionally produce a numerically singular innovation
#: covariance; a monitor that raises a LinAlgError instead of an advisory has
#: failed in the least useful way available to it.
_JITTER = 1e-9


@dataclass(frozen=True)
class Window:
    """Everything the twin needs about a stretch of mission.

    Also the seam for real telemetry: fill this from a CSV and nothing
    downstream knows the difference. `phase` and `gravity` are flight-plan and
    mission-phase facts rather than measurements -- the twin is *told* that
    Thursday's sleep period was shifted two hours for a docking, because a
    monitor that has to infer the flight plan from the physiology will always
    read a busy week as a medical finding.
    """
    u: np.ndarray                  # (days, N_INPUT) as flight ops logged it
    y: np.ndarray                  # (days, N_CH) NaN where not measured
    dose_cum: np.ndarray           # (days,)
    phase: np.ndarray              # (days,) hours of circadian demand
    gravity: np.ndarray            # (days,) fractional headward fluid shift

    @property
    def days(self) -> int:
        return int(self.u.shape[0])

    def slice(self, a: int, b: int) -> "Window":
        return Window(self.u[a:b], self.y[a:b], self.dose_cum[a:b],
                      self.phase[a:b], self.gravity[a:b])


def window_from_arrays(u: np.ndarray, obs: dict[str, np.ndarray],
                       dose_cum: np.ndarray, phase: np.ndarray,
                       gravity: np.ndarray) -> Window:
    y = np.column_stack([obs[ch] for ch in MONITORED])
    return Window(u=np.asarray(u, float), y=np.asarray(y, float),
                  dose_cum=np.asarray(dose_cum, float),
                  phase=np.asarray(phase, float),
                  gravity=np.asarray(gravity, float))


# --- the filter ---------------------------------------------------------------

@dataclass
class FilterOut:
    x: np.ndarray        # (days, N_STATE) posterior mean
    P: np.ndarray        # (days, N_STATE, N_STATE) posterior covariance
    xp: np.ndarray       # (days, N_STATE) one-step prior mean
    Pp: np.ndarray       # (days, N_STATE, N_STATE)
    innov: np.ndarray    # (days, N_CH) y - h(prior), NaN where unmeasured
    innov_z: np.ndarray  # (days, N_CH) innovation / sqrt(diag S)
    nll: float           # negative predictive log-likelihood, up to a constant
    n_obs: int


def _state_jacobian(x: np.ndarray, u: np.ndarray, p: Params,
                    phase: float, gravity: float) -> np.ndarray:
    h = np.array([0.02, 0.25, 0.004, 0.004, 0.02, 0.06, 0.02])
    F = np.zeros((N_STATE, N_STATE))
    for j in range(N_STATE):
        xp_, xm_ = x.astype(float).copy(), x.astype(float).copy()
        xp_[j] += h[j]
        xm_[j] -= h[j]
        F[:, j] = (step(xp_, u, p, phase_shift=phase, gravity_shift=gravity)
                   - step(xm_, u, p, phase_shift=phase, gravity_shift=gravity)
                   ) / (2.0 * h[j])
    return F


def run_filter(w: Window, p: Params, *, sd_scale: dict[str, float] | None = None,
               x0: np.ndarray | None = None, P0: np.ndarray | None = None,
               ) -> FilterOut:
    """Extended Kalman filter over the latent state."""
    days = w.days
    R_diag = obs_sd(sd_scale) ** 2
    Q = np.diag(PROCESS_SD ** 2)

    x = settle(p, NOMINAL_INPUT, days=4000) if x0 is None else x0.astype(float).copy()
    # Initial uncertainty: a few days of process noise. The window starts in a
    # declared quiet stretch, so the settled state is a good guess, but
    # pretending it is exact makes the first week's residuals meaningless.
    P = (np.diag((PROCESS_SD * 3.0) ** 2) if P0 is None else P0.copy())

    xs = np.zeros((days, N_STATE))
    Ps = np.zeros((days, N_STATE, N_STATE))
    xps = np.zeros((days, N_STATE))
    Pps = np.zeros((days, N_STATE, N_STATE))
    innov = np.full((days, N_CH), np.nan)
    innov_z = np.full((days, N_CH), np.nan)
    nll = 0.0
    n_obs = 0

    for d in range(days):
        if d == 0:
            xp, Pp = x, P
        else:
            F = _state_jacobian(x, w.u[d - 1], p,
                                float(w.phase[d - 1]), float(w.gravity[d - 1]))
            xp = step(x, w.u[d - 1], p, phase_shift=float(w.phase[d - 1]),
                      gravity_shift=float(w.gravity[d - 1]))
            Pp = F @ P @ F.T + Q
        Pp = 0.5 * (Pp + Pp.T)

        xps[d], Pps[d] = xp, Pp

        y = w.y[d]
        have = np.where(np.isfinite(y))[0]
        yhat = as_vector(predict(xp, p, w.u[d], float(w.dose_cum[d])))

        if have.size == 0:
            x, P = xp, Pp
            xs[d], Ps[d] = x, P
            continue

        H = jacobian(xp, p, w.u[d])[have]
        R = np.diag(R_diag[have])
        v = y[have] - yhat[have]
        S = H @ Pp @ H.T + R
        S = 0.5 * (S + S.T) + _JITTER * np.eye(S.shape[0])

        innov[d, have] = v
        innov_z[d, have] = v / np.sqrt(np.clip(np.diag(S), 1e-12, None))

        try:
            Sinv_v = np.linalg.solve(S, v)
            K = Pp @ H.T @ np.linalg.inv(S)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                logdet = float(np.sum(np.log(np.clip(np.diag(S), 1e-12, None))))
        except np.linalg.LinAlgError:
            x, P = xp, Pp
            xs[d], Ps[d] = x, P
            continue

        nll += 0.5 * (float(v @ Sinv_v) + logdet)
        n_obs += int(have.size)

        x = _clip_state(xp + K @ v)
        IKH = np.eye(N_STATE) - K @ H
        P = IKH @ Pp @ IKH.T + K @ R @ K.T          # Joseph form, stays PSD
        P = 0.5 * (P + P.T)

        xs[d], Ps[d] = x, P

    return FilterOut(x=xs, P=Ps, xp=xps, Pp=Pps, innov=innov,
                     innov_z=innov_z, nll=nll, n_obs=n_obs)


# --- the forecast -------------------------------------------------------------

@dataclass
class OpenLoop:
    """A forecast that was not allowed to see the measurements it is judged on."""
    anchor: int                  # day the state was anchored at
    x: np.ndarray                # (h, N_STATE) predicted latent trajectory
    yhat: np.ndarray             # (h, N_CH) predicted measurements
    S: np.ndarray                # (h, N_CH) predictive sd per channel
    resid: np.ndarray            # (h, N_CH) observed - predicted, NaN if absent
    resid_z: np.ndarray          # (h, N_CH) standardised
    #: (h, N_CH, N_STATE). Carried out of the forecast because `divergence.py`
    #: needs it to ask which latent change would explain the residual, and
    #: recomputing it there would mean re-deriving the trajectory it belongs to.
    H: np.ndarray
    #: (h, N_STATE, N_STATE) forecast state covariance, which grows with the
    #: horizon as process noise accumulates. `divergence.py` needs it because
    #: the detection statistic is a Mahalanobis distance against exactly this
    #: spread -- that is what makes the statistic independent of how far back
    #: the forecast was anchored.
    P: np.ndarray
    reached: int                 # days actually produced (the window can end)


def open_loop(w: Window, p: Params, x_anchor: np.ndarray, P_anchor: np.ndarray,
              anchor: int, horizon: int, *,
              sd_scale: dict[str, float] | None = None) -> OpenLoop:
    """Propagate from `anchor` for `horizon` days on inputs alone.

    The predictive standard deviation grows with the horizon, because the
    forecast genuinely gets worse -- process noise compounds and the anchor was
    uncertain to begin with. Scoring a seven-day-old forecast against the same
    sigma as a one-day-old one is how a monitor ends up with a detection it
    cannot defend.
    """
    R_diag = obs_sd(sd_scale) ** 2
    Q = np.diag(PROCESS_SD ** 2)

    x, P = x_anchor.astype(float).copy(), P_anchor.copy()
    xs = np.zeros((horizon, N_STATE))
    yh = np.zeros((horizon, N_CH))
    sd = np.zeros((horizon, N_CH))
    res = np.full((horizon, N_CH), np.nan)
    resz = np.full((horizon, N_CH), np.nan)
    Hs = np.zeros((horizon, N_CH, N_STATE))
    Ps = np.zeros((horizon, N_STATE, N_STATE))
    reached = 0

    for i in range(horizon):
        d = anchor + 1 + i
        if d >= w.days:
            break
        reached = i + 1
        F = _state_jacobian(x, w.u[d - 1], p,
                            float(w.phase[d - 1]), float(w.gravity[d - 1]))
        x = step(x, w.u[d - 1], p, phase_shift=float(w.phase[d - 1]),
                 gravity_shift=float(w.gravity[d - 1]))
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        H = jacobian(x, p, w.u[d])
        pred = as_vector(predict(x, p, w.u[d], float(w.dose_cum[d])))
        var = np.clip(np.einsum("ij,jk,ik->i", H, P, H) + R_diag, 1e-12, None)

        xs[i] = x
        yh[i] = pred
        sd[i] = np.sqrt(var)
        Hs[i] = H
        Ps[i] = P

        y = w.y[d]
        m = np.isfinite(y)
        res[i, m] = y[m] - pred[m]
        resz[i, m] = res[i, m] / sd[i, m]

    return OpenLoop(anchor=anchor, x=xs, yhat=yh, S=sd, resid=res,
                    resid_z=resz, H=Hs, P=Ps, reached=reached)


# --- forward simulation under uncertainty -------------------------------------

@dataclass
class Ensemble:
    """Monte Carlo forward runs: the cone, not the line."""
    days: np.ndarray                 # (h,) absolute day index
    x: np.ndarray                    # (n, h, N_STATE)
    y: np.ndarray                    # (n, h, N_CH)

    def quantiles(self, ch: str, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
        j = MONITORED.index(ch)
        return np.quantile(self.y[:, :, j], qs, axis=0)

    def state_quantiles(self, name: str,
                        qs=(0.05, 0.5, 0.95)) -> np.ndarray:
        j = STATE.index(name)
        return np.quantile(self.x[:, :, j], qs, axis=0)

    def prob_below(self, ch: str, level: float) -> np.ndarray:
        j = MONITORED.index(ch)
        return (self.y[:, :, j] < level).mean(axis=0)

    def prob_above(self, ch: str, level: float) -> np.ndarray:
        j = MONITORED.index(ch)
        return (self.y[:, :, j] > level).mean(axis=0)


def ensemble(p: Params, x0: np.ndarray, P0: np.ndarray,
             u_plan: np.ndarray, *, start_day: int = 0, n: int = 400,
             phase: np.ndarray | None = None,
             gravity: np.ndarray | None = None,
             param_sd: dict[str, float] | None = None,
             seed: int = 11,
             sd_scale: dict[str, float] | None = None) -> Ensemble:
    """Run the twin forward `n` times under a planned input schedule.

    Three sources of uncertainty are sampled, and leaving any of them out makes
    the cone a decoration: the current state (`P0`, from the filter), the daily
    process noise, and -- where `calibrate` supplied standard errors -- the
    parameters themselves. The third is the one usually omitted and it
    dominates at longer horizons, because being wrong about how fast somebody
    recovers compounds while today's measurement error does not.
    """
    rng = np.random.default_rng(seed)
    h = int(u_plan.shape[0])
    phase = np.zeros(h) if phase is None else phase
    gravity = np.zeros(h) if gravity is None else gravity
    R = obs_sd(sd_scale)

    # A PSD square root of the anchor covariance, tolerant of the zero
    # eigenvalues a clipped state produces.
    vals, vecs = np.linalg.eigh(0.5 * (P0 + P0.T))
    L = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))

    xs = np.zeros((n, h, N_STATE))
    ys = np.zeros((n, h, N_CH))

    keys = tuple((param_sd or {}).keys())
    base = p.as_vector(keys) if keys else np.empty(0)
    sds = np.array([param_sd[k] for k in keys]) if keys else np.empty(0)

    for i in range(n):
        pi = p
        if keys:
            draw = base + rng.normal(0.0, sds)
            lo = np.array([FIT_BOUNDS.get(k, (-np.inf, np.inf))[0] for k in keys])
            hi = np.array([FIT_BOUNDS.get(k, (-np.inf, np.inf))[1] for k in keys])
            pi = p.with_vector(keys, np.clip(draw, lo, hi))

        x = _clip_state(x0 + L @ rng.normal(size=N_STATE))
        for t in range(h):
            if t > 0:
                x = _clip_state(
                    step(x, u_plan[t - 1], pi,
                         phase_shift=float(phase[t - 1]),
                         gravity_shift=float(gravity[t - 1]))
                    + rng.normal(0.0, PROCESS_SD))
            xs[i, t] = x
            ys[i, t] = as_vector(predict(x, pi, u_plan[t])) + rng.normal(0.0, R)

    return Ensemble(days=np.arange(start_day, start_day + h), x=xs, y=ys)


# --- counterfactuals ----------------------------------------------------------

def plan_matrix(horizon: int, base: np.ndarray | None = None) -> np.ndarray:
    """A flat input schedule to modify."""
    b = NOMINAL_INPUT if base is None else base
    return np.tile(np.asarray(b, float), (horizon, 1))


def apply_change(u_plan: np.ndarray, change: dict[str, float], *,
                 days: slice | None = None, absolute: bool = False
                 ) -> np.ndarray:
    """Return a copy of a plan with named inputs shifted (or set)."""
    out = u_plan.copy()
    sl = days or slice(None)
    for k, v in change.items():
        j = UIDX[k]
        out[sl, j] = v if absolute else out[sl, j] + v
        lo, hi = INPUT_RANGE[k]
        out[:, j] = np.clip(out[:, j], lo, hi)
    return out


@dataclass
class Counterfactual:
    label: str
    change: dict[str, float]
    baseline: Ensemble
    modified: Ensemble

    def delta(self, ch: str, q: float = 0.5) -> np.ndarray:
        a = np.quantile(self.modified.y[:, :, MONITORED.index(ch)], q, axis=0)
        b = np.quantile(self.baseline.y[:, :, MONITORED.index(ch)], q, axis=0)
        return a - b

    def recovery_day(self, ch: str, target: float, *, prob: float = 0.80,
                     rising: bool = True) -> int | None:
        """First forecast day on which `ch` clears `target` with probability
        at least `prob` under the modified plan. None if it does not."""
        j = MONITORED.index(ch)
        s = self.modified.y[:, :, j]
        pr = (s > target).mean(axis=0) if rising else (s < target).mean(axis=0)
        hit = np.where(pr >= prob)[0]
        return int(self.modified.days[hit[0]]) if hit.size else None


def counterfactual(p: Params, x0: np.ndarray, P0: np.ndarray, *,
                   horizon: int, change: dict[str, float], label: str,
                   start_day: int = 0, base_plan: np.ndarray | None = None,
                   n: int = 300, param_sd: dict[str, float] | None = None,
                   sd_scale: dict[str, float] | None = None,
                   seed: int = 11) -> Counterfactual:
    """Same twin, same starting state, two input schedules.

    The two ensembles share a seed on purpose. Comparing a 300-run cone against
    an independently drawn 300-run cone adds Monte Carlo noise to exactly the
    quantity being measured -- the difference -- and at these sample sizes that
    noise is comparable to the effect of a real countermeasure.
    """
    base = plan_matrix(horizon) if base_plan is None else base_plan.copy()
    mod = apply_change(base, change)
    kw = dict(start_day=start_day, n=n, param_sd=param_sd,
              sd_scale=sd_scale, seed=seed)
    return Counterfactual(
        label=label, change=change,
        baseline=ensemble(p, x0, P0, base, **kw),
        modified=ensemble(p, x0, P0, mod, **kw),
    )


def explain_load(x: np.ndarray, p: Params, u: np.ndarray) -> list[tuple[str, float]]:
    """The load index broken into named costs, largest first.

    The answer to "why is this person's autonomic load 1.3" in the only form
    that supports a decision: a ranked list of the things that are costing
    them, in the same units as the total.
    """
    terms = load_terms(x, p, u)
    return sorted(terms.items(), key=lambda kv: -kv[1])
