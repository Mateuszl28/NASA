"""Reading the gap between the twin and the astronaut.

The open-loop forecast in `twin.py` produces a residual: what the flight plan,
run through this person's own model, failed to predict. This module turns that
residual into three answers.

**Is anything happening?** Not by scoring channels one at a time. The residual
vector is projected into latent space -- the question asked is "what single
change in autonomic load, plasma volume, inflammation, sleep debt, fitness,
circadian phase or psychological reserve would best explain all twelve
channels at once", and the statistic is how much of the residual that change
accounts for. A coherent half-sigma movement across four channels that share a
driver projects strongly onto one latent direction; four independent
half-sigma wobbles do not project anywhere. This is the formal version of the
brief's observation that no single parameter looks alarming while the
combination does, and it is why detection lands days before any channel leaves
its own range.

**What is it?** The same projection names it. The fitted latent change is a
vector in a space whose axes are physiological systems, so the dominant
component *is* the finding: a drop in plasma volume with a rise in heart rate
and a fall in systolic pressure is one pattern, and the same heart-rate rise
with a temperature rise and a breathing-rate rise is a different one. A chart
cannot distinguish those. A residual resolved onto latent axes does.

**Whose fault is it?** Three candidate explanations are scored, not one:

  *the astronaut*  -- a latent change explains the residual, and it is confined
                      to this crew member.
  *the habitat*    -- several crew diverge on the same day in the same
                      direction. Four simultaneous findings with an
                      environmental signal that moved first are one event, not
                      four medical workups.
  *the instrument* -- the residual is confined to one device's channels and no
                      latent state explains it. A garment that loses skin
                      contact produces precisely what a collapsing autonomic
                      system produces; the discriminator is that a real
                      autonomic collapse also moves mass, mood and stress,
                      which come from other sources. Calling this one wrong
                      costs a medical workup and, the next time, the crew's
                      trust in the monitor.

Two mechanisms are worth calling out because they were not obvious.

*The anchor is frozen while a divergence is open.* The forecast is anchored at
the last day the twin could vouch for. If the anchor simply trailed the current
day by a fixed horizon, a sustained event would walk into the anchor after a
week, the forecast would be launched from an already-sick state, the residual
would collapse and the alarm would quietly clear while the astronaut was still
deteriorating. Freezing the anchor is what makes a persistent finding persist.

*The threshold is measured, not derived.* The statistic would be chi-squared if
the residuals were independent and Gaussian. They are not: physiological
measurement noise is autocorrelated, the state is clipped at physiological
bounds, and the model is wrong in ways `truth.py` deliberately arranges. So the
threshold comes from pooled nominal missions and a declared false-alarm budget
(`calibrate_threshold`), and the budget is reported with the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    IDX, NOMINAL_INPUT, N_STATE, STATE, STATE_NAMES, UIDX, Params,
)
from .observe import obs_sd
from .signals import ABSOLUTE_LIMITS, MONITORED, SIGNALS, signals_of
from .twin import FilterOut, OpenLoop, Window, open_loop, run_filter

#: Scale of a "unit" change in each latent variable, so that a shift of 1.0
#: means the same thing on every axis. Without it a shift is reported in
#: whatever unit the axis happens to use, and plasma volume (a few hundredths)
#: always looks negligible beside sleep debt (hours).
#:
#: Set to the size of a change a flight surgeon would want a sentence about --
#: a 5 % plasma-volume deficit, three hours of sleep debt -- rather than to
#: something merely measurable. The distinction matters because the prior in
#: `_project` is expressed in these units, so a scale set too small makes a
#: real event look like a many-sigma excursion and shrinks it hard. Plasma
#: volume was at 0.022, which put a genuine 14 % deficit five units out; the
#: projection shrank it below the autonomic response it had itself caused, and
#: the finding came back named after the symptom instead of the cause.
LATENT_SCALE: dict[str, float] = {
    "A": 0.25, "S": 3.0, "V": 0.055, "F": 0.045,
    "I": 0.18, "C": 1.1, "M": 0.12,
}

#: Prior precision on the latent shift, in units of LATENT_SCALE. See
#: `_project` -- it is what stops the fit explaining one channel's noise by
#: moving whichever latent axis that channel happens to observe.
RIDGE = 0.4

#: Forecast lengths scanned every day. Short catches a step, long catches a
#: drift; see the comment in `scan` for why one length is not enough.
HORIZONS: tuple[int, ...] = (3, 7, 14)

#: Named physiological pictures, as the latent direction each one implies.
#: Matched by cosine similarity against the fitted shift, so a finding is
#: described by the pattern it resembles rather than by its largest single
#: component -- "plasma volume loss" and "autonomic overload" overlap in HRV and
#: heart rate and are separated by what blood pressure does.
PATTERNS: tuple[tuple[str, dict[str, float], str], ...] = (
    ("Autonomic overload",
     {"A": 1.0, "S": 0.45, "M": -0.35},
     "Load accumulating faster than this person sheds it: HRV down, resting "
     "rate up, sleep efficiency falling, completed training dropping."),
    ("Sleep debt accumulation",
     {"S": 1.0, "A": 0.55, "M": -0.3},
     "Debt leading the autonomic response rather than following it."),
    ("Plasma volume loss",
     {"V": -1.0, "A": 0.30},
     "Fluid and electrolyte deficit: heart rate up to defend output, systolic "
     "pressure sagging, body mass falling with total body water."),
    ("Inflammatory / immune activation",
     {"I": 1.0, "A": 0.35},
     "Skin temperature and breathing rate up, SpO2 fractionally down, sleep "
     "fragmenting -- an inflammatory response before anybody feels ill."),
    ("Circadian disruption",
     {"C": 1.0, "S": 0.4, "A": 0.3},
     "Phase misalignment degrading sleep independently of its duration."),
    ("Aerobic deconditioning",
     {"F": -1.0},
     "Slow loss of capacity: a trend, not an event, and judged on weeks."),
    ("Behavioural / psychological strain",
     {"M": -1.0, "A": 0.3},
     "Reserve falling ahead of the autonomic signals. The tier crews "
     "under-report, which is why it is monitored and privacy-tiered."),
)


@dataclass
class DayScore:
    """What the twin makes of one crew member on one day."""
    crew: str
    day: int
    anchor: int
    horizon: int
    #: Residual sum of squares explained by the best single latent change.
    #: The detection statistic.
    explained_ss: float
    #: Residual left over after that change. Large means *no* state of this
    #: person explains the readings -- the instrument signature.
    leftover_ss: float
    dof_used: int
    score: float                       # explained_ss, per degree of freedom
    latent_push: dict[str, float]      # the fitted change, in native units
    latent_z: dict[str, float]         # the same, standardised
    channel_z: dict[str, float]        # per-channel residual z, latest day
    device_share: dict[str, float]     # share of residual SS per device
    device_base: dict[str, float]      # the same shares when nothing is wrong
    #: Mean squared standardised residual per device. Near 1.0 means that
    #: device is behaving; much more means it is not.
    device_q: dict[str, float]
    pattern: str
    pattern_note: str
    pattern_fit: float
    limit_breach: tuple[str, ...] = ()
    in_band: bool = True               # every channel still inside personal 3-sigma
    #: Fraction of the deviation the logged flight plan accounts for. Computed
    #: only for days above threshold, since it costs a second forecast.
    plan_share: float = 0.0

    @property
    def dominant(self) -> str:
        if not self.latent_z:
            return ""
        return max(self.latent_z, key=lambda k: abs(self.latent_z[k]))


@dataclass
class Finding:
    """A divergence episode: a run of days above threshold, reported once."""
    crew: str
    start: int
    end: int
    peak_day: int
    peak_score: float
    attribution: str
    pattern: str
    pattern_note: str
    latent_z: dict[str, float]
    in_band_at_start: bool
    limit_breach: tuple[str, ...]
    device_share: dict[str, float]
    plan_share: float = 0.0
    note: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start + 1


# --- standardising the residual ----------------------------------------------

@dataclass
class ResidCal:
    """Per-channel correction measured on the quiet window.

    The forecast's own predictive sigma is not good enough to standardise by,
    for two reasons that both inflate the statistic in the same direction.

    *Model misfit.* The twin is wrong about this person in small, systematic
    ways that calibration could not remove. Those show up as residuals a little
    larger than predicted, and a statistic that pools twelve channels over
    fourteen days turns "a little larger" into a detection.

    *Autocorrelation.* Physiology and sensors are persistent -- body mass
    carries a lag-1 correlation near 0.7 -- while the twin treats measurement
    error as white. Fourteen correlated days are nothing like fourteen
    independent ones, so pooling them as if they were counted the same evidence
    several times. With body mass that was enough on its own: a 0.3 kg offset
    held for a fortnight projected cleanly onto plasma volume and produced a
    confident, entirely false "fluid loss" finding on a healthy crew member.

    So both are measured on exactly the statistic the detector consumes --
    open-loop residuals over the quiet window, same anchor and horizon
    mechanics -- and applied as a sigma scale and an effective-sample-size
    weight. Measuring rather than assuming also means a channel that behaves
    better than expected gets credit for it.
    """
    #: sd_curve[channel] -> array over forecast step. Not a single number per
    #: channel: the twin's error does not merely exceed its predicted sigma, it
    #: exceeds it by a margin that *grows along the forecast*, because the
    #: unmodelled parts of the physiology accumulate while the EKF's process
    #: noise assumes they do not. A flat per-channel ratio therefore
    #: under-corrects at long horizons and the statistic ends up measuring the
    #: horizon: the same healthy commander scored 4.3 at a three-day forecast
    #: and 14.5 at a fourteen-day one. Calibrating per step removes it.
    sd_curve: dict[str, np.ndarray] = field(default_factory=dict)
    rho: dict[str, float] = field(default_factory=dict)
    horizon: int = 14
    #: The quiet-window distribution of the raw statistic, per forecast
    #: horizon. This is what makes the score comparable across horizons, and
    #: it is measured rather than derived -- see `scan`.
    null_mean: dict[int, float] = field(default_factory=dict)
    null_sd: dict[int, float] = field(default_factory=dict)
    #: Each device's share of the residual sum of squares when nothing is
    #: wrong. Needed because the shares are wildly uneven by construction:
    #: Bio-Monitor carries nine of the twelve monitored channels, so its share
    #: sits near 0.8 on a perfectly healthy day. Testing against an absolute
    #: 0.8 therefore called almost every finding an instrument fault --
    #: including the schedule-driven one. Concentration only means something
    #: measured against what that device normally contributes.
    device_base: dict[str, float] = field(default_factory=dict)

    def studentise(self, raw: float, h: int) -> float:
        """Nominal standard deviations above this person's own quiet-window
        divergence at the same forecast horizon."""
        if not self.null_mean:
            return raw
        hs = min(self.null_mean, key=lambda k: abs(k - h))
        mu = self.null_mean[hs]
        sd = max(self.null_sd.get(hs, 1.0), 1e-6)
        return (raw - mu) / sd

    def step_scales(self, i: int) -> np.ndarray:
        """Sigma multiplier per channel at forecast step `i`."""
        out = np.ones(len(MONITORED))
        for j, ch in enumerate(MONITORED):
            cur = self.sd_curve.get(ch)
            if cur is None or len(cur) == 0:
                continue
            out[j] = float(cur[min(i, len(cur) - 1)])
        return np.clip(out, 0.3, 8.0)

    def weights(self) -> np.ndarray:
        """Row weight per channel: the effective-sample-size correction.

        For an AR(1) sequence the variance of a mean over n samples is inflated
        by (1+rho)/(1-rho), so weighting each row by the square root of the
        reciprocal makes a pooled sum of squares behave like one over
        independent samples.
        """
        out = np.ones(len(MONITORED))
        for j, ch in enumerate(MONITORED):
            r = float(np.clip(self.rho.get(ch, 0.0), 0.0, 0.95))
            out[j] = np.sqrt((1.0 - r) / (1.0 + r))
        return out

    def mean_scale(self, ch: str) -> float:
        cur = self.sd_curve.get(ch)
        return float(np.mean(cur)) if cur is not None and len(cur) else 1.0


#: Shrinkage weight for the per-step sigma curve. A 45-day quiet window yields
#: about 30 forecasts, so each (channel, step) cell has ~30 samples -- enough to
#: see a trend, not enough to trust a single cell. Each is pulled toward that
#: channel's own average across steps with the weight of 8 pseudo-observations.
_CURVE_SHRINK = 8.0


def fit_resid_cal(w: Window, p: Params, *, sd_scale: dict[str, float] | None,
                  fo: FilterOut, calib_days: int, horizon: int = 14,
                  min_h: int = 3) -> ResidCal:
    """Measure forecast-error growth and persistence over the quiet window.

    Runs exactly the forecasts the detector runs -- same anchor mechanics, same
    horizon -- so the correction is calibrated on the statistic it corrects
    rather than on a proxy for it.
    """
    per_step: dict[str, list[list[float]]] = {
        ch: [[] for _ in range(horizon)] for ch in MONITORED}
    chunks: dict[str, list[np.ndarray]] = {ch: [] for ch in MONITORED}

    for day in range(horizon + 1, min(calib_days, w.days)):
        anchor = day - horizon
        ol = open_loop(w, p, fo.x[anchor], fo.P[anchor], anchor, horizon,
                       sd_scale=sd_scale)
        if ol.reached == 0:
            continue
        for j, ch in enumerate(MONITORED):
            col = ol.resid_z[:ol.reached, j]
            for i, v in enumerate(col):
                if np.isfinite(v) and i < horizon:
                    per_step[ch][i].append(float(v))
            if np.isfinite(col).sum() >= 4:
                chunks[ch].append(col)

    sd_curve, rho = {}, {}
    for ch in MONITORED:
        flat = [v for cell in per_step[ch] for v in cell]
        overall = float(np.std(flat, ddof=1)) if len(flat) > 3 else 1.0
        overall = overall if np.isfinite(overall) and overall > 0 else 1.0

        cur = np.empty(horizon)
        for i in range(horizon):
            cell = per_step[ch][i]
            n = len(cell)
            s = float(np.std(cell, ddof=1)) if n > 3 else overall
            s = s if np.isfinite(s) and s > 0 else overall
            cur[i] = (n * s + _CURVE_SHRINK * overall) / (n + _CURVE_SHRINK)
        sd_curve[ch] = np.clip(cur, 0.3, 8.0)

        num = den = 0.0
        for c in chunks[ch]:
            m = np.isfinite(c)
            if m.sum() < 4:
                continue
            v = c[m] - np.mean(c[m])
            num += float(np.sum(v[:-1] * v[1:]))
            den += float(np.sum(v * v))
        rho[ch] = float(np.clip(num / den, 0.0, 0.95)) if den > 0 else 0.0

    rc = ResidCal(sd_curve=sd_curve, rho=rho, horizon=horizon)

    # Second pass: the quiet-window distribution of the raw statistic, for
    # every horizon the scan can use. Needed because the statistic is a sum
    # over (day, channel) residuals and so grows with the forecast length --
    # not from any miscalibration, but because a longer open-loop run
    # accumulates more genuine process noise and the projection is built to
    # find exactly that drift. Deriving the growth analytically means modelling
    # the full joint covariance of the residuals across days; measuring it on
    # the same person's own quiet stretch costs one pass and makes the score
    # mean something a reviewer can check: nominal standard deviations above
    # this crew member's own usual divergence at the same horizon.
    for h in range(min_h, horizon + 1):
        vals = []
        for day in range(h + 1, min(calib_days, w.days)):
            anchor = day - h
            if anchor < 0:
                continue
            ol = open_loop(w, p, fo.x[anchor], fo.P[anchor], anchor, h,
                           sd_scale=sd_scale)
            if ol.reached == 0:
                continue
            _, expl, _, _, _ = _project(ol, rc)
            vals.append(expl)
        if len(vals) >= 6:
            rc.null_mean[h] = float(np.mean(vals))
            rc.null_sd[h] = float(max(np.std(vals, ddof=1), 1e-3))

    # Baseline device shares, from the same quiet forecasts.
    acc: dict[str, float] = {}
    for day in range(horizon + 1, min(calib_days, w.days)):
        anchor = day - horizon
        ol = open_loop(w, p, fo.x[anchor], fo.P[anchor], anchor, horizon,
                       sd_scale=sd_scale)
        if ol.reached == 0:
            continue
        for k, v in _device_share(ol, rc).items():
            acc[k] = acc.get(k, 0.0) + v
    tot = sum(acc.values())
    if tot > 0:
        rc.device_base = {k: v / tot for k, v in acc.items()}

    return rc


# --- the statistic ------------------------------------------------------------

def _project(ol: OpenLoop, rcal: ResidCal | None = None
             ) -> tuple[np.ndarray, float, float, int, float]:
    """Best single latent change explaining the whole forecast window.

    Stacks every available (day, channel) residual, standardised by its own
    predictive sigma, and least-squares-solves for one constant latent shift.
    Constant rather than per-day on purpose: a per-day fit has as many
    parameters as data and explains noise perfectly, while a sustained
    physiological change is exactly a step that persists. The cost is reduced
    sensitivity to a one-day spike, which the absolute-limit check covers.
    """
    wt = rcal.weights() if rcal else np.ones(len(MONITORED))

    rows, rhs = [], []
    for i in range(ol.reached):
        m = np.isfinite(ol.resid[i])
        if not m.any():
            continue
        # Standardise by the measured forecast error at *this* step of the
        # forecast, then weight down the channels whose successive days are not
        # independent evidence.
        scale = (rcal.step_scales(i) if rcal else np.ones(len(MONITORED)))
        s = ol.S[i][m] * scale[m]
        k = wt[m]
        rows.append((ol.H[i][m] / s[:, None]) * k[:, None])
        rhs.append((ol.resid[i][m] / s) * k)

    if not rows:
        return np.zeros(N_STATE), 0.0, 0.0, 0, 0.5

    A = np.vstack(rows)
    b = np.concatenate(rhs)
    n = int(b.size)

    # Columns scaled by LATENT_SCALE so a unit of `g` means "a latent change
    # worth mentioning" in every direction, then ridged at unit strength --
    # which is not a numerical fudge but the prior that makes the statistic
    # mean what it claims.
    #
    # Without it, the fit is free to explain any single channel's noise by
    # moving whichever latent axis that channel happens to observe. Two axes
    # were effectively single-observer, and the result was a confident,
    # sustained, entirely false "psychological strain" finding on a healthy
    # commander, driven by nothing but a run of mood scores. A unit prior
    # shrinks a direction in proportion to how weakly the data constrain it, so
    # a coherent movement across four channels that share a driver survives
    # while a one-channel excursion does not.
    sc = np.array([LATENT_SCALE[k] for k in STATE])
    As = A * sc
    G = As.T @ As
    Mr = G + RIDGE * np.eye(N_STATE)
    g = np.linalg.solve(Mr, As.T @ b)

    # Effective degrees of freedom: the trace of the hat matrix, i.e. how many
    # independent directions the data actually pinned down. Normalising by a
    # constant seven instead let the statistic grow with the forecast horizon,
    # because a longer window constrains more directions -- so the same healthy
    # crew member scored 2.0 at a three-day horizon and 9.3 at fourteen, and the
    # threshold was measuring the horizon rather than the astronaut.
    dof_eff = float(np.trace(np.linalg.solve(Mr, G)))

    fit = As @ g
    total_ss = float(b @ b)
    leftover = float((b - fit) @ (b - fit))
    explained = max(total_ss - leftover, 0.0)
    return g * sc, explained, leftover, n, max(dof_eff, 0.5)


def _resid_norm(ol: OpenLoop, rcal: ResidCal | None = None) -> float:
    """Weighted residual norm of a forecast, over its whole window."""
    wt = rcal.weights() if rcal else np.ones(len(MONITORED))
    tot = 0.0
    for i in range(ol.reached):
        m = np.isfinite(ol.resid[i])
        if not m.any():
            continue
        scale = rcal.step_scales(i) if rcal else np.ones(len(MONITORED))
        z = (ol.resid[i][m] / (ol.S[i][m] * scale[m])) * wt[m]
        tot += float(z @ z)
    return float(np.sqrt(tot))


def plan_share(w: Window, p: Params, fo: FilterOut, day: int, horizon: int, *,
               sd_scale: dict[str, float] | None, rcal: ResidCal | None,
               ) -> float:
    """How much of the deviation the flight plan itself accounts for.

    Two forecasts from the same anchor: one driven by the inputs flight ops
    actually logged, one driven by an ordinary nominal day repeated. If the
    logged schedule explains the readings, its residual is much the smaller of
    the two and this returns a number near 1. If the schedule was unremarkable,
    the two forecasts coincide and it returns 0.

    This exists because the clean binary it replaces turned out to be a lie.
    The intent was that a crew member ground down by twelve days of short sleep
    and heavy workload would produce no finding at all, since the twin was
    given the short sleep as an input. In practice it produces a real one --
    the simulated body amplifies its own strain once load is already high, and
    a linear load index cannot express that, so the astronaut genuinely is
    worse than even a punishing schedule predicts. Suppressing that would be
    suppressing a true signal.

    The honest output is therefore not "medical finding / no finding" but
    *whose problem it is*: at 178 versus 20 on the same statistic, and with the
    plan accounting for most of the smaller one, the first is a body to examine
    and the second is a schedule to change. They need different actions, and a
    monitor that collapses them into one word has thrown away the distinction
    that decides which.
    """
    anchor = day - horizon
    if anchor < 0:
        return 0.0

    logged = open_loop(w, p, fo.x[anchor], fo.P[anchor], anchor, horizon,
                       sd_scale=sd_scale)

    w_nom = Window(u=np.tile(NOMINAL_INPUT, (w.days, 1)), y=w.y,
                   dose_cum=w.dose_cum, phase=w.phase, gravity=w.gravity)
    nominal = open_loop(w_nom, p, fo.x[anchor], fo.P[anchor], anchor, horizon,
                        sd_scale=sd_scale)

    r_log = _resid_norm(logged, rcal)
    r_nom = _resid_norm(nominal, rcal)
    if r_nom <= 1e-9:
        return 0.0
    return float(np.clip(1.0 - r_log / r_nom, 0.0, 1.0))


def _device_stats(ol: OpenLoop, rcal: ResidCal | None = None
                  ) -> tuple[dict[str, float], dict[str, float]]:
    """Residual share and mean squared residual, per physical device.

    The share says how the residual splits; the mean square says how loud each
    device is *in absolute terms*. The second is what distinguishes a broken
    sensor, and it is only meaningful because the residuals are standardised
    per channel per forecast step against the quiet window -- so a device with
    nothing wrong sits near 1.0 whatever its channel count.
    """
    wt = rcal.weights() if rcal else np.ones(len(MONITORED))
    acc: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for i in range(ol.reached):
        m = np.isfinite(ol.resid[i])
        scale = (rcal.step_scales(i) if rcal else np.ones(len(MONITORED)))
        z = ((ol.resid[i][m] / (ol.S[i][m] * scale[m])) * wt[m]) ** 2
        for ch, v in zip(np.array(MONITORED)[m], z):
            dev = SIGNALS[str(ch)].device
            acc[dev] = acc.get(dev, 0.0) + float(v)
            cnt[dev] = cnt.get(dev, 0) + 1
    tot = sum(acc.values())
    if tot <= 0.0:
        return {}, {}
    share = {k: v / tot for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
    mean_sq = {k: acc[k] / max(cnt[k], 1) for k in acc}
    return share, mean_sq


def _device_share(ol: OpenLoop, rcal: ResidCal | None = None) -> dict[str, float]:
    return _device_stats(ol, rcal)[0]


def _match_pattern(latent_z: dict[str, float]) -> tuple[str, str, float]:
    v = np.array([latent_z.get(k, 0.0) for k in STATE])
    nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return "No coherent pattern", "", 0.0
    v = v / nv
    best, note, score = "Unclassified", "", 0.0
    for name, sig, text in PATTERNS:
        s = np.array([sig.get(k, 0.0) for k in STATE])
        s = s / np.linalg.norm(s)
        cos = float(v @ s)
        if cos > score:
            best, note, score = name, text, cos
    if score < 0.45:
        return ("Mixed / unclassified",
                "No single named picture fits; the dominant axis is "
                f"{STATE_NAMES[max(latent_z, key=lambda k: abs(latent_z[k]))].lower()}.",
                score)
    return best, note, score


def personal_bands(w: Window, calib_days: int,
                   sd_scale: dict[str, float] | None = None,
                   k: float = 3.0) -> dict[str, tuple[float, float]]:
    """Each channel's own in-flight mean plus or minus `k` sigma.

    This is the straw man, built fairly: it is exactly what a well-implemented
    personal-baseline threshold monitor would use. Claiming detection "before
    any reading left its normal range" is only worth saying if the range is the
    one a competent alternative would have drawn.
    """
    out = {}
    for j, ch in enumerate(MONITORED):
        col = w.y[:calib_days, j]
        col = col[np.isfinite(col)]
        if col.size < 5:
            out[ch] = (-np.inf, np.inf)
            continue
        mu = float(np.mean(col))
        sd = float(np.std(col, ddof=1))
        sd = max(sd, obs_sd(sd_scale)[j] * 0.5)
        out[ch] = (mu - k * sd, mu + k * sd)
    return out


def _limit_breach(w: Window, day: int) -> tuple[str, ...]:
    """Protocol limits that hold regardless of whose body it is.

    The handful of places a personal baseline is the wrong instrument, because
    the number is about survivability rather than about deviation from habit.
    Dose rate is read from the inputs rather than the measurement matrix: it is
    the environment acting on the astronaut, so it enters the twin as a driver
    and is not one of the monitored channels. Checking it only against
    `MONITORED` meant the solar-particle limit could never fire at all -- the
    one event in the set that has to be called on its first day.
    """
    hits = []
    for ch, lim in ABSOLUTE_LIMITS.items():
        if ch == "dose_rate":
            v = float(w.u[day, UIDX["dose"]])
        elif ch in MONITORED:
            v = float(w.y[day, MONITORED.index(ch)])
        else:
            continue
        if not np.isfinite(v):
            continue
        if SIGNALS[ch].adverse == "low":
            if v <= lim["alert"]:
                hits.append(ch)
        elif v >= lim["alert"]:
            hits.append(ch)
    return tuple(hits)


# --- the scan -----------------------------------------------------------------

def scan(w: Window, p: Params, *, crew: str = "?",
         sd_scale: dict[str, float] | None = None,
         fo: FilterOut | None = None, rcal: ResidCal | None = None,
         threshold: float = 2.6, min_h: int = 3, max_h: int = 14,
         calib_days: int = 45) -> list[DayScore]:
    """Score every day of the window.

    `fo` is a filter pass over the same window; it is accepted so callers who
    already have one do not pay for it twice. Only `fo.x[anchor]` and
    `fo.P[anchor]` are read, and the filter at day `anchor` has seen no data
    after `anchor` -- so nothing from the forecast window leaks into the state
    the forecast starts from.
    """
    if fo is None:
        fo = run_filter(w, p, sd_scale=sd_scale)
    if rcal is None:
        rcal = fit_resid_cal(w, p, sd_scale=sd_scale, fo=fo,
                             calib_days=calib_days, horizon=max_h)

    bands = personal_bands(w, calib_days, sd_scale)
    out: list[DayScore] = []

    for day in range(calib_days, w.days):
        # Several forecast lengths, every day, and the strongest wins.
        #
        # One horizon cannot do this job, because deterioration does not come
        # at one speed. The autonomic overload case develops over three days
        # and is loudest in a short window. A plasma-volume deficit develops
        # over twelve, so any three-day window contains a quarter of its
        # change and -- worse -- the filtered anchor three days back has
        # already absorbed part of it, leaving the forecast predicting the
        # drift it was supposed to detect. At a three-day horizon the 14 %
        # volume loss scored 5.4 and was missed; the same event at fourteen
        # days is unmistakable. Each horizon carries its own measured null, so
        # the studentised scores are directly comparable and the maximum is
        # meaningful rather than just the largest number available.
        best: tuple[float, OpenLoop, int, np.ndarray, float, float, int] | None = None
        for h in HORIZONS:
            if h < min_h or h > max_h:
                continue
            anchor = day - h
            if anchor < 0:
                continue
            ol_h = open_loop(w, p, fo.x[anchor], fo.P[anchor], anchor, h,
                             sd_scale=sd_scale)
            if ol_h.reached == 0:
                continue
            push_h, expl_h, left_h, n_h, k_h = _project(ol_h, rcal)
            sc_h = rcal.studentise(expl_h, h)
            if best is None or sc_h > best[0]:
                best = (sc_h, ol_h, h, push_h, expl_h, left_h, n_h)

        if best is None:
            continue
        score, ol, horizon, push, expl, left, n = best
        anchor = day - horizon
        last = ol.reached - 1
        k_dirs = N_STATE

        latent_push = {k: float(push[IDX[k]]) for k in STATE}
        latent_z = {k: latent_push[k] / LATENT_SCALE[k] for k in STATE}
        pattern, note, fit = _match_pattern(latent_z)

        dev_share, dev_q = _device_stats(ol, rcal)
        rs = rcal.step_scales(last)
        ch_z = {ch: float(ol.resid_z[last, j] / rs[j])
                for j, ch in enumerate(MONITORED)
                if np.isfinite(ol.resid_z[last, j])}

        band_ok = True
        for ch, (lo, hi) in bands.items():
            v = w.y[day, MONITORED.index(ch)]
            if np.isfinite(v) and not (lo <= v <= hi):
                band_ok = False
                break

        share = 0.0
        if score >= threshold:
            share = plan_share(w, p, fo, day, horizon,
                               sd_scale=sd_scale, rcal=rcal)

        ds = DayScore(
            crew=crew, day=day, anchor=anchor, horizon=horizon,
            explained_ss=float(expl), leftover_ss=float(left),
            dof_used=int(round(k_dirs)),
            score=float(score), latent_push=latent_push, latent_z=latent_z,
            channel_z=ch_z,
            device_share=dev_share, device_q=dev_q,
            device_base=dict(rcal.device_base),
            pattern=pattern, pattern_note=note, pattern_fit=float(fit),
            limit_breach=_limit_breach(w, day), in_band=band_ok,
            plan_share=share,
        )
        out.append(ds)

    return out


# --- attribution and episode assembly ----------------------------------------

#: Share of the deviation the logged plan must account for before the finding
#: is called the schedule's rather than the body's.
_SCHEDULE_SHARE = 0.45

#: What a protocol-limit breach is called, and why it names the finding itself.
#:
#: A limit breach is detected on its first day, by design -- a dose rate
#: thirteen times its alert has to be called immediately. But on that first day
#: the physiology has not responded yet, so the residual projection has nothing
#: to work with and names the finding after whichever latent axis the day's
#: noise happened to favour. The solar particle event came back labelled
#: "behavioural / psychological strain", which is not a harmless cosmetic slip:
#: it is the system inventing a physiological story for an environmental fact.
LIMIT_PATTERNS: dict[str, tuple[str, str]] = {
    "dose_rate": (
        "Radiation exposure above protocol limit",
        "Dose rate past its protocol alert. Environmental, immediate, and not "
        "a judgement about anybody's body -- a personal baseline is the wrong "
        "instrument here, because the number is about exposure rather than "
        "about deviation from habit. Any physiological response follows over "
        "the next days and is tracked separately."),
    "spo2": (
        "Oxygen saturation below protocol limit",
        "SpO2 past its protocol alert. Check the habitat air before the crew "
        "member."),
    "skin_temp": (
        "Skin temperature in febrile range",
        "Past the protocol alert. Skin temperature is a proxy for core "
        "temperature, so this warrants a direct measurement rather than a "
        "conclusion."),
}

#: How many of an episode's highest-scoring days decide its name. See the
#: comment in `findings`: too few and one noisy day renames it, too many and
#: the aftermath dilutes the signature.
_NAME_DAYS = 5

#: Mean squared standardised residual above which a device counts as loud, and
#: below which every other device counts as quiet. Both are required.
_DEVICE_LOUD = 3.0
_DEVICE_QUIET = 1.8


def _attribute(ds: DayScore, concurrent: int, env_limit: bool,
               coherent: bool, plan_share_episode: float) -> str:
    """The astronaut, the schedule, the habitat, or the instrument.

    `plan_share_episode` is the episode's value, not this day's. Attribution is
    a claim about the whole episode, and reading it off the single peak day got
    the schedule case wrong: the episode's plan share was 0.60, comfortably
    over the threshold, while the peak day's happened to sit under it.
    """
    if env_limit:
        return "environment"

    # Several crew at once is only an environmental claim if they are doing the
    # *same thing*. Requiring a shared dominant axis matters because the
    # alternative double-counts: with a couple of unrelated false findings open
    # elsewhere on the roster, a genuine single-crew-member event got relabelled
    # environmental purely because three people were flagged the same day.
    if concurrent >= 3 and coherent:
        return "environment"

    # The schedule is checked before the instrument. A crew member being
    # ground down by a logged flight plan also concentrates its residual in
    # Bio-Monitor channels, because that is where nine of the twelve channels
    # live -- so asking "is it the sensor" first answered yes to a finding
    # whose cause the system had already correctly identified as the roster.
    if plan_share_episode >= _SCHEDULE_SHARE:
        return "schedule"

    # Instrument: one device is loud and every other device is quiet.
    #
    # The discriminator went through two wrong versions worth recording. The
    # first asked whether the residual was concentrated on one device, which
    # fails because Bio-Monitor carries nine of twelve channels and so holds
    # ~80 % of any residual on a perfectly healthy day. The second added "and
    # no coherent latent state explains it", which worked only while the
    # latent prior was tight enough to prevent a good fit -- seven free latent
    # dimensions against twelve channels can fit almost anything, so once the
    # prior was loosened to let a real plasma-volume deficit through, a
    # collapsed electrode became "explainable" too and the rule went silent.
    #
    # What actually separates them is physiological rather than statistical. A
    # body in autonomic collapse moves its mood, its perceived stress and its
    # mass -- channels that come from a questionnaire and an inertial device,
    # not from the garment. An electrode that has lost skin contact cannot
    # touch any of them. So the test is for *other devices staying quiet*,
    # which is only askable because the residuals are standardised per channel
    # per step: a device with nothing wrong sits near 1.0 regardless of how
    # many channels it carries.
    if ds.device_q:
        ranked = sorted(ds.device_q.items(), key=lambda kv: -kv[1])
        dev, loud = ranked[0]
        others = [q for _, q in ranked[1:]]
        if loud > _DEVICE_LOUD and (not others or max(others) < _DEVICE_QUIET):
            return "instrument"

    return "individual"


def findings(scans: dict[str, list[DayScore]], *, threshold: float = 2.6,
             gap: int = 2, min_hot: int = 3) -> list[Finding]:
    """Collapse per-day scores into episodes, attributed.

    Episodes rather than days, because the cost is paid per thing the crew has
    to read and answer. A stable advisory running for five days is one item on
    somebody's morning list, not five, and counting it five times makes a
    false-alarm rate look five times worse than it is while making a real
    finding look like a flood.

    `min_hot` is the persistence requirement, and it buys more than any amount
    of threshold tuning. Physiology that is actually changing stays above the
    line for days running; the nominal tail is made of isolated spikes. On the
    measured scenarios a healthy crew member's worst day reaches the same score
    as a real event's third day -- so the threshold alone cannot separate them,
    and demanding three hot days inside one episode does, without delaying a
    real finding, because a real finding was going to produce them anyway.

    A protocol-limit breach is exempt. A solar particle event has to be called
    on its first day, and waiting three days for persistence before mentioning
    a dose rate two orders of magnitude above background is not caution.
    """
    above = {c: {d.day for d in ss if d.score >= threshold}
             for c, ss in scans.items()}
    env_days = {c: {d.day for d in ss if d.limit_breach}
                for c, ss in scans.items()}
    # Limit-breach days seed episodes in their own right. Seeding only from
    # days above the divergence threshold meant a dose rate thirteen times its
    # protocol alert raised nothing at all unless the crew's physiology
    # happened to have reacted yet -- which, for radiation, it will not have.
    hot = {c: above[c] | env_days[c] for c in scans}

    by_day: dict[int, int] = {}
    dom_by_day: dict[int, list[str]] = {}
    for c, ss in scans.items():
        idx_c = {d.day: d for d in ss}
        for d in above[c]:
            by_day[d] = by_day.get(d, 0) + 1
            dom_by_day.setdefault(d, []).append(idx_c[d].dominant)

    out: list[Finding] = []
    for crew, ss in scans.items():
        idx = {d.day: d for d in ss}
        days = sorted(hot[crew])
        if not days:
            continue
        runs: list[list[int]] = [[days[0]]]
        for d in days[1:]:
            if d - runs[-1][-1] <= gap + 1:
                runs[-1].append(d)
            else:
                runs.append([d])

        for run in runs:
            breached = bool(env_days[crew] & set(run))
            if len(run) < min_hot and not breached:
                continue
            peak = max(run, key=lambda d: idx[d].score)
            pk = idx[peak]
            doms = dom_by_day.get(peak, [])
            coherent = bool(doms) and max(
                doms.count(x) for x in set(doms)) >= 3
            share_ep = max(idx[d].plan_share for d in run)
            attr = _attribute(pk, by_day.get(peak, 1), breached, coherent,
                              share_ep)
            breaches = tuple(sorted({b for d in run for b in idx[d].limit_breach}))

            # Name the episode from the whole run, score-weighted, not from the
            # single peak day. Taking the peak day's label let one outlying
            # mood score rename a correctly-identified inflammatory response as
            # "behavioural strain" -- the two days either side of the peak had
            # it right. An episode is a claim about a stretch of physiology, so
            # the evidence for its name should be the stretch.
            core = sorted(run, key=lambda d: -idx[d].score)[:_NAME_DAYS]
            acc = {k: 0.0 for k in STATE}
            wsum = 0.0
            for d in core:
                wt = max(idx[d].score, 0.0)
                wsum += wt
                for k in STATE:
                    acc[k] += wt * idx[d].latent_z.get(k, 0.0)
            mean_z = ({k: v / wsum for k, v in acc.items()} if wsum > 0
                      else pk.latent_z)
            pattern, note, _ = _match_pattern(mean_z)

            # A breach names the finding after itself. See LIMIT_PATTERNS.
            for b in breaches:
                if b in LIMIT_PATTERNS:
                    pattern, note = LIMIT_PATTERNS[b]
                    break

            out.append(Finding(
                crew=crew, start=run[0], end=run[-1], peak_day=peak,
                peak_score=pk.score, attribution=attr, pattern=pattern,
                pattern_note=note, latent_z=mean_z,
                in_band_at_start=idx[run[0]].in_band,
                limit_breach=breaches, device_share=pk.device_share,
                plan_share=share_ep,
            ))
    return sorted(out, key=lambda f: (f.start, f.crew))
