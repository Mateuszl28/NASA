"""From latent state to the twelve numbers a Bio-Monitor actually produces.

This is the only place the latent variables touch measurable reality, which
makes it the only place the model can be checked against a sensor. Two
properties are deliberate.

**It is many-to-few.** Twelve monitored channels are driven by seven latent
variables, and several channels share a driver -- HRV, resting heart rate,
sleep efficiency and achieved activity all respond to autonomic load, with
different gains and different signs. That redundancy is the whole reason a
coherent 1-sigma movement across four channels is strong evidence while four
independent 1-sigma movements are noise.

**It is differentiable and locally linear enough to invert.** `jacobian()`
gives the matrix that `divergence.attribute` least-squares-solves to answer the
question a chart cannot: *which latent system would have to move, and by how
much, to produce the residual pattern I am seeing?* That is how a residual
becomes "plasma volume" rather than "four channels look odd".
"""

from __future__ import annotations

import numpy as np

from .model import (
    IDX, N_STATE, Params, UIDX,
    achieved_activity, achieved_sleep, sleep_efficiency,
)
from .signals import MONITORED, SIGNALS


#: How much of the deficit-side gain the surplus side carries. Volume loading
#: is genuinely not the mirror image of volume loss -- a litre over does not
#: raise stroke volume the way a litre under lowers it -- but an exactly
#: one-sided response makes plasma volume *unobservable* above baseline except
#: through body mass. That left a latent axis with a single noisy observer, and
#: `divergence.py` would then happily explain any persistent body-mass wobble
#: as a volume change. A damped but non-zero surplus response keeps the
#: asymmetry and restores two-sided observability.
_SURPLUS_GAIN = 0.35

#: Width of the smooth transition between the deficit and surplus slopes, in
#: fractional plasma volume. Rounding the corner matters more than it looks:
#: the hard `max(0, -V)` form put a kink exactly at V = 0, which is exactly
#: where a healthy astronaut sits. Both the filter's state Jacobian and the
#: residual projection linearise there, and at a kink the derivative they get
#: depends on the finite-difference step rather than on the physiology -- the
#: two disagreed by a factor of 1.75. A tenth of a "worth mentioning" volume
#: change is narrow enough to leave the asymmetry intact at any clinically
#: interesting deficit and wide enough that nothing linearises on a corner.
_VOL_SMOOTH = 0.012


def vol_effect(V: float) -> float:
    """Signed plasma-volume effect, positive when depleted.

    Volume loading is genuinely not the mirror image of volume loss -- a litre
    over does not raise stroke volume the way a litre under lowers it -- so the
    two sides carry different gains, blended smoothly rather than switched.

    A strictly one-sided response was the first version and it made plasma
    volume *unobservable* above baseline except through body mass, leaving a
    latent axis with a single noisy observer that the projection would happily
    move to explain any persistent body-mass wobble.
    """
    w = _SURPLUS_GAIN + (1.0 - _SURPLUS_GAIN) / (
        1.0 + np.exp(np.clip(V / _VOL_SMOOTH, -60.0, 60.0)))
    return float(-V * w)


def predict(x: np.ndarray, p: Params, u: np.ndarray,
            dose_cum: float = 0.0) -> dict[str, float]:
    """Expected measurement for a latent state, exact (no noise)."""
    A, S, V, F, I, C, M = (float(v) for v in x)
    dep = vol_effect(V)

    out = {
        # --- autonomic and cardiovascular -----------------------------------
        # HRV is multiplicative in load: the same absolute drop means something
        # different to somebody who sits at 110 ms than to somebody at 38.
        "hrv_rmssd": p.hrv0 * np.exp(-p.b_hrv * A) * (1.0 - 0.30 * dep),
        "hr_rest":   p.hr0 * (1.0 + p.b_hr * A + 0.22 * dep + 0.05 * I)
                     - 2.5 * (F - 1.0),
        # Volume loss lowers systolic pressure; load raises it. The two can
        # cancel, which is why blood pressure alone is a poor early channel and
        # why it is worth having in a multivariate residual.
        "bp_sys":    p.bp0 + 4.2 * A - 26.0 * dep + 1.5 * I,
        "spo2":      p.spo0 - 0.85 * I - 0.12 * A,
        "br_rate":   p.br0 + 1.5 * A + 2.0 * I,
        "skin_temp": p.st0 + 0.70 * I + 0.045 * A,

        # --- sleep and activity: outputs of the state, not inputs -----------
        "sleep_h":   achieved_sleep(x, p, u),
        "sleep_eff": sleep_efficiency(x, p),
        "activity":  achieved_activity(x, p, u),

        # --- slower channels -------------------------------------------------
        # Body mass moves with total body water first and lean mass much later.
        "body_mass": p.bm0 * (1.0 + 0.055 * V) + 2.2 * (F - 1.0),
        "mood":      p.mood0 - 2.4 * (1.0 - M),
        # Perceived stress reads both the load and the reserve left to meet it.
        # The second term is there for identifiability as much as realism: with
        # mood as its only observer, psychological reserve was a latent axis
        # that any skipped or noisy questionnaire could move on its own, and a
        # single-channel axis is one the residual projection cannot
        # distinguish from measurement error.
        "stress":    p.stress0 + 1.75 * A + 0.50 * C + 0.60 * (1.0 - M),

        # --- environment, passed through rather than modelled ----------------
        "dose_rate": float(u[UIDX["dose"]]),
        "dose_cum":  float(dose_cum),
    }

    for k, v in out.items():
        lo, hi = SIGNALS[k].valid
        out[k] = float(np.clip(v, lo, hi))
    return out


#: Measurement noise, one standard deviation, in each signal's own unit. This is
#: sensor and day-to-day biological noise *around* a given latent state -- the
#: part the twin does not even try to predict. Floored by each channel's stated
#: resolution so a quantised signal cannot claim impossible precision.
OBS_SD: dict[str, float] = {
    "hrv_rmssd": 4.5,
    "hr_rest":   1.8,
    "bp_sys":    3.5,
    "spo2":      0.35,
    "br_rate":   0.55,
    "skin_temp": 0.09,
    "sleep_h":   0.35,
    "sleep_eff": 2.2,
    "activity":  11.0,
    "body_mass": 0.28,
    "mood":      0.42,
    "stress":    0.45,
}


def obs_sd(scale: dict[str, float] | None = None) -> np.ndarray:
    """Measurement sigma over `MONITORED`, optionally scaled per channel.

    `scale` is how one astronaut's own variability differs from the population:
    some people simply have noisier nights than others, and treating that as
    signal is how a monitor ends up crying wolf at the quietest crew member.
    """
    out = []
    for k in MONITORED:
        sd = OBS_SD[k] * (scale or {}).get(k, 1.0)
        out.append(max(sd, SIGNALS[k].resolution))
    return np.array(out)


def as_vector(obs: dict[str, float]) -> np.ndarray:
    """Monitored channels only, in `MONITORED` order, NaN where absent."""
    return np.array([obs.get(k, np.nan) for k in MONITORED], dtype=float)


def jacobian(x: np.ndarray, p: Params, u: np.ndarray) -> np.ndarray:
    """d(monitored signal) / d(latent state), shape (len(MONITORED), N_STATE).

    Central differences. The observation model is cheap, the state is seven
    dimensional, and an analytic Jacobian here would be fifty lines of algebra
    to maintain in parallel with `predict` -- which is how the two drift apart
    and the attribution starts quietly lying.
    """
    J = np.zeros((len(MONITORED), N_STATE))
    base_h = np.array([0.02, 0.25, 0.004, 0.004, 0.02, 0.06, 0.02])
    for j in range(N_STATE):
        h = base_h[j]
        xp, xm = x.astype(float).copy(), x.astype(float).copy()
        xp[j] += h
        xm[j] -= h
        yp = as_vector(predict(xp, p, u))
        ym = as_vector(predict(xm, p, u))
        J[:, j] = (yp - ym) / (2.0 * h)
    return J


def measure(x: np.ndarray, p: Params, u: np.ndarray, rng: np.random.Generator,
            *, dose_cum: float = 0.0,
            sd_scale: dict[str, float] | None = None) -> dict[str, float]:
    """A noisy reading of the state -- what a sensor would have written down."""
    truth = predict(x, p, u, dose_cum)
    sd = {k: OBS_SD[k] * (sd_scale or {}).get(k, 1.0) for k in MONITORED}
    out = dict(truth)
    for k in MONITORED:
        lo, hi = SIGNALS[k].valid
        out[k] = float(np.clip(truth[k] + rng.normal(0.0, sd[k]), lo, hi))
    return out
