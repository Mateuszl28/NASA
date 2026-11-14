"""Making the twin *his* twin, and saying how much of it is actually his.

Calibration fits the model to one astronaut's own quiet stretch of mission by
minimising the filter's one-step prediction errors, with a Gaussian pull toward
population values. Two things about it are worth more than the fit itself.

**The prior is not a formality.** Twenty parameters against forty-five days is
not a lot of data, and the parameters are not equally knowable from it. Where
somebody's HRV sits, what their resting heart rate is, how efficiently they
sleep -- a quiet month measures those precisely. How steeply they respond to a
heavy workload, how fast they shed autonomic load, how much a short night costs
them -- a quiet month says almost nothing, because nothing happened. You cannot
learn a stress response from a period without stress. Fitting those freely
produces a confident number with no evidence under it, which is worse than a
population default because it looks like knowledge.

**So the shrinkage is reported.** `prior_share` is, per parameter, the fraction
of the posterior precision that came from the prior rather than from this
person. When the dashboard says FE-1 sheds autonomic load with a rate of 0.19,
it also says that 71 % of that number is population prior. A twin that hides
that ratio is claiming to know the astronaut better than the data permits, and
every downstream forecast inherits the overconfidence.

Standard errors come from the same decomposition and feed the forward ensemble,
which is where parameter uncertainty actually bites: being wrong about somebody's
recovery rate compounds over a fourteen-day projection, while being wrong about
today's measurement noise does not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .model import FIT_BOUNDS, FIT_KEYS, FIT_PRIOR_SD, Params
from .twin import Window, run_filter

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "twins"


@dataclass
class Calibration:
    crew: str
    params: Params
    #: Posterior standard error per fitted parameter.
    se: dict[str, float]
    #: Share of each parameter's posterior precision contributed by the prior.
    #: 0 means the data determined it; 1 means the model is repeating itself.
    prior_share: dict[str, float]
    nll: float
    n_obs: int
    days: int
    fitted: tuple[str, ...]

    @property
    def data_driven(self) -> list[str]:
        """Parameters this person's own data actually pinned down."""
        return sorted((k for k, v in self.prior_share.items() if v < 0.5),
                      key=lambda k: self.prior_share[k])

    @property
    def prior_driven(self) -> list[str]:
        return sorted((k for k, v in self.prior_share.items() if v >= 0.5),
                      key=lambda k: -self.prior_share[k])

    def summary(self) -> str:
        n_data = len(self.data_driven)
        return (f"{self.crew}: {self.days} d, {self.n_obs} readings, "
                f"{n_data}/{len(self.fitted)} parameters data-driven")


def _residuals(theta: np.ndarray, w: Window, prior: Params,
               keys: tuple[str, ...], sd_scale: dict[str, float],
               prior_vec: np.ndarray, prior_sd: np.ndarray,
               n_data_slots: int) -> np.ndarray:
    p = prior.with_vector(keys, theta)
    try:
        out = run_filter(w, p, sd_scale=sd_scale)
        z = out.innov_z[np.isfinite(out.innov_z)]
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        z = np.full(n_data_slots, 50.0)

    if z.size != n_data_slots:                 # coverage is data-determined, so
        z = np.resize(z, n_data_slots)         # this should not happen; be safe
    z = np.nan_to_num(z, nan=50.0, posinf=50.0, neginf=-50.0)

    pen = (theta - prior_vec) / prior_sd
    return np.concatenate([z, pen])


def calibrate(w: Window, prior: Params, sd_scale: dict[str, float], *,
              crew: str = "?", keys: tuple[str, ...] = FIT_KEYS,
              max_nfev: int = 60, verbose: bool = False) -> Calibration:
    """Fit `keys` on the window, shrinking toward `prior`."""
    prior_vec = prior.as_vector(keys)
    prior_sd = np.array([FIT_PRIOR_SD[k] for k in keys])
    lo = np.array([FIT_BOUNDS[k][0] for k in keys])
    hi = np.array([FIT_BOUNDS[k][1] for k in keys])
    theta0 = np.clip(prior_vec, lo + 1e-9, hi - 1e-9)

    base = run_filter(w, prior, sd_scale=sd_scale)
    n_data = int(np.isfinite(base.innov_z).sum())

    args = (w, prior, keys, sd_scale, prior_vec, prior_sd, n_data)
    sol = least_squares(
        _residuals, theta0, bounds=(lo, hi), args=args,
        x_scale=np.maximum(prior_sd, 1e-6), diff_step=1e-3,
        max_nfev=max_nfev, verbose=2 if verbose else 0,
    )

    params = prior.with_vector(keys, sol.x)

    # Precision decomposition. The last len(keys) rows of the Jacobian are the
    # prior penalty, whose contribution is exactly diag(1/sd^2); everything
    # above it is what this person's measurements contributed.
    J = sol.jac
    n_pen = len(keys)
    Jd = J[:-n_pen] if n_pen else J
    data_prec = np.einsum("ij,ij->j", Jd, Jd)
    prior_prec = 1.0 / prior_sd ** 2
    total = data_prec + prior_prec

    se, share = {}, {}
    try:
        cov = np.linalg.inv(J.T @ J)
        se_vec = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se_vec = 1.0 / np.sqrt(np.clip(total, 1e-12, None))

    for i, k in enumerate(keys):
        se[k] = float(se_vec[i])
        share[k] = float(prior_prec[i] / max(total[i], 1e-12))

    out = run_filter(w, params, sd_scale=sd_scale)
    return Calibration(crew=crew, params=params, se=se, prior_share=share,
                       nll=float(out.nll), n_obs=out.n_obs, days=w.days,
                       fitted=tuple(keys))


# --- caching ------------------------------------------------------------------
# Calibration is the slow step: a filter pass per residual evaluation, a
# Jacobian column per parameter. It also depends only on the crew member and
# their quiet window, which every scenario shares -- so computing it once per
# roster rather than once per scenario is not an optimisation, it is removing
# repeated work that was never different.

def _key(crew: str, w: Window, keys: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    h.update(crew.encode())
    h.update(np.ascontiguousarray(w.u, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(np.nan_to_num(w.y, nan=-9e9),
                                  dtype=np.float64).tobytes())
    h.update("|".join(keys).encode())
    return h.hexdigest()[:20]


def _to_json(c: Calibration) -> dict:
    return {
        "crew": c.crew,
        "params": {k: getattr(c.params, k)
                   for k in Params.__dataclass_fields__},
        "se": c.se, "prior_share": c.prior_share, "nll": c.nll,
        "n_obs": c.n_obs, "days": c.days, "fitted": list(c.fitted),
    }


def _from_json(d: dict) -> Calibration:
    return Calibration(
        crew=d["crew"], params=Params(**d["params"]), se=d["se"],
        prior_share=d["prior_share"], nll=d["nll"], n_obs=d["n_obs"],
        days=d["days"], fitted=tuple(d["fitted"]),
    )


def calibrate_cached(w: Window, prior: Params, sd_scale: dict[str, float], *,
                     crew: str, keys: tuple[str, ...] = FIT_KEYS,
                     refresh: bool = False, **kw) -> Calibration:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{crew}-{_key(crew, w, keys)}.json"
    if path.exists() and not refresh:
        try:
            return _from_json(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass                              # a stale cache is not a reason to fail
    c = calibrate(w, prior, sd_scale, crew=crew, keys=keys, **kw)
    path.write_text(json.dumps(_to_json(c), indent=1))
    return c
