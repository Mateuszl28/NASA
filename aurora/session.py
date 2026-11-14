"""Wiring: a mission in, calibrated twins and findings out.

One function does the whole pipeline, because the pipeline is the product and
every entry point -- the CLI, the validator, the dashboard -- needs exactly the
same thing assembled the same way. Splitting it across callers is how the
dashboard and the self-test end up quietly disagreeing about what the system
found.

The ordering is the only subtle part. Calibration must see a window that
contains no event, so it runs on the declared quiet stretch at the start of the
mission and nothing later. The filter and the scan then run over the *whole*
mission using those fixed parameters. Letting calibration slide forward over
the full record would be the classic way to lose: the twin would learn the
event as this person's normal and then faithfully report that nothing is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import Calibration, calibrate_cached
from .crew import CREW, CREW_ORDER, population_params
from .divergence import DayScore, Finding, ResidCal, findings, fit_resid_cal, scan
from .truth import (
    CALIB_DAYS, Mission, gravity_shift_for, phase_shift_for, simulate,
)
from .twin import FilterOut, Window, run_filter, window_from_arrays

#: Chosen against the false-alarm budget on pooled nominal missions, then held.
#: `validate.sweep_threshold` prints the trade it was chosen from.
DEFAULT_THRESHOLD = 8.0


def plan_arrays(days: int) -> tuple[np.ndarray, np.ndarray]:
    """Circadian demand and headward-shift series the twin is told about."""
    return (np.array([phase_shift_for(d) for d in range(days)]),
            np.array([gravity_shift_for(d) for d in range(days)]))


def window_for(m: Mission, crew: str) -> Window:
    phase, grav = plan_arrays(m.days)
    return window_from_arrays(m.inputs[crew], m.obs[crew], m.dose_cum[crew],
                              phase, grav)


@dataclass
class CrewView:
    crew: str
    role: str
    window: Window
    calib: Calibration
    filt: FilterOut
    rcal: ResidCal
    scans: list[DayScore]

    def score_series(self, days: int) -> np.ndarray:
        s = np.full(days, np.nan)
        for d in self.scans:
            s[d.day] = d.score
        return s

    def day(self, day: int) -> DayScore | None:
        for d in self.scans:
            if d.day == day:
                return d
        return None


@dataclass
class Session:
    mission: Mission
    views: dict[str, CrewView]
    findings: list[Finding]
    threshold: float
    calib_days: int

    @property
    def days(self) -> int:
        return self.mission.days

    def findings_for(self, crew: str) -> list[Finding]:
        return [f for f in self.findings if f.crew == crew]


def build(scenario: str = "nominal", *, seed: int = 7,
          threshold: float = DEFAULT_THRESHOLD, days: int | None = None,
          refresh: bool = False, calib_days: int = CALIB_DAYS,
          crew_only: tuple[str, ...] | None = None) -> Session:
    m = simulate(scenario, seed=seed, **({"days": days} if days else {}))
    prior = population_params()
    views: dict[str, CrewView] = {}

    for c in (crew_only or CREW_ORDER):
        w = window_for(m, c)
        cal = calibrate_cached(w.slice(0, calib_days), prior,
                               CREW[c].sd_scale, crew=c, refresh=refresh)
        fo = run_filter(w, cal.params, sd_scale=CREW[c].sd_scale)
        rc = fit_resid_cal(w, cal.params, sd_scale=CREW[c].sd_scale, fo=fo,
                           calib_days=calib_days)
        ss = scan(w, cal.params, crew=c, sd_scale=CREW[c].sd_scale, fo=fo,
                  rcal=rc, threshold=threshold, calib_days=calib_days)
        views[c] = CrewView(crew=c, role=CREW[c].role, window=w, calib=cal,
                            filt=fo, rcal=rc, scans=ss)

    fs = findings({c: v.scans for c, v in views.items()}, threshold=threshold)
    return Session(mission=m, views=views, findings=fs, threshold=threshold,
                   calib_days=calib_days)
