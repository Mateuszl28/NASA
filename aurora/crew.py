"""Four astronauts who are not interchangeable.

The point of the roster is that the same event looks different on each of them,
and a monitor that scores all four against one range is wrong about at least
three. FE-2's rested HRV is 44 ms; FE-3's is 103 ms. A reading of 60 ms is a
good night for one and a developing problem for the other, and no population
percentile can tell you which.

Each profile is declared by what their charts read on a quiet day -- the
numbers a flight surgeon would actually have from pre-flight and early-mission
testing -- plus how steeply this person responds to load. The model's
intercepts are then *solved* so that those readings are the fixed point of the
dynamics (`build_params`). Declaring intercepts directly was the first version
and it was wrong: an intercept that is not an equilibrium means a quiet mission
drifts away from its own stated baseline, and the twin spends the mission
detecting its own initialisation.

Note on provenance: these are synthetic physiologies with plausible levels and
variability, not anybody's records. In particular nothing here is attributed to
a real crew member -- the Bio-Monitor framing is about what the *instrument*
measures, and inventing medical data for a named astronaut would be a different
and much worse thing than modelling a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .model import NOMINAL_INPUT, UIDX, Params, achieved_activity, settle
from .observe import vol_effect


@dataclass(frozen=True)
class Profile:
    key: str
    role: str
    note: str
    #: What this person's monitored channels read on an ordinary quiet day.
    targets: dict[str, float]
    #: How steeply they respond to load: the four observation sensitivities.
    sens: dict[str, float]
    #: Dynamics overrides -- their recovery rates and what costs them.
    dyn: dict[str, float] = field(default_factory=dict)
    #: Their own measurement variability, as a multiplier on population sigma
    #: per channel. Within-person variability is a stable trait, so unlike the
    #: adapted *level* it transfers from pre-flight testing -- which is what
    #: lets the twin have a usable sigma before the mission has produced one.
    sd_scale: dict[str, float] = field(default_factory=dict)


def build_params(prof: Profile) -> Params:
    """Solve the observation intercepts that make `prof.targets` the fixed point.

    Fixed-point iteration, because the circularity is real: sleep efficiency
    depends on autonomic load, load depends on sleep debt, and debt depends on
    efficiency. Two derived quantities come along for the ride --

      `sleep_need` is set to the hours a nominal night actually delivers, so a
      quiet day carries no standing sleep debt. A crew member whose declared
      sleep need exceeds what the schedule can deliver is not "nominal with a
      long sleep need", they are chronically sleep-deprived, and saying so is a
      finding rather than a baseline.

      `f_maint` is set so unit fitness is a fixed point. Without it, completed
      training sits below maintenance, fitness decays, lower fitness reduces
      completed training, and a healthy crew member deconditions to the bound
      over a long mission for no reason but a mis-set constant.
    """
    t = prof.targets
    opp = float(NOMINAL_INPUT[UIDX["sleep_opp"]])
    planned = float(NOMINAL_INPUT[UIDX["exercise"]])

    p = Params(
        **prof.sens,
        **prof.dyn,
        sleep_need=opp * t["sleep_eff"] / 100.0,
        hrv0=t["hrv_rmssd"], hr0=t["hr_rest"], bp0=t["bp_sys"],
        spo0=t["spo2"], br0=t["br_rate"], st0=t["skin_temp"],
        eff0=t["sleep_eff"], bm0=t["body_mass"],
        mood0=t["mood"], stress0=t["stress"],
    )

    # Warm-started: fitness has a ~220-day time constant, so a cold settle on
    # every iteration either truncates its transient (leaving F a few percent
    # below unity and the declared baseline quietly non-stationary) or costs
    # thousands of steps per iteration. Carrying the equilibrium forward costs
    # neither.
    x = settle(p, NOMINAL_INPUT, days=4000)
    for _ in range(200):
        x = settle(p, NOMINAL_INPUT, x0=x, days=4000)
        A, S, V, F, I, C, M = (float(v) for v in x)
        # The same function the observation model uses, not a second copy of
        # it. Two copies drifted apart the moment the surplus side gained a
        # gain, and the declared quiet-day readings stopped being a fixed
        # point by a few thousandths -- small, silent, and in exactly the three
        # channels that respond to volume.
        dep = vol_effect(V)

        # Hold unit fitness as the fixed point: pick the maintenance load that
        # exactly offsets microgravity decay at the completed training this
        # person achieves at their own nominal load.
        done_at_unit_F = achieved_activity(
            np.array([A, S, V, 1.0, I, C, M]), p, NOMINAL_INPUT)
        f_maint = max(30.0, done_at_unit_F / (1.0 + p.f_decay / p.k_F))

        nxt = replace(
            p,
            f_maint=f_maint,
            hrv0=t["hrv_rmssd"] / (np.exp(-p.b_hrv * A) * (1.0 - 0.30 * dep)),
            hr0=(t["hr_rest"] + 2.5 * (F - 1.0))
                / (1.0 + p.b_hr * A + 0.22 * dep + 0.05 * I),
            bp0=t["bp_sys"] - 4.2 * A + 26.0 * dep - 1.5 * I,
            spo0=t["spo2"] + 0.85 * I + 0.12 * A,
            br0=t["br_rate"] - 1.5 * A - 2.0 * I,
            st0=t["skin_temp"] - 0.70 * I - 0.045 * A,
            eff0=t["sleep_eff"] + p.b_eff * A + 3.2 * C
                 + 1.8 * max(0.0, I - 0.10),
            bm0=(t["body_mass"] - 2.2 * (F - 1.0)) / (1.0 + 0.055 * V),
            mood0=t["mood"] + 2.4 * (1.0 - M),
            stress0=t["stress"] - 1.75 * A - 0.50 * C - 0.60 * (1.0 - M),
        )
        moved = max(abs(getattr(nxt, k) - getattr(p, k)) for k in (
            "hrv0", "hr0", "bp0", "spo0", "br0", "st0", "eff0", "bm0",
            "mood0", "stress0", "f_maint"))
        p = nxt
        if moved < 1e-9:
            break

    # `sleep_need` was derived from the declared efficiency before the
    # intercepts moved; re-derive it against the settled state so the balance
    # it is supposed to guarantee actually holds.
    x = settle(p, NOMINAL_INPUT, x0=x, days=4000)
    from .model import achieved_sleep  # local: keeps the import graph flat
    return replace(p, sleep_need=achieved_sleep(x, p, NOMINAL_INPUT))


@dataclass(frozen=True)
class Astronaut:
    key: str
    role: str
    note: str
    truth: Params
    sd_scale: dict[str, float]


PROFILES: tuple[Profile, ...] = (
    Profile(
        key="cdr", role="Commander",
        note="44. Low resting rate, unremarkable HRV, sheds workload quickly. "
             "The crew member a population-range monitor is least likely to "
             "flag and most likely to be right about.",
        targets={"hrv_rmssd": 62.0, "hr_rest": 54.0, "bp_sys": 122.0,
                 "spo2": 97.6, "br_rate": 13.4, "skin_temp": 36.48,
                 "sleep_eff": 92.5, "body_mass": 81.2, "mood": 6.1,
                 "stress": 2.3},
        sens={"b_hrv": 0.49, "b_hr": 0.071, "b_eff": 5.4, "b_act": 0.19},
        dyn={"recov_rate": 0.68, "rho_A": 0.28, "g_sleep": 0.062,
             "g_work": 0.62, "exc_tol": 128.0, "k_V": 0.21, "k_I": 0.33},
        sd_scale={"hrv_rmssd": 0.85, "sleep_h": 0.80, "mood": 0.75},
    ),
    Profile(
        key="fe1", role="Flight Engineer 1",
        note="38. High HRV, markedly sleep-sensitive and slow to shed "
             "autonomic load: a short night costs this one nearly twice what "
             "it costs the commander. The profile the worked example follows.",
        # Note the HRV target is the *unloaded* fixed point. Under the actual
        # flight plan -- four EVAs, two dockings, a weekly workload rhythm --
        # this crew member's in-flight nominal HRV settles near 75 ms, which is
        # the "own normal" the monitoring is against.
        targets={"hrv_rmssd": 86.0, "hr_rest": 61.0, "bp_sys": 112.0,
                 "spo2": 98.1, "br_rate": 15.1, "skin_temp": 36.62,
                 "sleep_eff": 90.0, "body_mass": 68.4, "mood": 5.7,
                 "stress": 2.9},
        sens={"b_hrv": 0.62, "b_hr": 0.094, "b_eff": 7.8, "b_act": 0.28},
        dyn={"recov_rate": 0.47, "rho_A": 0.18, "g_sleep": 0.088,
             "g_work": 1.05, "exc_tol": 108.0, "k_V": 0.16, "k_I": 0.28},
        sd_scale={"hrv_rmssd": 1.15, "sleep_eff": 1.20, "stress": 1.10},
    ),
    Profile(
        key="fe2", role="Flight Engineer 2",
        note="51. Low rested HRV and slow autonomic recovery -- the oldest "
             "crew member, whose absolute numbers look worst and whose "
             "deviations are what actually matter.",
        targets={"hrv_rmssd": 44.0, "hr_rest": 66.0, "bp_sys": 131.0,
                 "spo2": 97.2, "br_rate": 14.8, "skin_temp": 36.59,
                 "sleep_eff": 87.5, "body_mass": 77.9, "mood": 5.8,
                 "stress": 2.7},
        sens={"b_hrv": 0.44, "b_hr": 0.088, "b_eff": 6.9, "b_act": 0.31},
        dyn={"recov_rate": 0.41, "rho_A": 0.15, "g_sleep": 0.079,
             "g_work": 1.22, "exc_tol": 96.0, "k_V": 0.13, "k_I": 0.24},
        sd_scale={"hrv_rmssd": 0.70, "bp_sys": 1.25, "body_mass": 1.10},
    ),
    Profile(
        key="fe3", role="Flight Engineer 3",
        note="35. Athletic, tolerates a heavy exercise prescription, recovers "
             "fast. Also the crew member whose HRV has the furthest to fall "
             "before it reaches anybody's population threshold.",
        targets={"hrv_rmssd": 103.0, "hr_rest": 52.0, "bp_sys": 108.0,
                 "spo2": 98.4, "br_rate": 12.9, "skin_temp": 36.51,
                 "sleep_eff": 93.8, "body_mass": 71.0, "mood": 6.3,
                 "stress": 2.1},
        sens={"b_hrv": 0.71, "b_hr": 0.065, "b_eff": 4.6, "b_act": 0.16},
        dyn={"recov_rate": 0.72, "rho_A": 0.31, "g_sleep": 0.054,
             "g_work": 0.53, "exc_tol": 142.0, "k_V": 0.23, "k_I": 0.36},
        sd_scale={"hrv_rmssd": 1.30, "activity": 0.85, "mood": 0.80},
    ),
)

CREW: dict[str, Astronaut] = {
    pr.key: Astronaut(pr.key, pr.role, pr.note, build_params(pr), pr.sd_scale)
    for pr in PROFILES
}

CREW_ORDER: tuple[str, ...] = tuple(pr.key for pr in PROFILES)


def population_params() -> Params:
    """The population mean, used as the prior each twin is shrunk toward.

    Averaged over the roster rather than hand-written, so the prior cannot
    accidentally be centred on one crew member and quietly flatter their twin.
    """
    mean = {}
    for k in Params.__dataclass_fields__:
        mean[k] = sum(getattr(CREW[c].truth, k)
                      for c in CREW_ORDER) / len(CREW_ORDER)
    return Params(**mean)
