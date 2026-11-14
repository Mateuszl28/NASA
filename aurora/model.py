"""The twin itself: a latent physiological state, and how one day becomes the next.

The design decision that everything else rests on is that the twin does not
model *signals*. It models a small set of unobservable quantities -- autonomic
load, sleep debt, plasma volume, fitness, inflammation, circadian alignment,
psychological reserve -- and then maps those onto the fourteen things a
Bio-Monitor and a dosimeter can actually see.

That indirection is what buys the brief's headline property. HRV, resting heart
rate, sleep efficiency and achieved activity are not four independent
measurements that happen to move together; in this model they are four windows
onto one number. A coherent nudge across all four is a large movement in `A`
even when no single channel has left its own normal range, because they are
projections of the same thing. Scoring each channel separately throws that away
and is exactly why threshold monitors are late.

The state is deliberately small. Seven latent dimensions against twelve
monitored channels keeps the map over-determined, which is what makes a residual
meaningful: if the observations can no longer be explained by *any* setting of
seven numbers, something is happening that this model of this person does not
contain. That is the alarm, and `divergence.py` is where it is read.

Three couplings are load-bearing and worth naming:

  sleep -> efficiency -> debt -> load -> sleep
      Autonomic load degrades sleep efficiency, which turns a fixed sleep
      opportunity into less actual sleep, which deepens the debt that drives
      load. The brief's example is this loop, two turns in.

  load -> achieved activity
      Exercise is prescribed but only partly completed. Activity is therefore
      an *output* of the state, not an input to it, and a quiet drop in
      completed training is one of the earliest things the loop produces.

  load as a smoothed load index
      `A` is written as an exponential moving average of an instantaneous load
      demand `L`, with rate `rho_A`. That is not a cosmetic rearrangement: it
      makes `A` converge to `L` rather than to `L / rho_A`, so every gain in
      `L` reads directly as "steady-state autonomic load this cost produces",
      and `rho_A` means only "how fast this person sheds it". Gains that have
      to be divided by a recovery rate before they mean anything are gains
      nobody can sanity-check, and they were the first thing to go wrong here.

Units, once: `A` is dimensionless with 1.0 at roughly the load where a flight
surgeon would want to know; `S` is hours of accumulated debt; `V` is fractional
plasma-volume deviation from this person's adapted baseline (negative is
depleted); `F` is fitness as a fraction of adapted baseline; `I` is
dimensionless inflammatory activation; `C` is hours of circadian misalignment;
`M` is psychological reserve on 0-1 with 1 good.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# --- the latent state ---------------------------------------------------------

STATE: tuple[str, ...] = ("A", "S", "V", "F", "I", "C", "M")
N_STATE = len(STATE)
IDX = {k: i for i, k in enumerate(STATE)}

STATE_NAMES: dict[str, str] = {
    "A": "Autonomic load",
    "S": "Sleep debt",
    "V": "Plasma volume",
    "F": "Aerobic fitness",
    "I": "Inflammatory activation",
    "C": "Circadian misalignment",
    "M": "Psychological reserve",
}

STATE_UNITS: dict[str, str] = {
    "A": "", "S": "h", "V": "frac", "F": "frac", "I": "", "C": "h", "M": "",
}

#: Hard bounds. These are physiological, not numerical: plasma volume does not
#: fall by half and a body does not run an inflammatory response indefinitely.
#: Clipping here rather than trusting the dynamics keeps a bad parameter draw
#: during Monte Carlo from producing a trajectory nobody could survive.
BOUNDS: dict[str, tuple[float, float]] = {
    "A": (0.0, 3.0),
    "S": (0.0, 30.0),
    "V": (-0.35, 0.12),
    "F": (0.55, 1.25),
    "I": (0.0, 1.6),
    "C": (0.0, 9.0),
    "M": (0.0, 1.0),
}

#: A plausible adapted starting point. Not an equilibrium of any particular
#: crew member's parameters -- `twin.settle()` runs the dynamics forward under
#: nominal inputs to find that, which is both more honest and less fragile than
#: hand-writing one.
NOMINAL_STATE = np.array([0.13, 0.4, -0.01, 1.00, 0.02, 0.15, 0.91])


# --- the inputs ---------------------------------------------------------------
# What the mission does to the astronaut. Everything here is either scheduled by
# flight ops or imposed by the environment, which is what makes counterfactuals
# meaningful: these are the only quantities anybody can actually change.

INPUTS: tuple[str, ...] = (
    "sleep_opp", "exercise", "workload", "fluid", "dose", "light",
)
N_INPUT = len(INPUTS)
UIDX = {k: i for i, k in enumerate(INPUTS)}

INPUT_NAMES: dict[str, str] = {
    "sleep_opp": "Sleep opportunity",
    "exercise":  "Prescribed exercise",
    "workload":  "Task workload",
    "fluid":     "Fluid intake",
    "dose":      "Dose rate",
    "light":     "Bright-light compliance",
}

INPUT_UNITS: dict[str, str] = {
    "sleep_opp": "h", "exercise": "min", "workload": "0-1",
    "fluid": "L", "dose": "uGy/h", "light": "0-1",
}

#: An ordinary ISS-like day: 8.5 h of sleep opportunity, the standard two-hour
#: exercise prescription, moderate task load, nominal galactic-cosmic-ray dose.
NOMINAL_INPUT = np.array([8.5, 120.0, 0.50, 2.5, 20.0, 0.70])

#: Sensible operational ranges, used to bound the counterfactual search so it
#: cannot propose eleven hours of sleep and no work.
INPUT_RANGE: dict[str, tuple[float, float]] = {
    "sleep_opp": (6.0, 9.5),
    "exercise":  (0.0, 180.0),
    "workload":  (0.20, 0.95),
    "fluid":     (1.8, 4.0),
    "dose":      (0.0, 5000.0),
    "light":     (0.0, 1.0),
}

#: Which direction each input has to move to *relieve* a loaded astronaut. Used
#: by the counterfactual search so it proposes rest rather than stumbling onto
#: it, and by the UI to orient its sliders.
INPUT_RELIEF: dict[str, float] = {
    "sleep_opp": +1.0, "exercise": -1.0, "workload": -1.0,
    "fluid": +1.0, "dose": -1.0, "light": +1.0,
}


# --- per-person parameters ----------------------------------------------------

@dataclass(frozen=True)
class Params:
    """One astronaut's physiology.

    Split into two groups on purpose, because the data identifies them very
    differently.

    The *observation* parameters -- someone's own HRV level, their own resting
    heart rate, their own night-to-night variability -- are what a quiet
    stretch of mission measures well. Forty nominal days pin them down.

    The *dynamics* parameters are how this person converts load into strain:
    their sleep need, how fast they recover, how much a heavy workload costs
    them. A quiet stretch of mission says almost nothing about these, for the
    obvious reason that nothing happened. You cannot learn somebody's stress
    response from a period without stress.

    `calibrate.py` therefore fits the first group freely and shrinks the second
    hard toward population priors, and reports how much of each parameter came
    from the prior rather than from this person. A twin that hides that is
    claiming to know something it does not.
    """

    # --- dynamics: sleep, recovery, timescales ------------------------------
    sleep_need: float = 7.6       # h of *actual* sleep that holds debt steady
    recov_rate: float = 0.55      # share of a surplus hour that repays debt
    # Debt dissipation is the gain control on the model's one feedback loop:
    #   load -> sleep efficiency -> achieved sleep -> debt -> load
    # whose loop gain is (g_sleep / s_leak) * (sleep_opp / 100) * b_eff. At
    # s_leak = 0.12 that is about 0.5, and while a gain below one is formally
    # stable it amplifies every ordinary perturbation two- to threefold -- a
    # nominal mission with four EVAs drove autonomic load to 1.25, which is the
    # model asserting that a healthy crew member on a normal flight plan is in
    # serious overload. At 0.25 the loop gain is about 0.23, the amplification
    # is 1.3, and an acute debt half-life of under three days is also the more
    # defensible number.
    s_leak: float = 0.25          # homeostatic dissipation of debt, per day
    rho_A: float = 0.22           # autonomic recovery rate, per day
    k_C: float = 0.35             # circadian realignment, per day

    # --- dynamics: the load index L. Each gain is read as "steady-state A
    # --- that one unit of this cost produces", which is only true because A is
    # --- an EMA of L rather than an accumulator.
    g_base: float = 0.12          # the standing cost of being in microgravity
    g_sleep: float = 0.075        # per hour of accumulated debt
    g_work: float = 0.90          # per unit workload above 0.5
    g_exc: float = 0.25           # per hour of exercise above tolerance
    exc_tol: float = 115.0        # min/day tolerated without a strain cost
    g_circ: float = 0.080         # per hour of circadian misalignment
    g_infl: float = 0.45          # per unit inflammatory activation
    g_vol: float = 4.0            # per unit fractional plasma-volume deficit

    # --- dynamics: the slower systems ---------------------------------------
    k_V: float = 0.18             # plasma volume restoration, per day
    v_fluid: float = 0.022        # volume gain per surplus litre
    v_sweat: float = 0.012        # volume cost per nominal exercise block
    fluid_need: float = 1.95      # L/day; set so a nominal day is in balance
    k_F: float = 0.012            # fitness response to a training surplus
    f_maint: float = 110.0        # completed min/day that holds fitness
    f_decay: float = 0.0016       # microgravity deconditioning, per day
    k_I: float = 0.30             # inflammatory clearance, per day
    i_dose: float = 0.0085        # activation per 10 uGy/h above background
    dose_nom: float = 20.0        # uGy/h treated as background
    m_recov: float = 0.085        # psychological reserve recovery
    m_strain: float = 0.055       # reserve cost per unit autonomic load
    m_circ: float = 0.012         # reserve cost per hour of misalignment

    # --- observation: this person's own levels (intercepts at zero load) -----
    hrv0: float = 81.0            # ms nocturnal RMSSD
    hr0: float = 58.0             # bpm
    bp0: float = 118.0            # mmHg
    spo0: float = 97.8            # %
    br0: float = 14.2             # breaths/min
    st0: float = 36.55            # degC skin
    eff0: float = 91.0            # % sleep efficiency
    bm0: float = 74.5             # kg
    mood0: float = 5.9            # 1-7, at full reserve
    stress0: float = 2.6          # 1-7, at zero load

    # --- observation: this person's own sensitivities ------------------------
    b_hrv: float = 0.55           # HRV log-response to autonomic load
    b_hr: float = 0.082           # HR fractional response to load
    b_eff: float = 6.5            # efficiency points lost per unit load
    b_act: float = 0.24           # fraction of training lost per unit load

    def as_vector(self, keys: tuple[str, ...]) -> np.ndarray:
        return np.array([getattr(self, k) for k in keys], dtype=float)

    def with_vector(self, keys: tuple[str, ...], v: np.ndarray) -> "Params":
        return replace(self, **{k: float(x) for k, x in zip(keys, v)})


#: Parameters `calibrate.py` is allowed to move, and the box it may move them in.
#: Everything absent from this map is held at its population value -- not
#: because it is known exactly, but because the data cannot distinguish it, and
#: a free parameter the data does not constrain is a way of overfitting quietly.
FIT_BOUNDS: dict[str, tuple[float, float]] = {
    # observation levels: well identified by any quiet stretch
    "hrv0":    (25.0, 150.0),
    "hr0":     (40.0, 85.0),
    "bp0":     (95.0, 145.0),
    "spo0":    (94.0, 100.0),
    "br0":     (9.0, 22.0),
    "st0":     (35.8, 37.2),
    "eff0":    (75.0, 99.0),
    "bm0":     (45.0, 110.0),
    "mood0":   (3.5, 7.0),
    "stress0": (1.0, 4.5),
    # observation sensitivities: identified only once the state has moved
    "b_hrv":   (0.25, 1.00),
    "b_hr":    (0.040, 0.150),
    "b_eff":   (3.0, 12.0),
    "b_act":   (0.10, 0.45),
    # dynamics: weakly identified in a quiet window, shrunk hard
    "sleep_need": (6.8, 8.6),
    "recov_rate": (0.30, 0.85),
    "rho_A":      (0.10, 0.40),
    "g_sleep":    (0.040, 0.110),
    "g_work":     (0.40, 1.40),
    "exc_tol":    (85.0, 150.0),
}

FIT_KEYS: tuple[str, ...] = tuple(FIT_BOUNDS)

#: Prior strength per fitted parameter, as the standard deviation of a Gaussian
#: pull toward the population value, in that parameter's own units. Small sigma
#: is a tight prior. The observation *levels* get loose priors because the data
#: speaks; the dynamics get tight ones because it does not.
FIT_PRIOR_SD: dict[str, float] = {
    "hrv0": 45.0, "hr0": 14.0, "bp0": 18.0, "spo0": 2.0, "br0": 4.0,
    "st0": 0.6, "eff0": 9.0, "bm0": 25.0, "mood0": 1.4, "stress0": 1.2,
    "b_hrv": 0.16, "b_hr": 0.030, "b_eff": 2.4, "b_act": 0.09,
    "sleep_need": 0.35, "recov_rate": 0.11, "rho_A": 0.045,
    "g_sleep": 0.014, "g_work": 0.22, "exc_tol": 11.0,
}


# --- dynamics -----------------------------------------------------------------

_LO = np.array([BOUNDS[k][0] for k in STATE])
_HI = np.array([BOUNDS[k][1] for k in STATE])


def _clip_state(x: np.ndarray) -> np.ndarray:
    return np.clip(x, _LO, _HI)


def sleep_efficiency(x: np.ndarray, p: Params) -> float:
    """Sleep efficiency implied by the current state, in percent.

    Shared by the dynamics and the observation model on purpose. If the twin
    predicted one efficiency for the chart and used another to accumulate sleep
    debt, the vicious cycle in the module docstring would be decorative.
    """
    eff = (p.eff0
           - p.b_eff * x[IDX["A"]]
           - 3.2 * x[IDX["C"]]
           - 1.8 * max(0.0, x[IDX["I"]] - 0.10))
    return float(np.clip(eff, 35.0, 99.5))


def achieved_sleep(x: np.ndarray, p: Params, u: np.ndarray) -> float:
    """Hours actually slept: the opportunity, discounted by efficiency."""
    opp = float(u[UIDX["sleep_opp"]])
    return float(np.clip(opp * sleep_efficiency(x, p) / 100.0, 0.0, 14.0))


def achieved_activity(x: np.ndarray, p: Params, u: np.ndarray) -> float:
    """Intensity-weighted exercise minutes actually completed.

    Prescribed load is an input; completed load is a measurement. The gap is
    one of the first things to open when somebody is accumulating strain, and
    it opens before they report feeling bad.
    """
    capacity = (1.0
                - p.b_act * x[IDX["A"]]
                - 0.35 * max(0.0, 1.0 - x[IDX["F"]]))
    planned = float(u[UIDX["exercise"]])
    return float(np.clip(planned * float(np.clip(capacity, 0.0, 1.1)),
                         0.0, 400.0))


def load_index(x: np.ndarray, p: Params, u: np.ndarray) -> float:
    """Instantaneous load demand `L`, the value `A` is relaxing toward.

    Every term is a cost this person is paying today. Reading the terms is the
    whole explanation of an autonomic finding, which is why `divergence.py`
    hands them to the UI rather than only the total.
    """
    return (p.g_base
            + p.g_sleep * x[IDX["S"]]
            + p.g_work * max(0.0, float(u[UIDX["workload"]]) - 0.50)
            + p.g_exc * max(0.0, float(u[UIDX["exercise"]]) - p.exc_tol) / 60.0
            + p.g_circ * x[IDX["C"]]
            + p.g_infl * x[IDX["I"]]
            + p.g_vol * max(0.0, -x[IDX["V"]]))


def load_terms(x: np.ndarray, p: Params, u: np.ndarray) -> dict[str, float]:
    """`load_index` broken into its named contributions, for display."""
    return {
        "baseline":  p.g_base,
        "sleep":     p.g_sleep * x[IDX["S"]],
        "workload":  p.g_work * max(0.0, float(u[UIDX["workload"]]) - 0.50),
        "exercise":  p.g_exc * max(0.0, float(u[UIDX["exercise"]]) - p.exc_tol) / 60.0,
        "circadian": p.g_circ * x[IDX["C"]],
        "immune":    p.g_infl * x[IDX["I"]],
        "volume":    p.g_vol * max(0.0, -x[IDX["V"]]),
    }


def step(x: np.ndarray, u: np.ndarray, p: Params,
         *, phase_shift: float = 0.0, gravity_shift: float = 0.0) -> np.ndarray:
    """Advance the latent state one day.

    `phase_shift` is circadian demand imposed by the plan for this day -- a
    shifted sleep period for a docking or an EVA -- in hours. `gravity_shift`
    is the headward fluid redistribution of early microgravity adaptation, as a
    fractional plasma-volume loss for this day; it is a mission-phase fact
    rather than something the astronaut or the flight plan controls.
    """
    A, S, V, F, I, C, M = (float(v) for v in x)
    sleep_opp, exercise, workload, fluid, dose, light = (float(v) for v in u)

    # --- sleep debt. Uses achieved sleep, so a degraded night costs twice:
    # once as a short night, and again through the efficiency that shortened
    # it. The leak term is why a 10-minute chronic deficit does not integrate
    # into a crisis over a 1000-day mission.
    slept = achieved_sleep(x, p, u)
    deficit = p.sleep_need - slept
    S_n = S * (1.0 - p.s_leak)
    S_n += deficit if deficit >= 0.0 else p.recov_rate * deficit
    S_n = float(np.clip(S_n, *BOUNDS["S"]))

    # --- circadian. Bright light is the only countermeasure that works, so
    # compliance scales the realignment rate rather than offsetting the demand.
    C_n = C * (1.0 - p.k_C * (0.35 + 0.65 * light)) + abs(phase_shift)
    C_n = float(np.clip(C_n, *BOUNDS["C"]))

    # --- inflammatory activation, driven by dose above background.
    I_n = I * (1.0 - p.k_I) + p.i_dose * max(0.0, dose - p.dose_nom) / 10.0
    I_n = float(np.clip(I_n, *BOUNDS["I"]))

    # --- plasma volume. Restoration pulls toward this person's adapted
    # baseline; fluid surplus helps, sweat loss and the headward shift take it
    # away. `fluid_need` is set so a nominal day is in balance, which is what
    # makes V = 0 mean "adapted" rather than "drifting slowly downhill".
    V_n = (V * (1.0 - p.k_V)
           + p.v_fluid * (fluid - p.fluid_need)
           - p.v_sweat * (exercise / 120.0)
           - gravity_shift)
    V_n = float(np.clip(V_n, *BOUNDS["V"]))

    # --- fitness. Responds to completed training, not prescribed -- and to
    # training measured against *current* capacity rather than an absolute
    # maintenance load. That denominator is not decoration. With an absolute
    # one, lower fitness reduces completed training, which reduces fitness: the
    # loop has positive feedback, unit fitness is an unstable fixed point, and
    # a perfectly healthy crew member deconditions to the floor over a long
    # mission. Dividing by F supplies the restoring force real physiology has,
    # because the same absolute workload is a larger relative stimulus to a
    # deconditioned body.
    done = achieved_activity(x, p, u)
    F_n = F + p.k_F * (done / (p.f_maint * max(F, 0.3)) - 1.0) - p.f_decay
    F_n = float(np.clip(F_n, *BOUNDS["F"]))

    # --- autonomic load: an EMA of today's load demand, with this person's
    # own recovery rate. Note the post-update states feed L -- today's debt and
    # today's inflammation are today's costs.
    x_mid = np.array([A, S_n, V_n, F_n, I_n, C_n, M])
    L = load_index(x_mid, p, u)
    A_n = float(np.clip(A + p.rho_A * (L - A), *BOUNDS["A"]))

    # --- psychological reserve. Recovers toward 1, spent by load.
    M_n = M + p.m_recov * (1.0 - M) - p.m_strain * A_n - p.m_circ * C_n
    M_n = float(np.clip(M_n, *BOUNDS["M"]))

    return _clip_state(np.array([A_n, S_n, V_n, F_n, I_n, C_n, M_n]))


#: Per-day process noise on the latent state: how much the twin concedes it
#: cannot predict even when it has the inputs right. Deliberately not tiny. A
#: twin with no process noise claims the body is clockwork, and then every
#: ordinary day looks like an event.
PROCESS_SD = np.array([0.022, 0.30, 0.0045, 0.0022, 0.016, 0.070, 0.013])


def step_noisy(x: np.ndarray, u: np.ndarray, p: Params,
               rng: np.random.Generator, **kw) -> np.ndarray:
    return _clip_state(step(x, u, p, **kw) + rng.normal(0.0, PROCESS_SD))


def settle(p: Params, u: np.ndarray | None = None, *,
           x0: np.ndarray | None = None, days: int = 600,
           tol: float = 1e-10) -> np.ndarray:
    """The latent state this person relaxes to under a repeated input day.

    Used to define what "nominal" means for a given physiology instead of
    hand-writing it. That matters more than it sounds: if the declared nominal
    state is not actually a fixed point of the dynamics, then a quiet mission
    drifts, the drift is indistinguishable from a slow event, and the twin
    spends the whole mission detecting its own initialisation.
    """
    u = NOMINAL_INPUT if u is None else u
    x = (NOMINAL_STATE if x0 is None else x0).astype(float).copy()
    for _ in range(days):
        nxt = step(x, u, p)
        if float(np.max(np.abs(nxt - x))) < tol:
            return nxt
        x = nxt
    return x
