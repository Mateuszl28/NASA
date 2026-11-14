"""The ground truth: a mission that actually happened, which the twin did not write.

This module is the honesty mechanism of the whole project, so it is worth being
explicit about what it refuses to do.

The lazy way to build a digital twin demo is to generate the data with the same
equations the twin uses. Detection then works beautifully and means nothing: the
residuals are pure measurement noise by construction, every event is visible the
instant it starts, and the "validation" is a test that floating-point arithmetic
is deterministic. Any reviewer who asks *what happens when the body does
something your model does not contain* gets no answer, which is unfortunate,
because on a real mission the answer is "constantly".

So the simulated astronaut here is deliberately **not** the twin:

1. **Unmodelled dynamics.** Real load does not accumulate linearly. Past a
   threshold, strain begins to amplify itself; sleep debt acts with a lag the
   twin's single-day step cannot represent; and there is a weekly rhythm in the
   crew's own schedule the twin never sees. All three are in the truth and none
   are in the model.

2. **Autocorrelated measurement noise.** The twin treats measurement error as
   white. Physiology is persistent, so the truth generates AR(1) noise. This is
   the single biggest driver of false alarms, and the reason the detection
   threshold in `divergence.py` is calibrated against pooled nominal runs
   rather than taken from a chi-squared table -- the table would be describing
   a mission nobody flew.

3. **Latent forcing with no input to explain it.** An event pushes the latent
   state directly. The twin can see the inputs -- sleep opportunity, workload,
   the exercise prescription, dose -- and it cannot see the forcing. That gap
   *is* the signal AURORA detects, and it is the right definition of a finding:
   a change in this person that the plan does not account for.

4. **Gaps.** The Bio-Monitor runs about 48 hours on a battery pair and is not
   worn in the shower or during some EVAs; body mass needs a scheduled
   measurement; self-report gets skipped. Missing days are missing, not
   interpolated.

Point 3 has a counterpart that matters just as much, and it is the scenario
`explained_load`. There, the crew member is genuinely hammered -- but by the
*schedule*, and the schedule is an input. Their HRV falls as far as it does in
the overload case. A monitor that scores deviation from baseline flags it; the
twin predicts it and says so. Getting that case right is most of the argument
for modelling a person rather than thresholding their charts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crew import CREW, CREW_ORDER
from .model import (
    IDX, NOMINAL_INPUT, N_INPUT, PROCESS_SD, STATE, UIDX,
    Params, _clip_state, achieved_sleep, settle, step,
)
from .observe import OBS_SD, predict
from .signals import MONITORED, SIGNALS, signals_of

MISSION_DAYS = 120

#: Events are barred from starting before this day, so every scenario contains
#: a genuine quiet stretch for the twin to calibrate on. A twin calibrated on a
#: window that already contains the event learns the event as normal -- which is
#: not a hypothetical, it is what happens if you let the window slide freely.
CALIB_DAYS = 45


# --- the flight plan ----------------------------------------------------------

#: Scheduled activities. EVAs cost sleep and raise workload for days around
#: them; dockings shift the sleep period. These are *known* to the twin, which
#: is the point: a monitor that cannot see the plan has to treat every busy week
#: as a medical finding.
EVA_DAYS: tuple[int, ...] = (34, 58, 86, 104)
DOCK_DAYS: tuple[int, ...] = (20, 72)


def phase_shift_for(day: int) -> float:
    """Hours of circadian demand imposed by the plan on a given day."""
    if day in DOCK_DAYS:
        return 2.4
    if day in EVA_DAYS:
        return 1.1
    return 0.0


def gravity_shift_for(day: int) -> float:
    """Headward fluid redistribution of early adaptation, per day.

    Front-loaded and decaying over the first three weeks. Known to the twin as
    a mission-phase fact rather than inferred, because it is a property of
    being in microgravity and not of the astronaut.
    """
    if day > 24:
        return 0.0
    return 0.0040 * float(np.exp(-day / 8.0))


def plan_inputs(day: int, crew: str) -> np.ndarray:
    """The flight plan's input vector for one crew member on one day.

    Everything here is logged by flight ops, so the twin gets it verbatim. The
    weekly rhythm in workload is real and the twin does see it -- what it does
    not see is the *response* rhythm added in `_truth_step`.
    """
    u = NOMINAL_INPUT.astype(float).copy()
    offset = CREW_ORDER.index(crew)

    # Ordinary week: a mid-week load peak, lighter at the weekend.
    u[UIDX["workload"]] = 0.50 + 0.10 * np.sin(2.0 * np.pi * (day + offset) / 7.0)

    # EVA campaigns. The day before is preparation, the day itself is long, and
    # the day after is still not a rest day.
    for e in EVA_DAYS:
        if day == e - 1:
            u[UIDX["workload"]] = 0.72
            u[UIDX["sleep_opp"]] = 7.6
        elif day == e:
            # Two crew go outside; the other two run the airlock and robotics.
            outside = offset in (1, 2)
            u[UIDX["workload"]] = 0.90 if outside else 0.74
            u[UIDX["sleep_opp"]] = 6.9
            u[UIDX["exercise"]] = 30.0 if outside else 90.0
            u[UIDX["fluid"]] = 2.1 if outside else 2.4
            u[UIDX["light"]] = 0.45
        elif day == e + 1:
            u[UIDX["workload"]] = 0.62
            u[UIDX["exercise"]] = 60.0

    for d in DOCK_DAYS:
        if day in (d - 1, d):
            u[UIDX["workload"]] = 0.80
            u[UIDX["sleep_opp"]] = 7.2
            u[UIDX["light"]] = 0.50

    return u


# --- events -------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Something that happens to the mission.

    `kind` decides whether the twin has any hope of explaining it:

      ``latent``  pushes the physiology directly. No input accounts for it, so
                  the twin's prediction and the measurements part company. This
                  is what a finding is.
      ``input``   changes the flight plan. Logged, visible to the twin, and
                  therefore *explained* -- the astronaut deteriorates and the
                  twin says "as expected".
      ``sensor``  corrupts a device's channels without touching the body. The
                  residual is real but confined to one device, which is what
                  `divergence.attribute` keys on.
    """
    kind: str
    label: str
    onset: int
    forcing: dict[str, float]
    crew: str | None = None          # None applies to the whole crew
    ramp: int = 2
    hold: int = 8
    decay: int = 8
    amplitude: float = 1.0

    def envelope(self, day: int) -> float:
        """Fraction of full amplitude in effect on `day`."""
        t = day - self.onset
        if t < 0:
            return 0.0
        if t < self.ramp:
            return self.amplitude * (t + 1) / max(self.ramp, 1)
        if t < self.ramp + self.hold:
            return self.amplitude
        t2 = t - self.ramp - self.hold
        if t2 < self.decay:
            return self.amplitude * (1.0 - (t2 + 1) / max(self.decay, 1))
        return 0.0

    def applies_to(self, crew: str) -> bool:
        return self.crew is None or self.crew == crew


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    note: str
    events: tuple[Event, ...]
    #: Who and what a validator should expect, as ground truth.
    expect_crew: tuple[str, ...] = ()
    expect_attribution: str = "individual"   # individual|environment|instrument|none
    expect_unexplained: bool = True


def _ev(**kw) -> Event:
    return Event(**kw)


SCENARIOS: dict[str, Scenario] = {}


def _add(s: Scenario) -> None:
    SCENARIOS[s.key] = s


_add(Scenario(
    key="nominal",
    title="Nominal cruise",
    note="A quiet mission with an ordinary flight plan: four EVAs, two "
         "dockings, nothing wrong with anybody. Every alarm raised here is a "
         "false one, which is what makes this the most important scenario in "
         "the set and the one the threshold is calibrated against.",
    events=(),
    expect_attribution="none",
    expect_unexplained=False,
))

_add(Scenario(
    key="overload",
    title="Autonomic overload, unexplained by the plan",
    note="The worked example. FE-1's HRV falls from 75 ms to the low 40s over "
         "a few days while resting heart rate rises, sleep degrades and "
         "completed exercise quietly drops. No single reading leaves this "
         "person's own range early on, and nothing in the flight plan accounts "
         "for it -- the schedule those days is unremarkable. The combination "
         "is what carries the signal, and the twin's job is to notice that the "
         "combination is one coherent movement rather than four coincidences.",
    events=(
        _ev(kind="latent", label="Unaccounted autonomic strain", crew="fe1",
            onset=62, ramp=1, hold=3, decay=14,
            forcing={"A": 0.180}),
    ),
    expect_crew=("fe1",),
))

_add(Scenario(
    key="explained_load",
    title="A brutal week that the schedule explains",
    note="The case that separates a twin from a baseline monitor. FE-1 is "
         "given twelve days of short sleep and heavy workload, all of it "
         "logged by flight ops. Their HRV falls about as far as it does under "
         "overload, and a deviation-from-baseline monitor raises the same "
         "medical finding for both.\n\n"
         "The twin was given the short sleep as an input, so it predicted most "
         "of this -- but not all of it, because a body already under load "
         "amplifies its own strain in a way a linear load index cannot "
         "express. So the honest answer is not silence. It is a finding "
         "roughly a tenth the size of the overload case, with the flight plan "
         "accounting for most of it, attributed to the *schedule* rather than "
         "to the body. Different number, different cause, different action: "
         "re-plan the week instead of examining the crew member.",
    events=(
        _ev(kind="input", label="Compressed sleep and heavy task load",
            crew="fe1", onset=62, ramp=1, hold=12, decay=2,
            forcing={"sleep_opp": -1.5, "workload": +0.30,
                     "exercise": +25.0, "light": -0.20}),
    ),
    expect_crew=("fe1",),
    expect_attribution="schedule",
))

_add(Scenario(
    key="fluid_loss",
    title="Plasma volume loss",
    note="A developing fluid and electrolyte deficit on FE-2: volume falls, "
         "resting heart rate rises to defend cardiac output, systolic pressure "
         "sags, HRV drops. The signature is distinct from autonomic overload "
         "chiefly in blood pressure, which falls here and rises under load -- "
         "which is exactly the kind of discrimination a single-channel alarm "
         "cannot make and a residual projected into latent space can.",
    events=(
        # -0.018/day against this crew member's restoration rate of 0.13
        # settles plasma volume about 14 % below their adapted baseline, which
        # is the range in-flight studies associate with orthostatic
        # intolerance. An earlier draft used a quarter of that and was simply
        # not detected -- correctly, since a 4 % shift moves every channel by
        # well under half its own measurement sigma. Keeping the weaker version
        # and lowering the threshold to catch it would have bought the
        # detection with false alarms on the whole healthy crew.
        _ev(kind="latent", label="Fluid and electrolyte deficit", crew="fe2",
            onset=58, ramp=4, hold=12, decay=10,
            forcing={"V": -0.018}),
    ),
    expect_crew=("fe2",),
))

_add(Scenario(
    key="infection",
    title="Early subclinical infection",
    note="An inflammatory response on FE-3 three days before anybody feels "
         "ill: skin temperature up a few tenths, breathing rate up, SpO2 down "
         "a fraction, HRV down, sleep fragmenting. Every one of those is within "
         "ordinary day-to-day variation on its own.",
    events=(
        _ev(kind="latent", label="Inflammatory activation", crew="fe3",
            onset=66, ramp=2, hold=7, decay=7,
            forcing={"I": 0.085}),
    ),
    expect_crew=("fe3",),
))

_add(Scenario(
    key="spe",
    title="Solar particle event",
    note="Dose rate rises two orders of magnitude for three days and the whole "
         "crew responds together. The correct answer is environmental: four "
         "simultaneous findings with a habitat signal that moved first are one "
         "event, not four medical workups. The dose channel is also the one "
         "place a personal baseline is the wrong instrument -- a protocol "
         "limit does not care whose body it is.",
    events=(
        _ev(kind="input", label="Solar particle event", crew=None,
            onset=70, ramp=1, hold=3, decay=3,
            forcing={"dose": +2600.0}),
    ),
    expect_crew=CREW_ORDER,
    expect_attribution="environment",
))

_add(Scenario(
    key="sensor_fault",
    title="Bio-Monitor electrode fault read as cardiac collapse",
    note="The garment's ECG loses skin contact on FE-2. HRV collapses and "
         "breathing rate goes erratic -- which is also what a failing "
         "autonomic system looks like. The discriminator is that the residual "
         "is confined to one device: body mass, dose and self-report, which "
         "come from elsewhere, are untouched, and no coherent latent state "
         "explains a body whose HRV collapsed while their mood, mass and "
         "stress did not move. Calling this cardiac costs a workup and, the "
         "next time, the crew's trust.",
    events=(
        _ev(kind="sensor", label="ECG contact degradation", crew="fe2",
            onset=64, ramp=2, hold=9, decay=4,
            forcing={"hrv_rmssd": -0.62, "br_rate": 3.4, "hr_rest": 0.9}),
    ),
    expect_crew=("fe2",),
    expect_attribution="instrument",
))


# --- the unmodelled part of the real astronaut --------------------------------

#: Lag-1 autocorrelation of measurement noise per channel. The twin assumes
#: white noise; bodies and sensors do not oblige.
NOISE_RHO: dict[str, float] = {
    "hrv_rmssd": 0.30, "hr_rest": 0.45, "bp_sys": 0.35, "spo2": 0.25,
    "br_rate": 0.30, "skin_temp": 0.40, "sleep_h": 0.20, "sleep_eff": 0.22,
    "activity": 0.25, "body_mass": 0.70, "mood": 0.50, "stress": 0.48,
}


def _truth_step(x: np.ndarray, u: np.ndarray, p: Params, day: int,
                hist_S: list[float], rng: np.random.Generator,
                forcing: dict[str, float]) -> np.ndarray:
    """One day of the *real* astronaut: the model, plus what it leaves out."""
    nxt = step(x, u, p,
               phase_shift=phase_shift_for(day),
               gravity_shift=gravity_shift_for(day))

    extra = np.zeros(len(STATE))

    # Strain amplifies itself once it is already high -- the part of a stress
    # response that a linear load index cannot express. Saturating, because an
    # unbounded version is not a nonlinearity, it is a divergence: each daily
    # push lands on top of the previous one, and past about A = 1.6 the
    # trajectory leaves the model for the bound regardless of what caused it.
    A = float(nxt[IDX["A"]])
    if A > 0.75:
        extra[IDX["A"]] += min(0.080, 0.060 * (A - 0.75) ** 1.4)

    # Sleep debt bites with a lag. The twin's one-day step has no memory, so a
    # two-day-old debt is invisible to it.
    if len(hist_S) >= 2:
        extra[IDX["A"]] += 0.0025 * hist_S[-2]

    # A weekly rhythm in how the crew member *responds*, distinct from the
    # weekly rhythm in what the schedule asks of them.
    extra[IDX["A"]] += 0.012 * np.sin(2.0 * np.pi * day / 7.0 + 1.1)
    extra[IDX["M"]] -= 0.006 * np.sin(2.0 * np.pi * day / 7.0)

    # The event itself: a push on the physiology with no input behind it.
    for k, v in forcing.items():
        if k in IDX:
            extra[IDX[k]] += v

    return _clip_state(nxt + extra + rng.normal(0.0, PROCESS_SD))


# --- the mission --------------------------------------------------------------

@dataclass
class Mission:
    scenario: Scenario
    days: int
    #: inputs[crew] -> (days, N_INPUT), exactly as flight ops logged them.
    inputs: dict[str, np.ndarray]
    #: obs[crew][channel] -> (days,) with NaN where no measurement exists.
    obs: dict[str, dict[str, np.ndarray]]
    #: The real latent state. Available to validation, never to the twin.
    truth_state: dict[str, np.ndarray]
    #: dose_cum[crew] -> (days,)
    dose_cum: dict[str, np.ndarray]
    seed: int = 0
    calib_days: int = CALIB_DAYS

    def channel(self, crew: str, ch: str) -> np.ndarray:
        return self.obs[crew][ch]

    def latent(self, crew: str, name: str) -> np.ndarray:
        return self.truth_state[crew][:, IDX[name]]

    def events_for(self, crew: str) -> tuple[Event, ...]:
        return tuple(e for e in self.scenario.events if e.applies_to(crew))

    def onset(self, crew: str) -> int | None:
        evs = self.events_for(crew)
        return min(e.onset for e in evs) if evs else None


def _coverage_mask(day: int, crew: str, ch: str,
                   rng: np.random.Generator) -> bool:
    """Whether a channel was actually measured. Gaps are declared, not filled."""
    sig = SIGNALS[ch]
    if sig.device == "slammd":
        # Scheduled inertial measurement, roughly every third day.
        return day % 3 == CREW_ORDER.index(crew) % 3
    if sig.device == "selfreport":
        return rng.random() > 0.08          # the odd skipped questionnaire
    if sig.device == "biomonitor":
        # Battery pairs last about 48 h, and the garment comes off.
        if day in EVA_DAYS and CREW_ORDER.index(crew) in (1, 2):
            return False
        return rng.random() > 0.06
    return True


def simulate(scenario: str = "nominal", *, seed: int = 7,
             days: int = MISSION_DAYS) -> Mission:
    """Fly a mission and write down what the instruments recorded."""
    sc = SCENARIOS[scenario]

    inputs: dict[str, np.ndarray] = {}
    obs: dict[str, dict[str, np.ndarray]] = {}
    truth_state: dict[str, np.ndarray] = {}
    dose_cum: dict[str, np.ndarray] = {}

    for crew in CREW_ORDER:
        # One independent stream per crew member. Sharing a generator across
        # the roster made one astronaut's measurement noise depend on another's
        # sensor fault -- the fault's extra draws shifted the stream for
        # everybody simulated after it, including over days before the fault
        # existed. That is not a reproducibility nicety: it silently
        # invalidated the shared quiet window that calibration is cached on,
        # and it would have made a per-crew comparison between scenarios
        # measure the generator rather than the physiology.
        rng = np.random.default_rng([seed, CREW_ORDER.index(crew)])
        p = CREW[crew].truth
        sd_scale = CREW[crew].sd_scale
        x = settle(p, NOMINAL_INPUT, days=4000)

        u_hist = np.zeros((days, N_INPUT))
        xs = np.zeros((days, len(STATE)))
        series = {ch: np.full(days, np.nan) for ch in MONITORED}
        series["dose_rate"] = np.full(days, np.nan)
        dc = np.zeros(days)
        cum = 0.0

        noise = {ch: 0.0 for ch in MONITORED}
        hist_S: list[float] = []

        for day in range(days):
            u = plan_inputs(day, crew)
            latent_forcing: dict[str, float] = {}
            sensor_forcing: dict[str, float] = {}

            for ev in sc.events:
                if not ev.applies_to(crew):
                    continue
                w = ev.envelope(day)
                if w <= 0.0:
                    continue
                if ev.kind == "input":
                    for k, v in ev.forcing.items():
                        u[UIDX[k]] += w * v
                elif ev.kind == "latent":
                    for k, v in ev.forcing.items():
                        latent_forcing[k] = latent_forcing.get(k, 0.0) + w * v
                elif ev.kind == "sensor":
                    for k, v in ev.forcing.items():
                        sensor_forcing[k] = sensor_forcing.get(k, 0.0) + w * v

            u = np.clip(u, [6.0, 0.0, 0.15, 1.5, 0.0, 0.0],
                           [10.0, 240.0, 1.0, 5.0, 1e6, 1.0])
            u_hist[day] = u

            cum += float(u[UIDX["dose"]]) * 24.0 / 1000.0   # uGy/h -> mGy/day
            dc[day] = cum

            truth_obs = predict(x, p, u, cum)

            for ch in MONITORED:
                rho = NOISE_RHO[ch]
                sd = OBS_SD[ch] * sd_scale.get(ch, 1.0)
                noise[ch] = (rho * noise[ch]
                             + np.sqrt(1.0 - rho ** 2) * rng.normal(0.0, sd))
                val = truth_obs[ch] + noise[ch]

                # A sensor fault corrupts the reading, not the body. HRV is
                # given a multiplicative hit because contact loss scales it;
                # breathing rate gets added variance.
                if ch in sensor_forcing:
                    f = sensor_forcing[ch]
                    if ch == "hrv_rmssd":
                        val *= max(0.05, 1.0 + f)
                    elif ch == "br_rate":
                        val += rng.normal(0.0, abs(f))
                    else:
                        val += f * rng.normal(1.0, 0.3)

                if _coverage_mask(day, crew, ch, rng):
                    lo, hi = SIGNALS[ch].valid
                    series[ch][day] = float(np.clip(val, lo, hi))

            series["dose_rate"][day] = float(u[UIDX["dose"]])

            xs[day] = x
            hist_S.append(float(x[IDX["S"]]))
            x = _truth_step(x, u, p, day, hist_S, rng, latent_forcing)

        inputs[crew] = u_hist
        obs[crew] = series
        truth_state[crew] = xs
        dose_cum[crew] = dc

    return Mission(scenario=sc, days=days, inputs=inputs, obs=obs,
                   truth_state=truth_state, dose_cum=dose_cum, seed=seed)
