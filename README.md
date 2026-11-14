# AURORA — a digital twin of the astronaut

**Every astronaut gets their own computational model of their own body, and is
monitored against themselves rather than against a population.**

Built around the Canadian Space Agency's **Bio-Monitor** — the Astroskin-derived
garment that David Saint-Jacques commissioned aboard the ISS in January 2019 —
AURORA takes the twelve things that instrument and a dosimeter can actually
measure, infers the seven physiological systems behind them, and reports what
the flight plan cannot explain.

```
FE-1 · Flight Engineer 1 · day 65
┌──────────────────────────────────────────────────────────────────────────┐
│ AUTONOMIC OVERLOAD — unexplained by the schedule        divergence 178   │
│                                                                          │
│ HRV        75 → 43 ms      resting HR   61 → 70 bpm                      │
│ sleep eff  90 → 67 %       activity    114 → 22 min completed            │
│                                                                          │
│ Every one of those readings is still inside this person's own ±3σ band.   │
│ A threshold monitor sees nothing. The combination is one coherent         │
│ movement in a single latent variable, and that is what was detected —     │
│ three days after it began, nine days before any channel leaves its range. │
│                                                                          │
│ latent shift   autonomic load +4.6σ · sleep debt +1.9σ · reserve −1.4σ    │
│ attribution    the astronaut (the plan accounts for 16 % of it)           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Run it

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/streamlit run app.py          # the dashboard
.venv/bin/python -m aurora --validate   # the detector validation sweep
.venv/bin/python -m aurora --scenarios  # what each mission is for
.venv/bin/python -m aurora --report     # the whole state as one document
.venv/bin/python -m pytest tests/ -q
```

The first run fits each twin and takes a couple of minutes; the fits are cached
under `data/twins/` and keyed by a hash of the window they were fitted on, so a
stale cache is ignored rather than silently wrong.

---

## The idea, in one paragraph

HRV, resting heart rate, sleep efficiency and completed exercise are not four
independent measurements that happen to move together. In this model they are
four *projections of one number* — autonomic load — with different gains and
different signs. So a coherent half-sigma movement across all four is a large
change in that one number, while four independent half-sigma wobbles are
nothing. Scoring channels one at a time throws that structure away, which is
precisely why threshold monitors are late. AURORA models the person, not the
charts.

---

## How it works

### 1 · A latent state, not a pile of signals

Seven unobservable quantities — autonomic load, sleep debt, plasma volume,
aerobic fitness, inflammatory activation, circadian misalignment, psychological
reserve — are mapped onto twelve measurable channels. Seven against twelve
keeps the map over-determined, and that is what makes a residual mean something:
if the readings can no longer be explained by *any* setting of seven numbers,
something is happening that this model of this person does not contain.

Three couplings carry most of the behaviour:

- **sleep → efficiency → debt → load → sleep.** Load degrades sleep efficiency,
  which turns a fixed sleep opportunity into less actual sleep, which deepens
  the debt that drives load.
- **load → completed activity.** Exercise is *prescribed* as an input but only
  partly *completed*, so achieved activity is an output of the state. A quiet
  drop in finished training is one of the earliest things this loop produces,
  and it happens before anybody reports feeling bad.
- **load as a smoothed index.** Autonomic load is written as an exponential
  moving average of an instantaneous load demand, so every gain reads directly
  as "the steady-state load this cost produces" and the recovery rate means
  only "how fast this person sheds it".

### 2 · Each crew member is declared by what their charts read on a quiet day

Four astronauts, none interchangeable. FE-2's rested HRV is 44 ms; FE-3's is
103 ms. A reading of 60 ms is a good night for one and a developing problem for
the other, and no population percentile can tell you which.

Each profile is specified by the numbers a flight surgeon would actually have —
quiet-day readings and response steepness — and the model's intercepts are then
*solved* so those readings are a fixed point of the dynamics. Declaring the
intercepts directly was the first version and it was wrong: an intercept that
is not an equilibrium means a quiet mission drifts away from its own stated
baseline, and the twin spends the mission detecting its own initialisation.

### 3 · Estimate, and forecast, are two different things

An extended Kalman filter maintains the best current guess at the hidden state
with an honest covariance. That alone cannot produce a finding, because a
filter's job is to explain the data: hand it a crew member whose HRV is
collapsing and it will dutifully raise its autonomic-load estimate and report
small, well-behaved residuals. Right about the state, silent about the event.

So detection uses a second, deliberately handicapped prediction — an
**open-loop forecast**, anchored days back and driven *only by the logged
flight-plan inputs*, with the intervening measurements withheld. It answers the
question that matters: given what we asked this person to do, what should they
look like today? The gap between the two is the finding.

### 4 · The residual is read in latent space, which is what names it

The standardised residual is projected onto the latent axes: *what single change
in load, volume, inflammation, debt, fitness, phase or reserve would best
explain all twelve channels at once?* The statistic is how much of the residual
that change accounts for, and because the axes are physiological systems, the
dominant component **is** the finding. Falling HRV with a rising heart rate and
a *falling* systolic pressure is plasma volume; the same heart-rate rise with a
*rising* pressure is autonomic load. A chart cannot separate those. A residual
resolved onto latent axes can.

### 5 · Three horizons, because degradation has three speeds

Every day is scanned at a 3-, 7- and 14-day forecast length and the strongest
wins. One length cannot do the job: autonomic overload develops over three days
and is loudest in a short window, while a plasma-volume deficit develops over
twelve, so any three-day window holds a quarter of its change — and the
filtered anchor three days back has already absorbed part of it, leaving the
forecast predicting the very drift it was meant to detect. At three days the
14 % volume loss scored 5.4 and was missed. At fourteen it is unmistakable.

### 6 · Four candidate explanations, not one

| attribution | evidence | what it costs to get wrong |
|---|---|---|
| **the astronaut** | a latent change explains it, confined to one crew member | — |
| **the schedule** | the logged flight plan accounts for most of the deviation | a medical workup instead of a re-plan |
| **the habitat** | several crew diverge the same day *in the same direction*, or a protocol limit is breached | four workups instead of one repair |
| **the instrument** | the residual concentrates on one device beyond its own normal share, and *no* latent state explains it | a workup on a loose electrode, and the crew muting the monitor next time |

The hardest of these is the schedule. A crew member ground down by twelve days
of short sleep and heavy workload deteriorates **exactly** like one becoming
ill, and a deviation-from-baseline monitor raises the same finding for both.
AURORA runs two forecasts from the same anchor — one on the logged inputs, one
on an ordinary day repeated — and reports the share of the deviation the plan
itself accounts for. Different number, different cause, different action.

### 7 · What-if, because that is the point of having a model

Forward Monte Carlo over the current state, the daily process noise, **and the
parameters themselves** — the third is usually omitted and it dominates at
longer horizons, because being wrong about how fast somebody recovers compounds
while today's measurement error does not. Change the sleep opportunity or the
training block and the twin answers "where is his HRV on Friday, and with what
confidence". The two cones share a random seed, so the difference between them
is the countermeasure rather than Monte Carlo noise.

### 8 · It says how much of itself is really this person

Twenty parameters against forty-five quiet days, and they are not equally
knowable from it. Where somebody's HRV sits, their resting rate, how
efficiently they sleep — a quiet month measures those precisely. How steeply
they respond to a heavy workload, how fast they shed load — a quiet month says
almost nothing, because nothing happened. **You cannot learn a stress response
from a period without stress.**

So each parameter is reported with the share of its posterior precision that
came from the population prior rather than from this person. When the dashboard
says FE-1 sheds load at 0.19/day, it also says that most of that number is
prior. A twin that hides that ratio is claiming to know the astronaut better
than the data permits, and every forecast inherits the overconfidence.

---

## Validation

`python3 -m aurora --validate` sweeps every scenario against the known injected
physiology. **Results are generated, not quoted — run it.**

<!-- RESULTS -->

### Specificity is the number that decides whether the rest matters

Measured on nominal missions with nothing wrong, pooled over independent seeds,
counted in **episodes rather than days** (a stable advisory running five days is
one thing for the crew to read, not five), and reported against a budget
declared before the measurement was taken.

### What went wrong first — all of it here

Each of these produced a confident, wrong answer, and each is why the test
suite exists.

1. **Declaring a baseline that was not a fixed point.** Intercepts were
   hand-written, so a quiet mission drifted away from its own stated normal and
   the drift was indistinguishable from a slow event. Now the intercepts are
   *solved* from the declared quiet-day readings.
2. **Deconditioning with positive feedback.** Lower fitness reduced completed
   training, which reduced fitness. Unit fitness was an *unstable* fixed point
   and a perfectly healthy crew member decayed to the floor over a long mission.
   The training stimulus is now measured against current capacity, which is the
   restoring force real physiology has.
3. **A feedback loop with too much gain.** The sleep↔load loop ran at gain ≈
   0.5 — formally stable, but it amplified every ordinary perturbation
   threefold, and a nominal mission with four EVAs drove autonomic load to 1.25.
   That is the model asserting a healthy crew member on a normal flight plan is
   in serious overload. Debt dissipation was the gain control.
4. **One RNG stream for the whole roster.** A sensor fault's extra draws shifted
   the measurement noise of everybody simulated *after* it, including over days
   before the fault existed. That silently invalidated the shared quiet window
   the twins are calibrated on.
5. **Latent axes with a single observer.** Psychological reserve was seen only
   through mood, and plasma-volume *surplus* only through body mass. The
   projection then explained any one noisy channel by moving the axis that
   channel happened to observe — producing a sustained, entirely false
   "psychological strain" finding on a healthy commander, built from nothing but
   a run of mood scores.
6. **A statistic that measured the horizon.** The residual projection grows with
   the forecast length — not from miscalibration, but because a longer open-loop
   run accumulates more genuine process noise and the projection is built to
   find exactly that drift. The same healthy commander scored 4.4 at three days
   and 14.5 at fourteen. The statistic is now studentised against that person's
   own quiet-window distribution *at the same horizon*.
7. **A kink at the nominal state.** The one-sided volume response put a
   non-differentiable corner at V = 0, which is exactly where a healthy
   astronaut sits — and both the filter's Jacobian and the residual projection
   linearise there. The two disagreed by a factor of 1.75 depending on step
   size.
8. **Two copies of one formula.** The intercept solver had its own copy of the
   volume response. The moment the surplus side gained a gain, the declared
   quiet-day readings stopped being a fixed point — by a few thousandths, in
   exactly the three channels that respond to volume.
9. **Testing an absolute device share.** Bio-Monitor carries nine of the twelve
   channels, so its share of any residual sits near 0.8 on a perfectly healthy
   day. Testing against an absolute 0.8 called almost every finding an
   instrument fault. Concentration only means something measured against what
   that device normally contributes.
10. **Naming an episode from its peak day.** One outlying mood score renamed a
    correctly-identified inflammatory response as behavioural strain; the days
    either side had it right. An episode is a claim about a stretch of
    physiology, so its name now comes from the stretch, score-weighted.

### Why the threshold is measured rather than derived

The statistic would be chi-squared if the residuals were independent and
Gaussian. They are not: measurement noise is autocorrelated, the state is
clipped at physiological bounds, and the model is wrong in ways the simulator
arranges on purpose. So the threshold comes from pooled nominal missions against
a declared false-alarm budget, and `--sweep` prints the trade it was chosen
from. Persistence does more work than any amount of threshold tuning: a healthy
crew member's worst day reaches the same score as a real event's third day, so
the threshold alone cannot separate them — requiring three hot days inside one
episode does, without delaying anything real, because a real finding was going
to produce them anyway. A protocol-limit breach is exempt: a solar particle
event has to be called on its first day.

---

## What it monitors

**Fourteen signals across four devices**, each declaring where it comes from —
because a twin fed a channel nobody can measure is a simulation, not a twin.

| device | channels |
|---|---|
| **Bio-Monitor** (Astroskin garment + forehead sensor) | HRV (RMSSD, nocturnal), resting heart rate, systolic blood pressure, SpO₂, breathing rate, skin temperature, sleep duration, sleep efficiency, activity completed |
| **SLAMMD** | body mass |
| **Active personal dosimeter** | dose rate, cumulative dose |
| **Crew self-report** | mood, perceived stress |

Device capabilities follow the real Bio-Monitor: a 3-lead ECG, thoracic and
abdominal respiration bands, a 3-axis accelerometer, skin temperature, a
forehead SpO₂ sensor, and a continuous estimate of arterial systolic pressure,
running about 48 hours per battery pair. What it does *not* measure is stated
just as carefully — **skin temperature is a proxy for core temperature, not a
substitute**, which matters exactly when the question is whether somebody is
running a fever.

Coverage is a variable, not an assumption: the garment comes off, batteries run
out, body mass needs a scheduled measurement, and questionnaires get skipped.
Missing days are **declared missing, never interpolated**.

---

## The simulator is deliberately not the twin

The lazy way to build a digital twin demo is to generate the data with the same
equations the twin uses. Detection then works beautifully and means nothing:
the residuals are measurement noise by construction, every event is visible the
instant it starts, and the "validation" tests that floating-point arithmetic is
deterministic.

So the simulated astronaut carries:

- **dynamics the model does not contain** — strain that amplifies itself once
  load is already high, sleep debt that bites with a lag a one-day step cannot
  represent, and a weekly rhythm in how the crew member *responds* as distinct
  from what the schedule asks;
- **autocorrelated measurement noise**, where the twin assumes white;
- **latent forcing with no input behind it** — the twin can see the sleep
  opportunity, the workload, the exercise prescription and the dose, and cannot
  see the forcing. That gap *is* the signal, and it is the right definition of a
  finding: a change in this person the plan does not account for.

The counterpart matters just as much, and it is the `explained_load` scenario:
there the crew member is genuinely hammered, but by the *schedule*, and the
schedule is an input.

---

## Data provenance

Stated plainly, because a health tool that overstates its inputs has already
failed.

- **Physiologies are synthetic**, with levels, variability and noise structure
  in published spaceflight ranges. They are not anybody's records.
- **Nothing is attributed to a named astronaut.** The Bio-Monitor framing is
  about what the *instrument* measures. Inventing medical data for a real crew
  member would be a different and much worse thing than modelling a device.
- **Every scenario is deterministically seeded**, per crew member, so the number
  on screen is the number that was rehearsed.
- **Countermeasure response is modelled, not observed.** A what-if changes the
  input schedule and re-runs the twin; it does not consult evidence about what
  that intervention achieves in real crews.
- **AURORA is decision support, not a diagnostic device.** It proposes; the crew
  medical officer and flight surgeon dispose.

---

## Architecture

```
aurora/
  signals.py      the observable surface: 14 channels, 4 devices, provenance
  model.py        7 latent systems, the dynamics, per-person parameters
  observe.py      latent state -> measurable channels, and the Jacobian
  crew.py         four astronauts, declared by their quiet-day readings
  truth.py        the mission that happened -- deliberately not the twin
  twin.py         the filter, the open-loop forecast, ensembles, what-ifs
  calibrate.py    fitting a twin, and reporting how much of it is prior
  divergence.py   the statistic, latent attribution, episode assembly
  session.py      the pipeline, assembled once so every caller agrees
  validate.py     detection latency and pooled specificity
  report.py       the whole state as one downlink-sized document
app.py            the dashboard
tests/            the invariants that make the rest mean anything
```

---

## It is not only about space

The same shape — a personal baseline instead of a population range, a model
driven by what was actually asked of the person, separating "deteriorating" from
"deteriorating for a reason we chose", and saying out loud how much of the model
is really them — is the problem in Antarctic stations, submarines, offshore
platforms, elite sport, intensive care, and the monitoring of any chronic
condition at home. Everywhere somebody has imperfect sensors, limited expertise
on site, and a slow link to anyone who knows more.

---

## Sources

- [Canadian Space Agency — Bio-Monitor](https://www.asc-csa.gc.ca/eng/sciences/space-flight-improve-monitor-health-earth.asp)
- [David Saint-Jacques on the Bio-Monitor smart shirt](https://www.asc-csa.gc.ca/eng/multimedia/search/video/18316)
- [Astroskin — Carré Technologies / Hexoskin](https://hexoskin.com/pages/astroskin-vital-signs-monitoring-platform-for-advanced-research)
- [NASA Human Research Roadmap](https://humanresearchroadmap.nasa.gov/)
- [NASA Open Science Data Repository](https://osdr.nasa.gov/)
