# AURORA — a digital twin of the astronaut

**Every astronaut gets their own computational model of their own body, and is
monitored against themselves rather than against a population.**

Built around the Canadian Space Agency's **Bio-Monitor** — the Astroskin-derived
garment that David Saint-Jacques commissioned aboard the ISS in January 2019 —
AURORA takes the twelve things that instrument and a dosimeter can actually
measure, infers the seven physiological systems behind them, and answers the
question a chart cannot: **not whether something is wrong, but what and whose
fault it is.**

```
FE-1 · Flight Engineer 1 · day 65                           divergence 168
┌──────────────────────────────────────────────────────────────────────────┐
│ AUTONOMIC OVERLOAD — not accounted for by the flight plan                │
│                                                                          │
│                     quiet   now    this person's own ±3σ band            │
│   HRV                80.5   33.4   55.2 — 97.2                           │
│   resting HR         61.8   69.1   57.1 — 67.1                           │
│   sleep efficiency   88.9   71.9   73.2 — 100.2                          │
│   activity completed  105     32   61.6 — 152.3                          │
│                                                                          │
│ On day 65 every one of those readings is still inside its own band. The  │
│ finding is not any single excursion — it is that all four moved together │
│ in the direction one latent variable predicts.                           │
│                                                                          │
│   latent shift    autonomic load +3.1σ                                   │
│   attribution     the astronaut — the logged schedule accounts for 15 %  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Run it

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/streamlit run app.py          # the dashboard
.venv/bin/python -m aurora --validate   # the validation sweep (the table below)
.venv/bin/python -m aurora --scenarios  # what each mission is for
.venv/bin/python -m aurora --report     # the whole state as one document
.venv/bin/python -m pytest tests/ -q
```

The first run fits each twin and takes a couple of minutes; fits are cached
under `data/twins/`, keyed by a hash of the window they were fitted on, so a
stale cache is ignored rather than silently wrong.

---

## The idea, in one paragraph

HRV, resting heart rate, sleep efficiency and completed exercise are not four
independent measurements that happen to move together. In this model they are
four *projections of one number* — autonomic load — with different gains and
different signs. So a coherent half-sigma movement across all four is a large
change in that one number, while four independent half-sigma wobbles are
nothing. AURORA models the person, not the charts, which is what lets it say
*which system moved* instead of *which line looks odd*.

---

## What the measurements actually show

Run `--validate` and you get this. It is generated, not quoted.

| scenario | AURORA | latency | cause identified | every channel still in band | a personal ±3σ monitor | AURORA's lead |
|---|---|---|---|---|---|---|
| Autonomic overload | day 65 | +3 | **the astronaut** ✓ | yes | day 67 | **+2 d** |
| A brutal week the schedule explains | day 74 | +12 | **the schedule** ✓ | — | day 65 | −9 d |
| Plasma volume loss | day 72 | +14 | **the astronaut** ✓ | yes | day 59 | −13 d |
| Early subclinical infection | day 73 | +7 | **the astronaut** ✓ | yes | day 71 | −2 d |
| Solar particle event | day 70 | +0 | **the habitat** ✓ | yes | day 71 | +1 d |
| Bio-Monitor electrode fault | day 66 | +2 | **the instrument** ✓ | — | day 65 | −1 d |

```
specificity, pooled over 4 independent nominal missions
  false findings : 16 episode(s) over 1200 crew-days (146 advisory-days)
  episodes/seed  : [2, 2, 7, 5]
  in plain terms : 1.33 false findings per crew member per 100 days
                   — about 13 over a 1000-day Mars mission
  budget         : PASS (1.33 vs 2.00 per 100 crew-days allowed)

RESULT: PASS (6/6 scenarios, specificity within budget)
```

### Read the lead column honestly

**AURORA is not reliably earlier than a well-built threshold monitor, and the
measurement is in the table.** Two scenarios it leads by a day or two; three it
trails, one of them by a fortnight. An earlier draft of this README claimed nine
days of lead, read off a trajectory by eye. The first measurement said two. The
second said *minus fifteen* — until the dates were checked and it turned out to
be timing the threshold monitor's first **false** alarm rather than its
detection, because twelve channels under a ±3σ band estimated from 45 days leave
that band on noise a couple of times a mission anyway. Both numbers are now
measured properly and both are reported.

So timing is roughly a wash. **The column that is not a wash is the cause.**

- A ±3σ band catches the electrode fault on day 65 — and calls it a collapsing
  heart. AURORA catches it on day 66 and says *the garment has lost skin
  contact*. One of those costs a medical workup, consumables, and the crew's
  trust in the monitor the next time it speaks.
- The band catches the punishing week on day 65 and raises the same medical
  finding it raises for illness. AURORA says the logged flight plan accounts for
  60 % of it and attributes it to the **schedule**. Re-plan the week; do not
  examine the crew member.
- The band tells you a channel is out of range. It cannot tell you that four
  channels moved as one thing, which system that thing is, or whether the cause
  is the body, the roster, the habitat or the hardware.

That is the actual claim, it is the one the architecture is built for, and
unlike earliness it holds 6/6.

---

## How it works

### 1 · A latent state, not a pile of signals

Seven unobservable quantities — autonomic load, sleep debt, plasma volume,
aerobic fitness, inflammatory activation, circadian misalignment, psychological
reserve — are mapped onto twelve measurable channels. Seven against twelve keeps
the map over-determined, and that is what makes a residual mean something: if
the readings can no longer be explained by *any* setting of seven numbers,
something is happening that this model of this person does not contain.

Three couplings carry most of the behaviour:

- **sleep → efficiency → debt → load → sleep.** Load degrades sleep efficiency,
  which turns a fixed sleep opportunity into less actual sleep, which deepens
  the debt that drives load.
- **load → completed activity.** Exercise is *prescribed* as an input but only
  partly *completed*, so achieved activity is an output of the state. A quiet
  drop in finished training is one of the earliest things this loop produces,
  and it happens before anybody reports feeling bad.
- **load as a smoothed index.** Autonomic load is an exponential moving average
  of an instantaneous load demand, so every gain reads directly as "the
  steady-state load this cost produces" and the recovery rate means only "how
  fast this person sheds it".

### 2 · Each crew member is declared by what their charts read on a quiet day

Four astronauts, none interchangeable. FE-2's rested HRV is 44 ms; FE-3's is
103 ms. A reading of 60 ms is a good night for one and a developing problem for
the other, and no population percentile can tell you which.

Each profile is specified by the numbers a flight surgeon would actually have —
quiet-day readings and response steepness — and the model's intercepts are then
*solved* so those readings are a fixed point of the dynamics. Declaring the
intercepts directly was the first version and it was wrong: an intercept that is
not an equilibrium means a quiet mission drifts away from its own stated
baseline, and the twin spends the mission detecting its own initialisation.

### 3 · Estimate and forecast are two different things

An extended Kalman filter maintains the best current guess at the hidden state
with an honest covariance. That alone cannot produce a finding, because a
filter's job is to explain the data: hand it a crew member whose HRV is
collapsing and it will dutifully raise its autonomic-load estimate and report
small, well-behaved residuals. Right about the state, silent about the event.

So detection uses a second, deliberately handicapped prediction — an **open-loop
forecast**, anchored days back and driven *only by the logged flight-plan
inputs*, with the intervening measurements withheld. It answers: given what we
asked this person to do, what should they look like today? The gap between the
two is the finding.

### 4 · The residual is read in latent space, which is what names it

The standardised residual is projected onto the latent axes: *what single change
in load, volume, inflammation, debt, fitness, phase or reserve would best
explain all twelve channels at once?* Because the axes are physiological
systems, the dominant component **is** the finding. Falling HRV with a rising
heart rate and a *falling* systolic pressure is plasma volume; the same
heart-rate rise with a *rising* pressure is autonomic load. A chart cannot
separate those.

### 5 · Three horizons, because degradation has three speeds

Every day is scanned at a 3-, 7- and 14-day forecast length and the strongest
wins. One length cannot do the job: autonomic overload develops over three days
and is loudest in a short window, while a plasma-volume deficit develops over
twelve, so any three-day window holds a quarter of its change — and the filtered
anchor three days back has already absorbed part of it, leaving the forecast
predicting the very drift it was meant to detect. At three days the 14 % volume
loss scored 5.4 and was missed. At fourteen it is unmistakable.

### 6 · Four candidate explanations, not one

| attribution | evidence | what it costs to get wrong |
|---|---|---|
| **the astronaut** | a latent change explains it, confined to one crew member | — |
| **the schedule** | the logged flight plan accounts for most of the deviation | a medical workup instead of a re-plan |
| **the habitat** | several crew diverge the same day *in the same direction*, or a protocol limit is breached | four workups instead of one repair |
| **the instrument** | one device is loud while every other device is quiet | a workup on a loose electrode, and a muted monitor next time |

The instrument rule went through two wrong versions worth recording. Asking
whether the residual is *concentrated* on one device fails, because Bio-Monitor
carries nine of twelve channels and so holds ~80 % of any residual on a
perfectly healthy day. Adding "and no latent state explains it" worked only
while the latent prior was tight enough to prevent a good fit — seven free
dimensions against twelve channels fit almost anything, so once the prior was
loosened to let a real plasma-volume deficit through, a collapsed electrode
became explainable too and the rule went silent.

What actually separates them is physiological. A body in autonomic collapse
moves its mood, its perceived stress and its mass — channels from a
questionnaire and an inertial device, not from the garment. An electrode that
has lost skin contact cannot touch any of them. So the test is for *other
devices staying quiet*, which is only askable because residuals are standardised
per channel per forecast step: a device with nothing wrong sits near 1.0
regardless of how many channels it carries.

### 7 · What-if, because that is the point of having a model

Forward Monte Carlo over the current state, the daily process noise, **and the
parameters themselves** — the third is usually omitted and it dominates at
longer horizons, because being wrong about how fast somebody recovers compounds
while today's measurement error does not. The two cones share a random seed, so
the difference between them is the countermeasure rather than Monte Carlo noise.

On FE-1 at the peak of the overload episode (day 78), one extra hour of sleep
opportunity and a tenth off the task load returns **+7.9 ms of HRV at 14 days**
— and does **not** restore the quiet-window average of 76.2 ms within that
horizon. That second half is the more useful sentence: the intervention helps
and is not sufficient, which is a thing worth knowing before promising a flight
surgeon that protecting someone's sleep will fix it.

### 8 · It says how much of itself is really this person

Twenty parameters against forty-five quiet days, and they are not equally
knowable from it. Where somebody's HRV sits, their resting rate, how efficiently
they sleep — a quiet month measures those well: on FE-1 the fitted resting rate,
systolic pressure, body mass and sleep efficiency land within 0.2–0.7 % of truth,
and the HRV level within 3.9 %. How steeply they respond to a heavy workload, how
fast they shed load — a quiet month says almost nothing, because nothing
happened. **You cannot learn a stress response from a period without stress.**

So each parameter is reported with the share of its posterior precision that
came from the population prior rather than from this person. For FE-1, recovery
rate is 100 % prior and is labelled as such on screen. A twin that hides that
ratio is claiming to know the astronaut better than the data permits, and every
forecast inherits the overconfidence.

---

## What went wrong first — all of it here

Each of these produced a confident, wrong answer, and each is why the test suite
exists.

1. **Declaring a baseline that was not a fixed point.** Intercepts were
   hand-written, so a quiet mission drifted away from its own stated normal and
   the drift was indistinguishable from a slow event.
2. **Deconditioning with positive feedback.** Lower fitness reduced completed
   training, which reduced fitness. Unit fitness was an *unstable* fixed point
   and a healthy crew member decayed to the floor over a long mission. The
   training stimulus is now measured against current capacity, which is the
   restoring force real physiology has.
3. **A feedback loop with too much gain.** The sleep↔load loop ran at gain ≈ 0.5
   — formally stable, but it amplified every ordinary perturbation threefold,
   and a nominal mission with four EVAs drove autonomic load to 1.25. That is
   the model asserting a healthy crew member on a normal flight plan is in
   serious overload.
4. **One RNG stream for the whole roster.** A sensor fault's extra draws shifted
   the measurement noise of everybody simulated *after* it, including over days
   before the fault existed — silently invalidating the shared quiet window the
   twins are calibrated on.
5. **Latent axes with a single observer.** Psychological reserve was seen only
   through mood, and plasma-volume *surplus* only through body mass. The
   projection then explained any one noisy channel by moving the axis that
   channel happened to observe, producing a sustained, entirely false
   "psychological strain" finding on a healthy commander built from nothing but
   a run of mood scores.
6. **A statistic that measured the horizon.** The residual projection grows with
   forecast length — not from miscalibration, but because a longer open-loop run
   accumulates more genuine process noise and the projection is built to find
   exactly that drift. The same healthy commander scored 4.4 at three days and
   14.5 at fourteen. The statistic is now studentised against that person's own
   quiet-window distribution *at the same horizon*.
7. **A kink at the nominal state.** The one-sided volume response put a
   non-differentiable corner at V = 0, which is exactly where a healthy
   astronaut sits — and both the filter's Jacobian and the residual projection
   linearise there. They disagreed by a factor of 1.75 depending on step size.
8. **Two copies of one formula.** The intercept solver had its own copy of the
   volume response. The moment the surplus side gained a gain, the declared
   quiet-day readings stopped being a fixed point — by thousandths, in exactly
   the three channels that respond to volume.
9. **Naming an episode from its peak day.** One outlying mood score renamed a
   correct inflammatory finding as behavioural strain. Averaging the whole run
   instead failed the other way: an episode outlives its event, so a 26-day
   episode over a 14-day deficit spent half its length averaging in aftermath.
   Episodes are now named from their highest-scoring core.
10. **Letting noise name an environmental event.** A dose rate thirteen times
    its alert is called on day one, by design — but on day one the physiology
    has not responded, so the projection named the solar particle event
    "behavioural / psychological strain". A protocol breach now names itself.
11. **Quoting a lead time read off a chart.** See the table above. Twice.

### Why the threshold is measured rather than derived

The statistic would be chi-squared if the residuals were independent and
Gaussian. They are not: measurement noise is autocorrelated, the state is
clipped at physiological bounds, and the simulator is wrong in ways arranged on
purpose. So the threshold comes from pooled nominal missions against a declared
budget, and `--sweep` prints the trade it was chosen from.

Persistence does more work than any amount of threshold tuning: a healthy crew
member's worst day reaches the same score as a real event's third day, so the
threshold alone cannot separate them — requiring three hot days inside one
episode does, without delaying anything real, because a real finding was going
to produce them anyway. A protocol-limit breach is exempt.

---

## What it monitors

**Fourteen signals across four devices**, each declaring where it comes from,
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
out, body mass needs a scheduled measurement, questionnaires get skipped.
Missing days are **declared missing, never interpolated**.

---

## The simulator is deliberately not the twin

The lazy way to build a digital twin demo is to generate the data with the same
equations the twin uses. Detection then works beautifully and means nothing: the
residuals are measurement noise by construction, every event is visible the
instant it starts, and the "validation" tests that floating-point arithmetic is
deterministic.

So the simulated astronaut carries:

- **dynamics the model does not contain** — strain that amplifies itself once
  load is already high, sleep debt that bites with a lag a one-day step cannot
  represent, and a weekly rhythm in how the crew member *responds* as distinct
  from what the schedule asks;
- **autocorrelated measurement noise**, where the twin assumes white;
- **latent forcing with no input behind it** — the twin sees the sleep
  opportunity, the workload, the exercise prescription and the dose, and cannot
  see the forcing. That gap *is* the signal, and it is the right definition of a
  finding: a change in this person the plan does not account for.

This has a visible cost, and it is the `explained_load` result. The twin was
given the short sleep as an input and still raises a finding, because the
simulated body amplifies its own strain in a way a linear load index cannot
express — so the astronaut genuinely is worse than even a punishing schedule
predicts. The honest output is not silence but *a tenth the score of the
overload case, attributed to the schedule*.

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
  validate.py     detection, attribution, lead time, pooled specificity
  report.py       the whole state as one downlink-sized document
app.py            the dashboard
tests/            the invariants that make the rest mean anything
```

---

## It is not only about space

The same shape — a personal baseline instead of a population range, a model
driven by what was actually asked of the person, separating "deteriorating" from
"deteriorating for a reason we chose", distinguishing a failing sensor from a
failing body, and saying out loud how much of the model is really them — is the
problem in Antarctic stations, submarines, offshore platforms, elite sport,
intensive care, and the home monitoring of any chronic condition. Everywhere
somebody has imperfect sensors, limited expertise on site, and a slow link to
anyone who knows more.

---

## Sources

- [Canadian Space Agency — Bio-Monitor](https://www.asc-csa.gc.ca/eng/sciences/space-flight-improve-monitor-health-earth.asp)
- [David Saint-Jacques on the Bio-Monitor smart shirt](https://www.asc-csa.gc.ca/eng/multimedia/search/video/18316)
- [Astroskin — Carré Technologies / Hexoskin](https://hexoskin.com/pages/astroskin-vital-signs-monitoring-platform-for-advanced-research)
- [NASA Human Research Roadmap](https://humanresearchroadmap.nasa.gov/)
- [NASA Open Science Data Repository](https://osdr.nasa.gov/)
