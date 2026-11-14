"""Invariants that, if broken, make the rest of AURORA meaningless.

Most of these are not about numerical accuracy. They are about the handful of
properties the whole argument rests on -- that the forecast cannot see the
measurements it is judged against, that the twin is calibrated on a window
containing no event, that a declared baseline is actually a fixed point. Each
one is cheap to get wrong in a way no chart would reveal, and two of them were
wrong at some point during this build.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.crew import CREW, CREW_ORDER, PROFILES, build_params
from aurora.divergence import HORIZONS, personal_bands
from aurora.model import (
    IDX, NOMINAL_INPUT, PROCESS_SD, STATE, UIDX, achieved_sleep, settle,
    sleep_efficiency, step,
)
from aurora.observe import as_vector, jacobian, predict
from aurora.signals import DEVICES, MONITORED, SIGNALS
from aurora.truth import CALIB_DAYS, SCENARIOS, simulate
from aurora.twin import open_loop, run_filter
from aurora.session import window_for


# --- the registry -------------------------------------------------------------

def test_every_monitored_signal_has_a_real_device():
    for ch in MONITORED:
        assert SIGNALS[ch].device in DEVICES, ch


def test_dose_cum_is_not_monitored():
    """A cumulative ledger carries every past residual with it, so monitoring
    it would turn one event into a permanent alarm."""
    assert "dose_cum" not in MONITORED
    assert "dose_cum" in SIGNALS


# --- the declared baseline really is a baseline -------------------------------

@pytest.mark.parametrize("prof", PROFILES, ids=[p.key for p in PROFILES])
def test_nominal_readings_are_a_fixed_point(prof):
    """Each crew member's declared quiet-day numbers must be an equilibrium of
    their own dynamics. If they are not, a quiet mission drifts away from its
    own stated baseline and the twin spends it detecting its initialisation."""
    p = build_params(prof)
    x = settle(p, NOMINAL_INPUT, days=8000)
    obs = predict(x, p, NOMINAL_INPUT)
    for ch, target in prof.targets.items():
        assert obs[ch] == pytest.approx(target, abs=1e-3), ch


@pytest.mark.parametrize("prof", PROFILES, ids=[p.key for p in PROFILES])
def test_unit_fitness_is_a_stable_fixed_point(prof):
    """Deconditioning must be self-limiting. With the training stimulus taken
    against an absolute maintenance load instead of current capacity, this
    loop has positive feedback and a healthy crew member decays to the floor."""
    p = build_params(prof)
    x = settle(p, NOMINAL_INPUT, days=8000)
    assert x[IDX["F"]] == pytest.approx(1.0, abs=1e-3)

    # Perturb and check it comes back rather than running away.
    x2 = x.copy()
    x2[IDX["F"]] = 0.90
    for _ in range(4000):
        x2 = step(x2, NOMINAL_INPUT, p)
    assert x2[IDX["F"]] > 0.97


def test_no_standing_sleep_debt_on_a_nominal_day():
    for c in CREW_ORDER:
        p = CREW[c].truth
        x = settle(p, NOMINAL_INPUT, days=8000)
        assert x[IDX["S"]] < 0.05, c


# --- the dynamics and the observation model agree -----------------------------

def test_sleep_efficiency_is_shared_between_dynamics_and_observation():
    """If the twin predicted one efficiency for the chart and used another to
    accumulate debt, the load-sleep feedback loop would be decorative."""
    p = CREW["fe1"].truth
    x = settle(p, NOMINAL_INPUT)
    assert predict(x, p, NOMINAL_INPUT)["sleep_eff"] == pytest.approx(
        sleep_efficiency(x, p))
    assert predict(x, p, NOMINAL_INPUT)["sleep_h"] == pytest.approx(
        achieved_sleep(x, p, NOMINAL_INPUT))


def test_jacobian_matches_finite_difference_of_predict():
    p = CREW["fe2"].truth
    x = settle(p, NOMINAL_INPUT)
    J = jacobian(x, p, NOMINAL_INPUT)
    h = 1e-4
    for j, name in enumerate(STATE):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        num = (as_vector(predict(xp, p, NOMINAL_INPUT))
               - as_vector(predict(xm, p, NOMINAL_INPUT))) / (2 * h)
        # Loose: `predict` clips to survivable ranges and several channels are
        # one-sided in V, so the two step sizes need only agree in direction
        # and rough magnitude.
        np.testing.assert_allclose(J[:, j], num, rtol=0.25, atol=0.05)


# --- causality: the part that makes a finding mean anything --------------------

def test_open_loop_forecast_cannot_see_the_measurements_it_is_judged_on():
    """The forecast is the twin's claim about what the plan implies. If a
    measurement from inside the forecast window could change it, the residual
    would be partly self-referential and the whole statistic would be circular.
    """
    m = simulate("overload", seed=7)
    w = window_for(m, "fe1")
    p = CREW["fe1"].truth
    fo = run_filter(w, p, sd_scale=CREW["fe1"].sd_scale)

    anchor, horizon = 60, 14
    a = open_loop(w, p, fo.x[anchor], fo.P[anchor], anchor, horizon,
                  sd_scale=CREW["fe1"].sd_scale)

    # Corrupt every measurement strictly after the anchor.
    y2 = w.y.copy()
    y2[anchor + 1:] = np.where(np.isfinite(y2[anchor + 1:]),
                               y2[anchor + 1:] * 1.5 + 7.0, np.nan)
    w2 = type(w)(u=w.u, y=y2, dose_cum=w.dose_cum, phase=w.phase,
                 gravity=w.gravity)
    b = open_loop(w2, p, fo.x[anchor], fo.P[anchor], anchor, horizon,
                  sd_scale=CREW["fe1"].sd_scale)

    np.testing.assert_allclose(a.yhat, b.yhat, rtol=0, atol=0)
    np.testing.assert_allclose(a.x, b.x, rtol=0, atol=0)
    # The residuals must move, or the test is checking nothing.
    assert not np.allclose(np.nan_to_num(a.resid), np.nan_to_num(b.resid))


def test_filter_state_at_a_day_does_not_depend_on_later_data():
    """The anchor state is taken from the filter. A filter that leaked future
    data would hand the forecast the answer."""
    m = simulate("overload", seed=7)
    w = window_for(m, "fe1")
    p = CREW["fe1"].truth
    cut = 62

    full = run_filter(w, p, sd_scale=CREW["fe1"].sd_scale)
    trunc = run_filter(w.slice(0, cut), p, sd_scale=CREW["fe1"].sd_scale)

    np.testing.assert_allclose(full.x[:cut], trunc.x, rtol=1e-10, atol=1e-12)


def test_personal_bands_use_only_the_calibration_window():
    m = simulate("overload", seed=7)
    w = window_for(m, "fe1")
    before = personal_bands(w, CALIB_DAYS, CREW["fe1"].sd_scale)

    y2 = w.y.copy()
    y2[CALIB_DAYS:] = np.where(np.isfinite(y2[CALIB_DAYS:]),
                               y2[CALIB_DAYS:] * 3.0, np.nan)
    w2 = type(w)(u=w.u, y=y2, dose_cum=w.dose_cum, phase=w.phase,
                 gravity=w.gravity)
    after = personal_bands(w2, CALIB_DAYS, CREW["fe1"].sd_scale)
    assert before == after


# --- the simulation is honest about what it shares ---------------------------

def test_calibration_window_contains_no_event_in_any_scenario():
    """Calibration fits on the opening stretch. An event inside it would be
    learned as this person's normal, and the twin would then faithfully report
    that nothing is wrong."""
    for key, sc in SCENARIOS.items():
        for ev in sc.events:
            assert ev.onset >= CALIB_DAYS, (key, ev.label)


def test_quiet_window_is_identical_across_scenarios():
    """Calibration is cached per crew member on the hash of this window, and
    comparisons between scenarios are only meaningful if the twins are the
    same. Sharing one RNG across the roster broke this: a sensor fault's extra
    draws shifted the noise of everybody simulated after it."""
    base = simulate("nominal", seed=7)
    for key in SCENARIOS:
        other = simulate(key, seed=7)
        for c in CREW_ORDER:
            for ch in MONITORED:
                a = np.nan_to_num(base.obs[c][ch][:CALIB_DAYS], nan=-9e9)
                b = np.nan_to_num(other.obs[c][ch][:CALIB_DAYS], nan=-9e9)
                np.testing.assert_allclose(a, b, rtol=0, atol=0,
                                           err_msg=f"{key}/{c}/{ch}")


def test_one_crew_members_event_does_not_touch_another_crew_members_data():
    nom = simulate("nominal", seed=7)
    flt = simulate("sensor_fault", seed=7)        # injected on fe2 only
    for c in CREW_ORDER:
        if c == "fe2":
            continue
        for ch in MONITORED:
            np.testing.assert_allclose(
                np.nan_to_num(nom.obs[c][ch], nan=-9e9),
                np.nan_to_num(flt.obs[c][ch], nan=-9e9),
                rtol=0, atol=0, err_msg=f"{c}/{ch}")


def test_nominal_mission_does_not_drive_anybody_into_overload():
    """A model that says a healthy crew member on an ordinary flight plan is in
    serious autonomic overload has a gain problem, not a finding."""
    m = simulate("nominal", seed=7)
    for c in CREW_ORDER:
        A = m.latent(c, "A")
        assert A.mean() < 0.60, (c, A.mean())
        assert A.max() < 1.00, (c, A.max())


def test_measurement_gaps_are_declared_not_filled():
    m = simulate("nominal", seed=7)
    total = sum(int(np.isnan(m.obs[c][ch]).sum())
                for c in CREW_ORDER for ch in MONITORED)
    assert total > 0, "a mission with no missing readings is not a mission"


# --- the detector ------------------------------------------------------------

def test_horizons_are_within_the_measured_null_range():
    """`ResidCal` measures the null for horizons 3..14; a scan horizon outside
    that range would be studentised against the nearest neighbour instead of
    its own distribution."""
    assert min(HORIZONS) >= 3
    assert max(HORIZONS) <= 14


def test_process_noise_is_not_negligible():
    """A twin with no process noise claims the body is clockwork, and then
    every ordinary day looks like an event."""
    assert np.all(PROCESS_SD > 0)
    assert PROCESS_SD[IDX["A"]] > 0.01
