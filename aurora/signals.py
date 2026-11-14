"""The observable surface: what AURORA can actually measure, and from what.

Every signal declares the physical device it comes from. That matters for two
reasons. A twin fed a channel nobody can measure is a simulation, not a twin --
so the registry is the contract that keeps the model honest about its inputs.
And when a residual pattern turns out to be confined to one device's channels,
the explanation is the device, not the astronaut (see `divergence.attribute`).

Device capabilities are taken from the Canadian Space Agency's Bio-Monitor, the
Astroskin-derived garment built by Carre Technologies that David Saint-Jacques
commissioned aboard the ISS in January 2019: a 3-lead ECG, thoracic and
abdominal respiration bands, a 3-axis accelerometer, skin temperature, a
forehead SpO2 sensor, and a continuous estimate of arterial systolic pressure.

What Bio-Monitor does *not* measure is stated just as carefully. Body mass needs
an inertial device, dose needs a dosimeter, and how somebody feels has to be
asked. Those arrive on their own cadence and with their own failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- devices -----------------------------------------------------------------

@dataclass(frozen=True)
class Device:
    key: str
    name: str
    note: str


DEVICES: dict[str, Device] = {
    "biomonitor": Device(
        "biomonitor",
        "Bio-Monitor (Astroskin garment + forehead sensor)",
        "CSA/Carre Technologies. Continuous while worn; 48 h per battery pair, "
        "so coverage is a real variable rather than an assumption.",
    ),
    "slammd": Device(
        "slammd",
        "Space Linear Acceleration Mass Measurement Device",
        "Inertial body-mass measurement. Scheduled, not continuous.",
    ),
    "dosimeter": Device(
        "dosimeter",
        "Active personal dosimeter",
        "Radiation dose rate and accumulated dose.",
    ),
    "selfreport": Device(
        "selfreport",
        "Crew self-report",
        "Mood and perceived stress. The signals crews under-report, and the "
        "only ones the garment cannot infer.",
    ),
}


# --- signals -----------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    key: str
    name: str
    unit: str
    device: str
    #: Which way is bad. Used for plain-language framing, never for the maths.
    adverse: str  # "low" | "high" | "both"
    #: Physiologically survivable range. Values outside this are data faults,
    #: not findings -- a 540 bpm heart rate is a broken lead, and letting it
    #: reach a reference window would blind the twin for a fortnight.
    valid: tuple[float, float]
    #: Rough measurement resolution, used to floor noise estimates so that a
    #: quantised channel cannot present as impossibly precise.
    resolution: float
    source: str


SIGNALS: dict[str, Signal] = {s.key: s for s in [
    # --- Bio-Monitor: cardiovascular and autonomic ---------------------------
    Signal("hrv_rmssd", "HRV (RMSSD, nocturnal)", "ms", "biomonitor", "low",
           (2.0, 220.0), 0.5,
           "Derived from the 3-lead ECG over the main sleep period."),
    Signal("hr_rest", "Resting heart rate", "bpm", "biomonitor", "high",
           (25.0, 190.0), 0.5,
           "ECG, lowest sustained 5-minute mean while recumbent."),
    Signal("bp_sys", "Systolic blood pressure", "mmHg", "biomonitor", "both",
           (60.0, 220.0), 1.0,
           "Bio-Monitor's continuous arterial systolic estimate."),
    Signal("spo2", "SpO2", "%", "biomonitor", "low",
           (60.0, 100.0), 0.1,
           "Forehead pulse oximetry."),
    Signal("br_rate", "Breathing rate", "breaths/min", "biomonitor", "both",
           (4.0, 40.0), 0.1,
           "Thoracic and abdominal respiration bands."),
    Signal("skin_temp", "Skin temperature", "degC", "biomonitor", "both",
           (28.0, 41.0), 0.01,
           "Garment thermistor. A proxy for core temperature, not a "
           "substitute for it -- stated because the difference matters when "
           "the question is whether somebody is running a fever."),

    # --- Bio-Monitor: sleep and activity ------------------------------------
    Signal("sleep_h", "Sleep duration", "h", "biomonitor", "low",
           (0.0, 14.0), 0.05,
           "Actigraphy and cardiorespiratory staging over the sleep period."),
    Signal("sleep_eff", "Sleep efficiency", "%", "biomonitor", "low",
           (0.0, 100.0), 0.5,
           "Asleep time as a fraction of time in the sleep opportunity."),
    Signal("activity", "Activity load achieved", "min", "biomonitor", "low",
           (0.0, 400.0), 1.0,
           "Accelerometry, intensity-weighted. What was *done*, as opposed to "
           "what the plan asked for."),

    # --- other devices -------------------------------------------------------
    Signal("body_mass", "Body mass", "kg", "slammd", "both",
           (35.0, 150.0), 0.1,
           "SLAMMD. Scheduled measurement, roughly every third day."),
    Signal("dose_rate", "Dose rate", "uGy/h", "dosimeter", "high",
           (0.0, 500000.0), 0.1,
           "Active personal dosimeter."),
    Signal("dose_cum", "Cumulative dose", "mGy", "dosimeter", "high",
           (0.0, 100000.0), 0.001,
           "Integrated exposure since launch. Monotone by construction, so it "
           "is a ledger rather than a detection channel."),
    Signal("mood", "Mood", "1-7", "selfreport", "low",
           (1.0, 7.0), 0.1,
           "Daily single-item scale, 7 best."),
    Signal("stress", "Perceived stress", "1-7", "selfreport", "high",
           (1.0, 7.0), 0.1,
           "Daily single-item scale, 7 worst."),
]}


#: Signals the divergence monitor scores. `dose_cum` is excluded because it is a
#: cumulative ledger -- its residual carries every past residual with it, which
#: would quietly turn one event into a permanent alarm. `dose_rate` is the
#: environment acting *on* the astronaut rather than a property of the
#: astronaut, so it enters the twin as an input and is monitored against its own
#: protocol limit instead.
MONITORED: tuple[str, ...] = (
    "hrv_rmssd", "hr_rest", "bp_sys", "spo2", "br_rate", "skin_temp",
    "sleep_h", "sleep_eff", "activity", "body_mass", "mood", "stress",
)

#: Protocol limits that hold regardless of whose body it is -- the handful of
#: places where a personal baseline is the wrong instrument, because the number
#: is about survivability rather than about deviation from habit.
ABSOLUTE_LIMITS: dict[str, dict[str, float]] = {
    "spo2":      {"watch": 94.0, "alert": 90.0},     # falling
    "dose_rate": {"watch": 40.0, "alert": 200.0},    # rising, uGy/h
    "skin_temp": {"watch": 37.6, "alert": 38.2},     # rising
}


def signals_of(device: str) -> tuple[str, ...]:
    """Every monitored signal carried by one physical device."""
    return tuple(k for k in MONITORED if SIGNALS[k].device == device)


def device_of(signal: str) -> Device:
    return DEVICES[SIGNALS[signal].device]
