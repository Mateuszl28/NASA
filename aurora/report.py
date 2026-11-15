"""The whole state as one document, sized for a downlink.

Everything else in AURORA lives on a screen somebody has to be sitting in
front of. That assumption breaks twice: a flight surgeon picking up a crew
months into a transit needs one document, and so does the next rotation.

Markdown rather than PDF, for reasons that matter at Mars range -- a few
kilobytes, survives a lossy link, diffs against yesterday's, and stays readable
in a terminal when the renderer is the thing that broke.
"""

from __future__ import annotations

import numpy as np

from .crew import CREW
from .model import INPUT_NAMES, INPUT_UNITS, STATE, STATE_NAMES, STATE_UNITS
from .session import Session
from .signals import DEVICES, MONITORED, SIGNALS
from .twin import explain_load


def _f(v: float, n: int = 2) -> str:
    return "--" if not np.isfinite(v) else f"{v:.{n}f}"


def render(s: Session, *, day: int | None = None) -> str:
    d = s.days - 1 if day is None else day
    L: list[str] = []

    L.append(f"# AURORA mission report -- day {d}")
    L.append("")
    L.append(f"Scenario **{s.mission.scenario.title}** "
             f"(`{s.mission.scenario.key}`), seed {s.mission.seed}, "
             f"{s.days} days, {len(s.views)} crew.")
    L.append("")
    L.append("Decision support, not a diagnostic device: AURORA proposes and "
             "the crew medical officer and flight surgeon dispose. Physiology "
             "here is simulated -- see the provenance section.")
    L.append("")

    # --- findings -------------------------------------------------------------
    L.append("## Findings")
    L.append("")
    open_now = [f for f in s.findings if f.start <= d <= f.end + 3]
    if not s.findings:
        L.append("None raised. Detection threshold "
                 f"{s.threshold:.1f} studentised units.")
    for f in s.findings:
        state = "OPEN" if f in open_now else "closed"
        L.append(f"### {CREW[f.crew].role} -- {f.pattern}  _({state})_")
        L.append("")
        L.append(f"- days {f.start}-{f.end} ({f.length} d), peak "
                 f"{f.peak_score:.1f} on day {f.peak_day}")
        L.append(f"- attributed to: **{f.attribution}**")
        if f.plan_share > 0.05:
            L.append(f"- the logged flight plan accounts for "
                     f"{100 * f.plan_share:.0f}% of it")
        if f.in_band_at_start:
            L.append("- every channel was still inside its own personal "
                     "3-sigma band when this was raised")
        if f.limit_breach:
            L.append(f"- protocol limit breached: {', '.join(f.limit_breach)}")
        if f.device_q:
            ranked = sorted(f.device_q.items(), key=lambda kv: -kv[1])
            L.append("- device loudness (1.0 = behaving): " + ", ".join(
                f"{DEVICES[d].name.split(' (')[0]} {q:.1f}" for d, q in ranked))
        top = sorted(f.latent_z.items(), key=lambda kv: -abs(kv[1]))[:3]
        L.append("- latent shift: " + ", ".join(
            f"{STATE_NAMES[k]} {v:+.1f}" for k, v in top))
        if f.pattern_note:
            L.append("")
            L.append(f"  {f.pattern_note}")
        L.append("")

    # --- per crew member ------------------------------------------------------
    L.append("## Crew")
    L.append("")
    for c, v in s.views.items():
        a = CREW[c]
        L.append(f"### {a.role} (`{c}`)")
        L.append("")
        L.append(f"{a.note}")
        L.append("")

        x = v.filt.x[d]
        L.append("| latent system | estimate | unit |")
        L.append("|---|---|---|")
        for k in STATE:
            L.append(f"| {STATE_NAMES[k]} | {x[STATE.index(k)]:+.3f} | "
                     f"{STATE_UNITS[k] or '--'} |")
        L.append("")

        terms = [(k, t) for k, t in explain_load(x, v.calib.params, v.window.u[d])
                 if t > 0.005]
        if terms:
            L.append("What is costing them today, in units of autonomic load: "
                     + ", ".join(f"{k} {t:.2f}" for k, t in terms) + ".")
            L.append("")

        ds = v.day(d)
        if ds:
            L.append(f"Divergence score {ds.score:.1f} "
                     f"(threshold {s.threshold:.1f}), best at a "
                     f"{ds.horizon}-day forecast horizon.")
            L.append("")

        L.append("| channel | reading | twin | z | device |")
        L.append("|---|---|---|---|---|")
        for j, ch in enumerate(MONITORED):
            y = v.window.y[d, j]
            sig = SIGNALS[ch]
            z = (ds.channel_z.get(ch, np.nan) if ds else np.nan)
            L.append(f"| {sig.name} | {_f(y)} {sig.unit} | -- | {_f(z, 1)} | "
                     f"{sig.device} |")
        L.append("")

        cal = v.calib
        L.append(f"Calibrated on {cal.days} quiet days, {cal.n_obs} readings. "
                 f"{len(cal.data_driven)} of {len(cal.fitted)} parameters were "
                 f"determined by this person's own data; the rest are "
                 f"population priors and are listed as such:")
        L.append("")
        for k in cal.prior_driven:
            L.append(f"- `{k}` = {getattr(cal.params, k):.4g} "
                     f"({100 * cal.prior_share[k]:.0f}% prior)")
        L.append("")

    # --- instruments ----------------------------------------------------------
    L.append("## Instruments")
    L.append("")
    for key, dev in DEVICES.items():
        chans = [SIGNALS[c].name for c in MONITORED
                 if SIGNALS[c].device == key]
        L.append(f"- **{dev.name}** -- {dev.note}")
        if chans:
            L.append(f"  - {', '.join(chans)}")
    L.append("")

    # --- provenance -----------------------------------------------------------
    L.append("## Provenance")
    L.append("")
    L.append("- Device capabilities follow the Canadian Space Agency's "
             "Bio-Monitor, the Astroskin-derived garment built by Carre "
             "Technologies that was commissioned aboard the ISS in January "
             "2019: 3-lead ECG, thoracic and abdominal respiration bands, "
             "3-axis accelerometer, skin temperature, forehead SpO2, and a "
             "continuous estimate of arterial systolic pressure.")
    L.append("- Skin temperature is a proxy for core temperature, not a "
             "substitute, and is labelled as such throughout.")
    L.append("- Physiologies are synthetic, with levels and variability in "
             "published spaceflight ranges. They are not anybody's records, "
             "and nothing here is attributed to a named astronaut.")
    L.append("- The simulated astronaut is deliberately not the twin: it "
             "carries dynamics the model does not contain, autocorrelated "
             "measurement noise, and latent forcing with no input behind it. "
             "Generating the data with the model being tested would make "
             "every detection a tautology.")
    L.append("- The detection threshold is measured against a declared "
             "false-alarm budget on pooled nominal missions, not taken from a "
             "distributional table the residuals do not follow.")
    L.append("")
    return "\n".join(L)
