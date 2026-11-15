"""AURORA dashboard.

    streamlit run app.py

Five views, in the order somebody actually asks the questions:

  Twin        the headline -- what was measured against what the plan predicted
  Systems     the seven latent variables, and what is costing this person today
  Findings    what was raised, why, and whose problem it is
  What if     change the schedule, re-run the twin, read the consequence
  Calibration how much of this twin is really this person

Design notes that are not decoration. Every chart has one y-axis; where two
quantities matter they get two charts rather than two scales, because a
dual-axis chart lets the author choose the correlation the reader sees. Series
colors come from a validated categorical order and are assigned to crew members
rather than to rank, so filtering the roster never repaints anybody. Status
colors are reserved for finding severity and always carry a word, never hue
alone.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aurora.crew import CREW, CREW_ORDER
from aurora.divergence import HORIZONS, personal_bands
from aurora.model import (
    INPUT_NAMES, INPUT_RANGE, INPUT_UNITS, INPUTS, NOMINAL_INPUT, STATE,
    STATE_NAMES, STATE_UNITS, UIDX,
)
from aurora.session import DEFAULT_THRESHOLD, build
from aurora.signals import DEVICES, MONITORED, SIGNALS
from aurora.truth import SCENARIOS
from aurora.twin import counterfactual, explain_load, open_loop, plan_matrix

st.set_page_config(page_title="AURORA", page_icon="*", layout="wide")


# --- theme -------------------------------------------------------------------
# Both modes are selected rather than flipped: the dark steps are the same hues
# re-stepped for the dark surface.

def theme() -> dict[str, str | list[str]]:
    dark = str(st.get_option("theme.base") or "light").lower() == "dark"
    if dark:
        return {
            "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
            "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
            "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
            "band": "rgba(57,135,229,0.16)",
            "good": "#0ca30c", "warning": "#fab219",
            "serious": "#ec835a", "critical": "#d03b3b",
        }
    return {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "band": "rgba(42,120,214,0.14)",
        "good": "#0ca30c", "warning": "#fab219",
        "serious": "#ec835a", "critical": "#d03b3b",
    }


T = theme()
CREW_COLOR = {c: T["series"][i] for i, c in enumerate(CREW_ORDER)}

SEVERITY = [(60.0, "critical", "critical"), (25.0, "serious", "serious"),
            (12.0, "warning", "elevated"), (0.0, "good", "watch")]


def severity(score: float) -> tuple[str, str]:
    for cut, role, word in SEVERITY:
        if score >= cut:
            return T[role], word
    return T["good"], "watch"


def frame(fig: go.Figure, *, height: int = 330, ylab: str = "",
          xlab: str = "mission day", legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["ink2"], size=12,
                  family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=xlab, gridcolor=T["grid"], zeroline=False,
                     linecolor=T["axis"], tickfont=dict(color=T["muted"]))
    fig.update_yaxes(title_text=ylab, gridcolor=T["grid"], zeroline=False,
                     linecolor=T["axis"], tickfont=dict(color=T["muted"]))
    return fig


@st.cache_resource(show_spinner="Flying the mission and fitting the twins...")
def session(scenario: str, seed: int, threshold: float):
    return build(scenario, seed=seed, threshold=threshold)


# --- sidebar -----------------------------------------------------------------

st.sidebar.title("AURORA")
st.sidebar.caption("A digital twin of the astronaut, monitored against itself.")

keys = list(SCENARIOS)
scenario = st.sidebar.selectbox(
    "Mission", keys, index=keys.index("overload"),
    format_func=lambda k: SCENARIOS[k].title)
seed = st.sidebar.number_input("Seed", 1, 999, 7, step=1)
threshold = st.sidebar.slider("Detection threshold", 2.0, 30.0,
                              DEFAULT_THRESHOLD, 0.5,
                              help="Studentised units above this crew "
                                   "member's own quiet-window divergence at "
                                   "the same forecast horizon.")

S = session(scenario, int(seed), float(threshold))
day = st.sidebar.slider("Mission day", S.calib_days, S.days - 1, S.days - 1)
crew = st.sidebar.radio("Crew member", list(CREW_ORDER),
                        format_func=lambda c: CREW[c].role)

V = S.views[crew]
A = CREW[crew]
onset = S.mission.onset(crew)

st.sidebar.divider()
st.sidebar.caption(SCENARIOS[scenario].note.split("\n")[0])
if onset is not None:
    st.sidebar.caption(f"Injected physiology begins day {onset}. "
                       "Ground truth, never shown to the twin.")

# --- header ------------------------------------------------------------------

st.title(SCENARIOS[scenario].title)

ds = V.day(day)
openf = [f for f in S.findings_for(crew) if f.start <= day <= f.end]

k = st.columns(4)
k[0].metric("Crew member", A.role, help=A.note)
if ds:
    col, word = severity(ds.score)
    k[1].metric("Divergence today", f"{ds.score:.1f}",
                delta=word, delta_color="off",
                help=f"Strongest at a {ds.horizon}-day forecast horizon. "
                     f"Threshold {S.threshold:.1f}.")
else:
    k[1].metric("Divergence today", "--")
k[2].metric("Findings open", str(len(openf)))
k[3].metric("Twin fitted on",
            f"{len(V.calib.data_driven)}/{len(V.calib.fitted)} own params",
            help="How many parameters this person's own quiet window pinned "
                 "down. The rest are population priors -- see Calibration.")

if openf:
    f = openf[0]
    col, word = severity(f.peak_score)
    bits = [f"**{f.pattern}** -- {word}, attributed to **{f.attribution}**"]
    if f.plan_share > 0.05:
        bits.append(f"the logged flight plan accounts for "
                    f"{100 * f.plan_share:.0f}% of it")
    if f.in_band_at_start:
        bits.append("every channel was still inside its own personal 3-sigma "
                    "band when this was raised")
    if f.attribution == "instrument":
        st.warning("  \n".join(bits), icon=":material/sensors_off:")
    elif f.attribution == "schedule":
        st.info("  \n".join(bits), icon=":material/calendar_month:")
    else:
        st.error("  \n".join(bits), icon=":material/monitor_heart:")
    if f.pattern_note:
        st.caption(f.pattern_note)

tabs = st.tabs(["Twin", "Systems", "Findings", "What if", "Calibration"])

# --- Twin --------------------------------------------------------------------

with tabs[0]:
    st.subheader("Measured against what the plan predicted")
    st.caption(
        "The band is the twin's forecast, anchored days back and driven only "
        "by the logged flight plan -- it has not seen the readings it is "
        "being judged against. Where the dots leave the band, something is "
        "happening that this person's own model, given this schedule, does "
        "not account for.")

    cc = st.columns([3, 1])
    ch = cc[0].selectbox("Channel", list(MONITORED), index=0,
                         format_func=lambda c: SIGNALS[c].name)
    h = cc[1].selectbox("Forecast horizon", list(HORIZONS),
                        index=len(HORIZONS) - 1,
                        format_func=lambda v: f"{v} days")

    j = MONITORED.index(ch)
    sig = SIGNALS[ch]
    days = np.arange(S.days)
    y = V.window.y[:, j]

    # One open-loop forecast per anchor across the mission, so the band is the
    # same object the detector consumes rather than a redrawing of it.
    pred = np.full(S.days, np.nan)
    lo = np.full(S.days, np.nan)
    hi = np.full(S.days, np.nan)
    for d in range(S.calib_days, S.days):
        anchor = d - h
        if anchor < 0:
            continue
        ol = open_loop(V.window, V.calib.params, V.filt.x[anchor],
                       V.filt.P[anchor], anchor, h, sd_scale=A.sd_scale)
        if ol.reached == 0:
            continue
        i = ol.reached - 1
        sc = V.rcal.step_scales(i)[j]
        pred[d] = ol.yhat[i, j]
        lo[d] = ol.yhat[i, j] - 2.0 * ol.S[i, j] * sc
        hi[d] = ol.yhat[i, j] + 2.0 * ol.S[i, j] * sc

    fig = go.Figure()
    m = np.isfinite(lo)
    fig.add_trace(go.Scatter(
        x=np.concatenate([days[m], days[m][::-1]]),
        y=np.concatenate([hi[m], lo[m][::-1]]),
        fill="toself", fillcolor=T["band"], line=dict(width=0),
        hoverinfo="skip", name="twin forecast, 2 sigma"))
    fig.add_trace(go.Scatter(
        x=days, y=pred, mode="lines", name="twin forecast",
        line=dict(color=CREW_COLOR[crew], width=2)))
    fig.add_trace(go.Scatter(
        x=days, y=y, mode="markers", name="Bio-Monitor reading",
        marker=dict(color=T["ink2"], size=5)))

    band = personal_bands(V.window, S.calib_days, A.sd_scale)[ch]
    if np.isfinite(band[0]):
        for v, lab in ((band[0], "personal -3 sigma"), (band[1], "personal +3 sigma")):
            fig.add_hline(y=v, line=dict(color=T["muted"], width=1, dash="dot"),
                          annotation_text=lab,
                          annotation_font=dict(color=T["muted"], size=10))
    if onset is not None:
        fig.add_vline(x=onset, line=dict(color=T["muted"], width=1, dash="dash"),
                      annotation_text="injected onset",
                      annotation_font=dict(color=T["muted"], size=10))
    fig.add_vline(x=day, line=dict(color=T["ink"], width=1))
    st.plotly_chart(frame(fig, height=380, ylab=f"{sig.name} ({sig.unit})"),
                    use_container_width=True)
    st.caption(f"{sig.source} Device: {DEVICES[sig.device].name}.")

    st.subheader("Divergence, whole crew")
    st.caption("One line per crew member. A finding needs the line to stay "
               "above the threshold for several days -- the nominal tail is "
               "made of isolated spikes, and demanding persistence separates "
               "them without delaying anything real.")
    f2 = go.Figure()
    for c in CREW_ORDER:
        f2.add_trace(go.Scatter(
            x=days, y=S.views[c].score_series(S.days), mode="lines",
            name=CREW[c].role, line=dict(color=CREW_COLOR[c], width=2)))
    f2.add_hline(y=S.threshold, line=dict(color=T["critical"], width=1,
                                          dash="dash"),
                 annotation_text="threshold",
                 annotation_font=dict(color=T["muted"], size=10))
    f2.add_vline(x=day, line=dict(color=T["ink"], width=1))
    st.plotly_chart(frame(f2, ylab="studentised divergence"),
                    use_container_width=True)

# --- Systems -----------------------------------------------------------------

with tabs[1]:
    st.subheader("The seven systems behind the twelve channels")
    st.caption(
        "None of these is measured. Each is inferred from the channels that "
        "respond to it, which is what lets a coherent movement across four "
        "readings be read as one change rather than four coincidences.")

    pick = st.multiselect("Systems", list(STATE), default=["A", "S", "V", "I"],
                          format_func=lambda k: STATE_NAMES[k])
    f3 = go.Figure()
    for i, k_ in enumerate(pick):
        jj = STATE.index(k_)
        sd = np.sqrt(np.clip(V.filt.P[:, jj, jj], 0, None))
        est = V.filt.x[:, jj]
        colr = T["series"][i % len(T["series"])]
        f3.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([est + 2 * sd, (est - 2 * sd)[::-1]]),
            fill="toself", fillcolor=T["band"], line=dict(width=0),
            hoverinfo="skip", showlegend=False))
        f3.add_trace(go.Scatter(
            x=days, y=est, mode="lines", line=dict(color=colr, width=2),
            name=f"{STATE_NAMES[k_]}"
                 + (f" ({STATE_UNITS[k_]})" if STATE_UNITS[k_] else "")))
    f3.add_vline(x=day, line=dict(color=T["ink"], width=1))
    st.plotly_chart(frame(f3, height=360, ylab="latent estimate"),
                    use_container_width=True)
    st.caption("Shaded: 2 sigma from the filter. Sleep debt reaches the "
               "measurements only through its effect on autonomic load, so it "
               "is the least directly observed of the seven and its band is "
               "correspondingly wide.")

    st.subheader(f"What is costing {A.role} on day {day}")
    st.caption("The load index broken into its named contributions, in the "
               "same units as the total. This is the whole explanation of an "
               "autonomic finding in a form that supports a decision.")
    terms = [(k_, v) for k_, v in
             explain_load(V.filt.x[day], V.calib.params, V.window.u[day])
             if v > 0.002]
    if terms:
        f4 = go.Figure(go.Bar(
            x=[v for _, v in terms][::-1], y=[k_ for k_, _ in terms][::-1],
            orientation="h", marker=dict(color=T["series"][0],
                                         cornerradius=4),
            text=[f"{v:.2f}" for _, v in terms][::-1],
            textposition="outside",
            textfont=dict(color=T["ink2"]), hoverinfo="skip"))
        st.plotly_chart(frame(f4, height=max(200, 46 * len(terms)),
                              ylab="", xlab="contribution to autonomic load",
                              legend=False),
                        use_container_width=True)

    st.subheader("What the plan asked of them")
    pick_u = st.selectbox("Input", list(INPUTS), index=0,
                          format_func=lambda k: INPUT_NAMES[k])
    f5 = go.Figure(go.Scatter(
        x=days, y=V.window.u[:, UIDX[pick_u]], mode="lines",
        line=dict(color=T["series"][1], width=2, shape="hv"),
        name=INPUT_NAMES[pick_u]))
    f5.add_vline(x=day, line=dict(color=T["ink"], width=1))
    st.plotly_chart(frame(f5, height=260, legend=False,
                          ylab=f"{INPUT_NAMES[pick_u]} ({INPUT_UNITS[pick_u]})"),
                    use_container_width=True)
    st.caption("Logged by flight ops and handed to the twin verbatim. A "
               "monitor that has to infer the flight plan from the physiology "
               "reads every busy week as a medical finding.")

# --- Findings ----------------------------------------------------------------

with tabs[2]:
    st.subheader("What was raised")
    if not S.findings:
        st.success("No findings on this mission.", icon=":material/check:")
    for f in S.findings:
        col, word = severity(f.peak_score)
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**{CREW[f.crew].role} -- {f.pattern}**")
            c1.caption(f"days {f.start}-{f.end} ({f.length} d), peak "
                       f"{f.peak_score:.1f} on day {f.peak_day} | "
                       f"{word} | attributed to **{f.attribution}**")
            if f.pattern_note:
                c1.caption(f.pattern_note)
            notes = []
            if f.in_band_at_start:
                notes.append("every channel still inside its own personal "
                             "3-sigma band at detection")
            if f.plan_share > 0.05:
                notes.append(f"flight plan accounts for "
                             f"{100 * f.plan_share:.0f}%")
            if f.limit_breach:
                notes.append("protocol limit breached: "
                             + ", ".join(f.limit_breach))
            if f.device_q:
                rk = sorted(f.device_q.items(), key=lambda kv: -kv[1])
                notes.append("device loudness (1.0 = behaving): " + ", ".join(
                    f"{DEVICES[d].name.split(' (')[0]} {q:.1f}"
                    for d, q in rk))
            for n in notes:
                c1.markdown(f"- {n}")

            top = sorted(f.latent_z.items(), key=lambda kv: -abs(kv[1]))[:5]
            fg = go.Figure(go.Bar(
                x=[v for _, v in top][::-1],
                y=[STATE_NAMES[k_] for k_, _ in top][::-1],
                orientation="h",
                marker=dict(color=[T["series"][0] if v >= 0 else T["series"][1]
                                   for _, v in top][::-1], cornerradius=4),
                hoverinfo="skip"))
            c2.plotly_chart(frame(fg, height=190, legend=False,
                                  xlab="standardised latent shift", ylab=""),
                            use_container_width=True,
                            key=f"lat-{f.crew}-{f.start}")

    st.divider()
    st.subheader("Where the readings sit against this person's own range")
    if ds:
        rows = sorted(ds.channel_z.items(), key=lambda kv: -abs(kv[1]))
        f6 = go.Figure(go.Bar(
            x=[v for _, v in rows][::-1],
            y=[SIGNALS[k_].name for k_, _ in rows][::-1], orientation="h",
            marker=dict(color=[T["series"][0] if v >= 0 else T["series"][1]
                               for _, v in rows][::-1], cornerradius=4),
            hoverinfo="skip"))
        for v in (-2, 2):
            f6.add_vline(x=v, line=dict(color=T["muted"], width=1, dash="dot"))
        st.plotly_chart(frame(f6, height=max(240, 30 * len(rows)), legend=False,
                              xlab="residual, standardised", ylab=""),
                        use_container_width=True)
        st.caption("Residual against the twin's forecast, not against a "
                   "population range. Dotted lines at 2 sigma.")

# --- What if -----------------------------------------------------------------

with tabs[3]:
    st.subheader("Change the schedule, re-run the twin")
    st.caption(
        "Monte Carlo over the current state, the daily process noise, and the "
        "parameters themselves -- the third is the one usually left out and it "
        "dominates at longer horizons, because being wrong about how fast "
        "somebody recovers compounds while today's measurement error does "
        "not. Both cones share a random seed, so the difference between them "
        "is the countermeasure rather than Monte Carlo noise.")

    c = st.columns(3)
    horizon = c[0].slider("Forecast days", 5, 28, 14)
    runs = c[1].select_slider("Runs", [100, 200, 300, 500], value=200)
    ch2 = c[2].selectbox("Channel", list(MONITORED), index=0,
                         format_func=lambda k: SIGNALS[k].name, key="cfch")

    st.markdown("**The intervention**")
    cols = st.columns(3)
    change: dict[str, float] = {}
    for i, k_ in enumerate(("sleep_opp", "exercise", "workload",
                            "fluid", "light")):
        base = float(NOMINAL_INPUT[UIDX[k_]])
        rlo, rhi = INPUT_RANGE[k_]
        step = 0.05 if rhi - rlo <= 2 else (0.1 if rhi - rlo <= 5 else 5.0)
        val = cols[i % 3].slider(
            f"{INPUT_NAMES[k_]} ({INPUT_UNITS[k_]})",
            float(rlo), float(rhi), base, float(step), key=f"cf-{k_}")
        if abs(val - base) > 1e-9:
            change[k_] = val - base

    if not change:
        st.info("Move a slider to compare a modified schedule against the "
                "nominal one.", icon=":material/tune:")
    else:
        cf = counterfactual(
            V.calib.params, V.filt.x[day], V.filt.P[day],
            horizon=horizon, change=change, label="intervention",
            start_day=day, n=int(runs), sd_scale=A.sd_scale,
            param_sd={k_: V.calib.se[k_] for k_ in V.calib.fitted})

        f7 = go.Figure()
        for ens, name, colr in ((cf.baseline, "nominal schedule", T["series"][0]),
                                (cf.modified, "modified schedule", T["series"][1])):
            q = ens.quantiles(ch2, (0.1, 0.5, 0.9))
            xs = ens.days
            f7.add_trace(go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([q[2], q[0][::-1]]),
                fill="toself", line=dict(width=0), hoverinfo="skip",
                fillcolor=(T["band"] if colr == T["series"][0]
                           else "rgba(235,104,52,0.16)"),
                showlegend=False))
            f7.add_trace(go.Scatter(x=xs, y=q[1], mode="lines", name=name,
                                    line=dict(color=colr, width=2)))
        st.plotly_chart(frame(f7, height=360,
                              ylab=f"{SIGNALS[ch2].name} ({SIGNALS[ch2].unit})"),
                        use_container_width=True)
        st.caption("Median with a 10th-to-90th-percentile cone.")

        base_now = float(np.median(cf.baseline.y[:, 0, MONITORED.index(ch2)]))
        target = float(np.nanmean(V.window.y[:S.calib_days,
                                             MONITORED.index(ch2)]))
        rising = SIGNALS[ch2].adverse == "low"
        rec = cf.recovery_day(ch2, target, prob=0.8, rising=rising)
        d_end = float(cf.delta(ch2)[-1])

        m = st.columns(3)
        m[0].metric(f"{SIGNALS[ch2].name} at day {day + horizon}",
                    f"{d_end:+.1f} {SIGNALS[ch2].unit}",
                    help="Median difference the intervention makes.")
        m[1].metric("Back to quiet-window average",
                    f"day {rec}" if rec else "not within horizon",
                    help="First day the modified schedule clears this "
                         "person's own quiet-window average with at least "
                         "80% probability.")
        m[2].metric("Quiet-window average",
                    f"{target:.1f} {SIGNALS[ch2].unit}")

# --- Calibration -------------------------------------------------------------

with tabs[4]:
    st.subheader("How much of this twin is really this person")
    st.caption(
        "A twin that does not report this is claiming to know the astronaut "
        "better than the data permits, and every forecast inherits the "
        "overconfidence. Where somebody's HRV sits is measured precisely by a "
        "quiet month. How steeply they respond to a heavy workload is not -- "
        "because nothing happened. You cannot learn a stress response from a "
        "period without stress.")

    cal = V.calib
    st.markdown(f"**{cal.summary()}**")

    rows = sorted(cal.fitted, key=lambda k_: cal.prior_share[k_])
    f8 = go.Figure(go.Bar(
        x=[100 * (1 - cal.prior_share[k_]) for k_ in rows][::-1],
        y=rows[::-1], orientation="h",
        marker=dict(color=T["series"][2], cornerradius=4),
        text=[f"{100 * (1 - cal.prior_share[k_]):.0f}%" for k_ in rows][::-1],
        textposition="outside", textfont=dict(color=T["ink2"]),
        hoverinfo="skip"))
    f8.add_vline(x=50, line=dict(color=T["muted"], width=1, dash="dot"))
    st.plotly_chart(frame(f8, height=max(300, 26 * len(rows)), legend=False,
                          xlab="share of this parameter determined by this "
                               "person's own data (%)", ylab=""),
                    use_container_width=True)

    st.markdown("**Held at population values, and stated as such**")
    for k_ in cal.prior_driven:
        st.markdown(f"- `{k_}` = {getattr(cal.params, k_):.4g} "
                    f"-- {100 * cal.prior_share[k_]:.0f}% population prior")

    st.divider()
    st.subheader("Where the numbers come from")
    for key, dev in DEVICES.items():
        chans = [SIGNALS[c].name for c in MONITORED
                 if SIGNALS[c].device == key]
        with st.container(border=True):
            st.markdown(f"**{dev.name}**")
            st.caption(dev.note)
            if chans:
                st.caption("Channels: " + ", ".join(chans))

    st.caption(
        "Device capabilities follow the Canadian Space Agency's Bio-Monitor, "
        "the Astroskin-derived garment built by Carre Technologies that was "
        "commissioned aboard the ISS in January 2019. Physiologies here are "
        "synthetic, with levels and variability in published spaceflight "
        "ranges; they are not anybody's records, and nothing is attributed to "
        "a named astronaut. AURORA is decision support, not a diagnostic "
        "device.")
