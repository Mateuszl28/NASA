"""Does it work, and how often is it wrong.

Two measurements, and the second is the one that decides whether the first is
worth anything.

**Detection.** For each scenario with known injected physiology: was the right
crew member flagged, how many days after onset, with what attribution, and --
the claim the whole design rests on -- was every individual channel still
inside its own personal three-sigma band at the moment of detection. That last
column is what separates "earlier than a threshold monitor" from "louder than a
threshold monitor", and the band it is measured against is built in
`divergence.personal_bands` the way a competent threshold monitor would build
it, not a strawman.

**Specificity.** The same machinery on nominal missions with nothing wrong,
pooled over independent seeds. Three deliberate choices:

*Pooled over seeds*, because one quiet run is luck and a rate is evidence.

*Counted in episodes, not days.* A stable advisory that persists for five days
is one thing for the crew to read and dismiss, not five. Counting days makes a
single false finding look like a week-long failure, and makes a real finding
look like a flood.

*Reported against a declared budget* rather than as a bare number, because
"0.4 false findings per crew member per 100 days" only means something next to
what was considered acceptable before the measurement was taken.

`sweep_threshold` exists because the detection threshold should be chosen
against that budget and then held, not nudged until the scenarios pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .signals import MONITORED

from .crew import CREW, CREW_ORDER
from .divergence import Finding, personal_bands
from .session import DEFAULT_THRESHOLD, Session, build
from .truth import SCENARIOS


@dataclass
class ScenarioResult:
    key: str
    title: str
    expect_crew: tuple[str, ...]
    expect_attribution: str
    expect_unexplained: bool
    onset: int | None
    detected_day: int | None
    detected_crew: str | None
    latency: int | None
    attribution: str | None
    pattern: str
    in_band: bool
    #: First day at or after onset that any single channel leaves this crew
    #: member's own +/-3 sigma band -- i.e. when a personal-baseline threshold
    #: monitor would have caught *this event*.
    band_exit: int | None
    #: Band exits before the event existed. The cost of running that monitor,
    #: and the reason its detection day cannot be quoted without it.
    band_false_exits: int
    off_target: dict[str, int]
    findings: list[Finding] = field(default_factory=list)

    @property
    def lead(self) -> int | None:
        """Days AURORA is ahead of a personal 3-sigma threshold monitor.

        Measured, not asserted -- this is the claim most worth being wrong
        about, and both attempts at it were wrong in instructive ways.

        An early README said "nine days" from a trajectory read by eye. The
        first measurement said two. The second measurement said *minus
        fifteen*, which looked like a refutation until the dates were checked:
        the band exits it was finding landed on days 54 to 58, before the
        events began. It was timing the threshold monitor's first *false*
        alarm, not its detection -- which is its own result, since twelve
        channels under a 3-sigma band estimated from 45 days will leave that
        band on noise several times a mission.

        So the comparison is now made properly: the threshold monitor's
        detection is its first band exit at or after onset, and its pre-onset
        exits are counted and reported beside it. Quoting either number alone
        flatters somebody.
        """
        if self.detected_day is None or self.band_exit is None:
            return None
        return self.band_exit - self.detected_day

    @property
    def ok(self) -> bool:
        if not self.expect_unexplained:
            # Nothing should be raised on the crew member the scenario is about.
            return self.detected_crew not in self.expect_crew
        if self.detected_crew is None:
            return False
        if self.detected_crew not in self.expect_crew:
            return False
        return self.attribution == self.expect_attribution


def run_scenario(key: str, *, seed: int = 7,
                 threshold: float = DEFAULT_THRESHOLD) -> ScenarioResult:
    s = build(key, seed=seed, threshold=threshold)
    sc = SCENARIOS[key]

    onset = None
    for c in sc.expect_crew or CREW_ORDER:
        o = s.mission.onset(c)
        if o is not None:
            onset = o if onset is None else min(onset, o)

    on_target = [f for f in s.findings
                 if f.crew in sc.expect_crew
                 and (onset is None or f.start >= onset)]
    on_target.sort(key=lambda f: f.start)

    off: dict[str, int] = {}
    for f in s.findings:
        if f in on_target:
            continue
        tag = f"{f.crew}/{f.pattern}"
        if onset is not None and f.start < onset:
            tag += " (pre-onset)"
        off[tag] = off.get(tag, 0) + 1

    first = on_target[0] if on_target else None

    # When would a personal-baseline threshold monitor first have seen
    # anything? Measured on the same crew member, against bands built the way
    # a competent one would build them.
    band_exit, false_exits = None, 0
    if first is not None:
        v = s.views[first.crew]
        bands = personal_bands(v.window, s.calib_days, CREW[first.crew].sd_scale)
        for d in range(s.calib_days, s.days):
            out = any(
                np.isfinite(v.window.y[d, MONITORED.index(ch)])
                and not (bands[ch][0] <= v.window.y[d, MONITORED.index(ch)]
                         <= bands[ch][1])
                for ch in MONITORED)
            if not out:
                continue
            if onset is not None and d < onset:
                false_exits += 1
            elif band_exit is None:
                band_exit = d

    return ScenarioResult(
        key=key, title=sc.title, expect_crew=sc.expect_crew,
        expect_attribution=sc.expect_attribution,
        expect_unexplained=sc.expect_unexplained,
        onset=onset,
        detected_day=first.start if first else None,
        detected_crew=first.crew if first else None,
        latency=(first.start - onset) if (first and onset is not None) else None,
        attribution=first.attribution if first else None,
        pattern=first.pattern if first else "",
        in_band=first.in_band_at_start if first else False,
        band_exit=band_exit, band_false_exits=false_exits,
        off_target=off, findings=s.findings,
    )


@dataclass
class Specificity:
    seeds: tuple[int, ...]
    episodes: int
    per_seed: list[int]
    crew_days: int
    rate_per_100: float
    budget: float
    advisory_days: int

    @property
    def passes(self) -> bool:
        return self.rate_per_100 <= self.budget

    def lines(self) -> list[str]:
        return [
            f"false findings  : {self.episodes} episode(s) over "
            f"{self.crew_days} crew-days ({self.advisory_days} advisory-days)",
            f"episodes/seed   : {self.per_seed}",
            f"in plain terms  : {self.rate_per_100:.2f} false findings per crew "
            f"member per 100 days -- about "
            f"{self.rate_per_100 * 10:.0f} over a 1000-day Mars mission",
            f"budget          : {'PASS' if self.passes else 'FAIL'} "
            f"({self.rate_per_100:.2f} vs {self.budget:.2f} per 100 crew-days "
            f"allowed)",
        ]


def specificity(*, seeds: tuple[int, ...] = (7, 11, 23, 41, 57, 73),
                threshold: float = DEFAULT_THRESHOLD,
                budget: float = 2.0) -> Specificity:
    """Pooled false-finding rate on missions with nothing wrong."""
    episodes, per_seed, crew_days, adv_days = 0, [], 0, 0
    for sd in seeds:
        s = build("nominal", seed=sd, threshold=threshold)
        n = len(s.findings)
        episodes += n
        per_seed.append(n)
        adv_days += sum(f.length for f in s.findings)
        scored = sum(len(v.scans) for v in s.views.values())
        crew_days += scored
    rate = 100.0 * episodes / max(crew_days, 1)
    return Specificity(seeds=seeds, episodes=episodes, per_seed=per_seed,
                       crew_days=crew_days, rate_per_100=rate, budget=budget,
                       advisory_days=adv_days)


def sweep_threshold(*, seeds: tuple[int, ...] = (7, 11, 23),
                    candidates: tuple[float, ...] = (
                        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0),
                    budget: float = 2.0) -> list[tuple[float, float, int]]:
    """False-finding rate per candidate threshold, plus scenarios detected.

    Printed as a curve rather than reduced to a single recommended number,
    because the trade is a judgement about how much an interruption costs on a
    mission where nobody can be replaced, and that is not the detector's call.
    """
    out = []
    detect_keys = [k for k, v in SCENARIOS.items() if v.expect_unexplained]
    for t in candidates:
        sp = specificity(seeds=seeds, threshold=t, budget=budget)
        hits = sum(1 for k in detect_keys
                   if run_scenario(k, threshold=t).ok)
        out.append((t, sp.rate_per_100, hits))
    return out


# --- reporting ----------------------------------------------------------------

def _fmt(r: ScenarioResult) -> list[str]:
    head = f"-- {r.key:<15} {r.title}"
    lines = [head]
    if not r.expect_unexplained:
        verdict = "PASS" if r.ok else "FAIL"
        got = (f"raised {r.pattern} on {r.detected_crew} day {r.detected_day}"
               if r.detected_crew else "raised nothing")
        lines.append(f"   expected        : no unexplained finding -- {got}")
        lines.append(f"   result          : {verdict}")
    elif r.detected_day is None:
        lines.append("   detection       : MISSED")
    else:
        lines.append(
            f"   detection       : day {r.detected_day} "
            f"(+{r.latency} from onset {r.onset}) on {r.detected_crew}, "
            f"score {max((f.peak_score for f in r.findings), default=0):.1f}")
        lines.append(f"   pattern         : {r.pattern}")
        lines.append(f"   attribution     : {r.attribution} "
                     f"({'correct' if r.attribution == r.expect_attribution else 'EXPECTED ' + r.expect_attribution})")
        if r.in_band:
            lines.append("   in band         : yes -- every channel still "
                         "inside its own personal 3-sigma band at detection")
        if r.lead is not None:
            lines.append(f"   lead            : {r.lead:+d} days vs a personal "
                         f"3-sigma threshold monitor, which catches this on "
                         f"day {r.band_exit}")
        elif r.detected_day is not None:
            lines.append("   lead            : a personal 3-sigma threshold "
                         "monitor never catches this event at all")
        if r.detected_day is not None:
            lines.append(f"   that monitor's   : {r.band_false_exits} band "
                         f"exit(s) on this crew member before the event "
                         f"existed")
    lines.append(f"   off-target      : {r.off_target or 'none'}")
    return lines


def report(*, seeds: tuple[int, ...] = (7, 11, 23, 41, 57, 73),
           threshold: float = DEFAULT_THRESHOLD) -> str:
    out: list[str] = []
    results = []
    for key in SCENARIOS:
        if key == "nominal":
            continue
        r = run_scenario(key, threshold=threshold)
        results.append(r)
        out.extend(_fmt(r))
        out.append("")

    sp = specificity(seeds=seeds, threshold=threshold)
    out.append(f"-- specificity, pooled over {len(seeds)} nominal missions")
    out.extend("   " + l for l in sp.lines())
    out.append("")

    passed = sum(1 for r in results if r.ok)
    ok = passed == len(results) and sp.passes
    out.append(f"RESULT: {'PASS' if ok else 'FAIL'} "
               f"({passed}/{len(results)} scenarios, specificity "
               f"{'within' if sp.passes else 'over'} budget)")
    return "\n".join(out)
