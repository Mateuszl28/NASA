"""Command line: run a mission, validate the detectors, print a report."""

from __future__ import annotations

import argparse
import sys

from .session import DEFAULT_THRESHOLD, build
from .truth import SCENARIOS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aurora",
        description="AURORA -- a digital twin of the astronaut.")
    ap.add_argument("--scenario", default="overload",
                    choices=sorted(SCENARIOS),
                    help="which mission to fly (default: overload)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--day", type=int, default=None,
                    help="report as of this day (default: last)")
    ap.add_argument("--report", action="store_true",
                    help="print the full downlink report")
    ap.add_argument("--validate", action="store_true",
                    help="run the detector validation sweep")
    ap.add_argument("--sweep", action="store_true",
                    help="false-alarm rate against detection, per threshold")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refit the twins, ignoring the cache")
    ap.add_argument("--scenarios", action="store_true",
                    help="list the scenarios and what each one is for")
    args = ap.parse_args(argv)

    if args.scenarios:
        for key, sc in SCENARIOS.items():
            print(f"{key}\n    {sc.title}")
            for line in sc.note.split("\n"):
                print(f"    {line}" if line else "")
            print()
        return 0

    if args.validate:
        from .validate import report as vreport
        print(vreport(threshold=args.threshold))
        return 0

    if args.sweep:
        from .validate import sweep_threshold
        print(f"{'threshold':>10}  {'false/100 crew-days':>20}  "
              f"{'scenarios detected':>19}")
        for t, rate, hits in sweep_threshold():
            print(f"{t:>10.1f}  {rate:>20.2f}  {hits:>19d}")
        return 0

    s = build(args.scenario, seed=args.seed, threshold=args.threshold,
              refresh=args.recalibrate)

    if args.report:
        from .report import render
        print(render(s, day=args.day))
        return 0

    print(f"{s.mission.scenario.title}  (seed {s.mission.seed}, "
          f"{s.days} days, threshold {s.threshold:.1f})")
    print()
    for c, v in s.views.items():
        print(f"  {v.role:<20} {v.calib.summary()}")
    print()
    if not s.findings:
        print("  no findings")
    for f in s.findings:
        flag = " [all channels still in personal band]" if f.in_band_at_start else ""
        plan = (f"  plan explains {100 * f.plan_share:.0f}%"
                if f.plan_share > 0.05 else "")
        print(f"  day {f.start:>3}-{f.end:<3} {f.crew:<4} "
              f"{f.attribution:<12} {f.pattern:<36} peak {f.peak_score:>6.1f}"
              f"{plan}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
