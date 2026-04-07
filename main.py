"""
Entry point for the dispute resolution system.

Usage:
    python main.py                  # run all three test cases
    python main.py --case DISP-001  # run a specific case
"""

import argparse
import sys

from agent import resolve_dispute
from schemas import Verdict
from test_cases import ALL_CASES


def print_verdict(verdict: Verdict, expected: str | None = None) -> None:
    split = verdict.liability_split
    print(f"\n{'─'*60}")
    print(f"  VERDICT — {verdict.case_id}")
    print(f"{'─'*60}")
    print(f"  Primary liable party : {verdict.primary_liable_party.upper()}")
    print(f"  Confidence           : {verdict.confidence:.0%}")
    print()
    print("  Liability split:")
    print(f"    Consumer        : {split.consumer_pct:>5.1f}%")
    print(f"    Agent platform  : {split.agent_platform_pct:>5.1f}%")
    print(f"    Merchant        : {split.merchant_pct:>5.1f}%")
    print()
    print(f"  Explanation:\n    {verdict.explanation}")
    print()
    print(f"  Recommended resolution:\n    {verdict.recommended_resolution}")

    if expected:
        print()
        print(f"  Expected (reference):\n    {expected}")
    print(f"{'─'*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic dispute resolution system")
    parser.add_argument(
        "--case",
        metavar="CASE_ID",
        help="Run a single case by ID (e.g. DISP-001). Omit to run all cases.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress tool-call trace output",
    )
    args = parser.parse_args()

    cases = ALL_CASES
    if args.case:
        cases = [c for c in ALL_CASES if c.case_id == args.case]
        if not cases:
            print(f"Error: case '{args.case}' not found.", file=sys.stderr)
            print(f"Available cases: {[c.case_id for c in ALL_CASES]}", file=sys.stderr)
            sys.exit(1)

    verdicts: list[tuple[Verdict, str | None]] = []

    for case in cases:
        verdict = resolve_dispute(case, verbose=not args.quiet)
        verdicts.append((verdict, case.expected_verdict))

    print("\n\n" + "="*60)
    print("  SUMMARY OF VERDICTS")
    print("="*60)
    for verdict, expected in verdicts:
        print_verdict(verdict, expected)


if __name__ == "__main__":
    main()
