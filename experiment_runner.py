"""
Experiment runner — runs all 9 cases in ALL_CASES_BY_POLICY through the
three-agent pipeline and prints a Markdown summary table.

Usage:
    python3 experiment_runner.py

The table is printed to stdout and is ready to paste into any Markdown-aware
report (GitHub, Notion, Obsidian, etc.).
"""

import time
import traceback

import coordinator
from test_cases import ALL_CASES_BY_POLICY

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

HEADERS = [
    "Case ID",
    "Policy",
    "Primary Liable",
    "Consumer %",
    "Platform %",
    "Merchant %",
    "Confidence",
    "Status",
]

# Minimum column widths (expands if content is wider)
MIN_WIDTHS = [17, 8, 14, 10, 10, 10, 10, 7]


def _col_widths(rows: list[list[str]]) -> list[int]:
    widths = list(MIN_WIDTHS)
    for row in [HEADERS] + rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def _md_row(cells: list[str], widths: list[int]) -> str:
    return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"


def _md_separator(widths: list[int]) -> str:
    return "| " + " | ".join("-" * w for w in widths) + " |"


def _print_table(rows: list[list[str]]) -> None:
    widths = _col_widths(rows)
    print(_md_row(HEADERS, widths))
    print(_md_separator(widths))
    prev_base_id = None
    for row in rows:
        # Blank separator line between case groups (different base case IDs)
        base_id = row[0].rsplit("-", 1)[0]  # e.g. "DISP-001-BALANCED" → "DISP-001"
        if prev_base_id is not None and base_id != prev_base_id:
            print(_md_row([""] * len(HEADERS), widths))
        prev_base_id = base_id
        print(_md_row(row, widths))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiments(verbose_agents: bool = False) -> None:
    total = len(ALL_CASES_BY_POLICY)
    print(f"Running {total} cases ({total // 3} base cases × 3 policy presets)…\n")

    rows: list[list[str]] = []
    passed = 0
    failed = 0

    for idx, (case, preset) in enumerate(ALL_CASES_BY_POLICY, start=1):
        print(f"  [{idx:2d}/{total}] {case.case_id} … ", end="", flush=True)
        t0 = time.time()

        try:
            verdict = coordinator.resolve_dispute(case, verbose=verbose_agents)
            elapsed = time.time() - t0
            split = verdict.liability_split
            rows.append([
                case.case_id,
                preset,
                verdict.primary_liable_party,
                f"{split.consumer_pct:.1f}",
                f"{split.agent_platform_pct:.1f}",
                f"{split.merchant_pct:.1f}",
                f"{verdict.confidence:.2f}",
                f"ok ({elapsed:.0f}s)",
            ])
            print(f"done in {elapsed:.0f}s")
            passed += 1

        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            rows.append([
                case.case_id,
                preset,
                "ERROR",
                "—",
                "—",
                "—",
                "—",
                "FAILED",
            ])
            print(f"FAILED ({elapsed:.0f}s)")
            print(f"       {type(exc).__name__}: {exc}")
            if verbose_agents:
                traceback.print_exc()
            failed += 1

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("EXPERIMENT RESULTS")
    print("=" * 72)
    print()
    _print_table(rows)
    print()
    print(f"Completed: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed", end="")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all dispute cases and print results table.")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-agent turn output while running",
    )
    args = parser.parse_args()

    run_experiments(verbose_agents=args.verbose)
