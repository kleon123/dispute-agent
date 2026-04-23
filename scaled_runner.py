"""
scaled_runner.py — Three-part scale test for the dispute resolution pipeline.

  Part 1  Parallel:      10 cases run simultaneously via ThreadPoolExecutor.
                         Reports per-case latency and any failures.
  Part 2  Consistency:   DISP-001 run 10× sequentially; variance in platform %.
  Part 3  Cost profile:  Token + cost breakdown per agent for DISP-001/002/003.

Pricing basis (claude-sonnet-4-5):
  Input  $3.00 / million tokens
  Output $15.00 / million tokens

Usage:
    python3 scaled_runner.py
    python3 scaled_runner.py --part parallel
    python3 scaled_runner.py --part consistency
    python3 scaled_runner.py --part cost
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Token instrumentation — must happen before any agent module is imported
# so the patched version is the only one ever bound.
# ---------------------------------------------------------------------------

from anthropic.resources.messages.messages import Messages as _Messages

INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000   # $3.00  per million input  tokens
OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000   # $15.00 per million output tokens


@dataclass
class TokenRecord:
    agent: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens  * INPUT_COST_PER_TOKEN +
                self.output_tokens * OUTPUT_COST_PER_TOKEN)


# Per-execution-context token accumulator and current-agent label
_token_log:     ContextVar[list[TokenRecord] | None] = ContextVar("_token_log",     default=None)
_current_agent: ContextVar[str]                      = ContextVar("_current_agent", default="unknown")

_orig_create = _Messages.create


def _instrumented_create(self, *args, **kwargs):
    response = _orig_create(self, *args, **kwargs)
    log = _token_log.get()
    if log is not None and hasattr(response, "usage"):
        log.append(TokenRecord(
            agent=_current_agent.get(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ))
    return response


_Messages.create = _instrumented_create  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Agent imports — after patching so they pick up the instrumented create
# ---------------------------------------------------------------------------

import arbitrator_agent
import coordinator
import intent_agent
import policy_agent
from policies import BALANCED_POLICY
from schemas import DisputeCase, Verdict
from test_cases import CASE_1, CASE_2, CASE_3

# Wrap each agent's run() to stamp _current_agent in the active context.
# coordinator.py calls intent_agent.run / policy_agent.run / arbitrator_agent.run
# via module-level attribute lookup, so replacing the attribute here is enough.

_orig_intent     = intent_agent.run
_orig_policy     = policy_agent.run
_orig_arbitrator = arbitrator_agent.run


def _traced_intent(case, verbose=False):
    _current_agent.set("intent")
    return _orig_intent(case, verbose=verbose)


def _traced_policy(case, intent_analysis, verbose=False):
    _current_agent.set("policy")
    return _orig_policy(case, intent_analysis, verbose=verbose)


def _traced_arbitrator(case, intent_analysis, policy_report, verbose=False):
    _current_agent.set("arbitrator")
    return _orig_arbitrator(case, intent_analysis, policy_report, verbose=verbose)


intent_agent.run     = _traced_intent      # type: ignore[assignment]
policy_agent.run     = _traced_policy      # type: ignore[assignment]
arbitrator_agent.run = _traced_arbitrator  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    case_id:   str
    preset:    str
    verdict:   Verdict | None     = None
    tokens:    list[TokenRecord]  = field(default_factory=list)
    elapsed_s: float              = 0.0
    error:     str                = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not None

    def total_input_tokens(self)  -> int:   return sum(t.input_tokens  for t in self.tokens)
    def total_output_tokens(self) -> int:   return sum(t.output_tokens for t in self.tokens)
    def total_cost_usd(self)      -> float: return sum(t.cost_usd      for t in self.tokens)


def run_case(case: DisputeCase, preset: str = "balanced") -> RunResult:
    """Run one case in an isolated context so token logs don't bleed across threads."""
    result = RunResult(case_id=case.case_id, preset=preset)
    tokens: list[TokenRecord] = []

    def _inner():
        _token_log.set(tokens)
        t0 = time.time()
        try:
            result.verdict = coordinator.resolve_dispute(case, verbose=False)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = time.time() - t0
        result.tokens = list(tokens)

    copy_context().run(_inner)
    return result


# ---------------------------------------------------------------------------
# Markdown table helpers (column widths expand to fit content)
# ---------------------------------------------------------------------------

def _widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(w):
                w[i] = max(w[i], len(cell))
    return w


def _row(cells: list[str], widths: list[int], right_cols: set[int] | None = None) -> str:
    right_cols = right_cols or set()
    parts = []
    for i, (c, w) in enumerate(zip(cells, widths)):
        parts.append(c.rjust(w) if i in right_cols else c.ljust(w))
    return "| " + " | ".join(parts) + " |"


def _sep(widths: list[int]) -> str:
    return "| " + " | ".join("-" * w for w in widths) + " |"


def _print_md_table(headers: list[str], rows: list[list[str]],
                    right_cols: set[int] | None = None,
                    group_col: int | None = None) -> None:
    w = _widths(headers, rows)
    print(_row(headers, w, right_cols))
    print(_sep(w))
    prev_group = None
    for row in rows:
        if group_col is not None:
            g = row[group_col]
            if prev_group is not None and g != prev_group:
                print(_row([""] * len(headers), w))
            prev_group = g
        print(_row(row, w, right_cols))


# ===========================================================================
# Part 1 — Parallel (10 cases, ThreadPoolExecutor)
# ===========================================================================

def run_parallel() -> list[RunResult]:
    base_cases = [CASE_1, CASE_2, CASE_3]
    # Build 10 cases by cycling through the three base cases; suffix IDs for clarity
    cases: list[DisputeCase] = []
    for i in range(10):
        base = base_cases[i % 3]
        from dataclasses import replace
        cases.append(replace(base, case_id=f"{base.case_id}-P{i+1:02d}"))

    print(f"\n{'='*72}")
    print(f"  PART 1 — PARALLEL  ({len(cases)} cases, max_workers=10)")
    print(f"{'='*72}\n")
    print(f"  Submitting {len(cases)} cases …\n")

    results: list[RunResult] = [RunResult(case_id="", preset="balanced")] * len(cases)
    wall_t0 = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(run_case, case, "balanced"): idx
            for idx, case in enumerate(cases)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = RunResult(
                    case_id=cases[idx].case_id,
                    preset="balanced",
                    error=str(exc),
                )
            completed += 1
            r = results[idx]
            status = f"ok ({r.elapsed_s:.0f}s)" if r.ok else f"FAILED"
            print(f"  [{completed:2d}/10] {r.case_id:<22} {status}")

    wall_elapsed = time.time() - wall_t0

    headers = ["#", "Case ID", "Primary Liable", "Consumer%", "Platform%", "Merchant%",
               "Confidence", "Latency(s)", "Status"]
    rows = []
    for i, r in enumerate(results, start=1):
        if r.ok:
            s = r.verdict.liability_split
            rows.append([
                str(i),
                r.case_id,
                r.verdict.primary_liable_party,
                f"{s.consumer_pct:.1f}",
                f"{s.agent_platform_pct:.1f}",
                f"{s.merchant_pct:.1f}",
                f"{r.verdict.confidence:.2f}",
                f"{r.elapsed_s:.1f}",
                "ok",
            ])
        else:
            rows.append([str(i), r.case_id, "ERROR", "—", "—", "—", "—",
                         f"{r.elapsed_s:.1f}", "FAILED"])

    right = {0, 3, 4, 5, 6, 7}
    print()
    _print_md_table(headers, rows, right_cols=right)
    passed  = sum(1 for r in results if r.ok)
    failed  = len(results) - passed
    print(f"\n  Passed: {passed}/10  |  Failed: {failed}/10  |  "
          f"Wall time: {wall_elapsed:.1f}s  |  "
          f"Avg latency: {sum(r.elapsed_s for r in results)/len(results):.1f}s")

    return results


# ===========================================================================
# Part 2 — Consistency (DISP-001 × 10 sequential)
# ===========================================================================

def run_consistency() -> list[RunResult]:
    print(f"\n{'='*72}")
    print(f"  PART 2 — CONSISTENCY  (DISP-001 × 10 sequential runs)")
    print(f"{'='*72}\n")

    results: list[RunResult] = []
    for i in range(1, 11):
        print(f"  Run {i:2d}/10 … ", end="", flush=True)
        r = run_case(CASE_1, "balanced")
        results.append(r)
        if r.ok:
            s = r.verdict.liability_split
            print(f"platform={s.agent_platform_pct:.1f}%  ({r.elapsed_s:.0f}s)")
        else:
            print(f"FAILED — {r.error}")

    # Stats over successful runs
    ok_results = [r for r in results if r.ok]
    platform_pcts = [r.verdict.liability_split.agent_platform_pct for r in ok_results]
    consumer_pcts = [r.verdict.liability_split.consumer_pct        for r in ok_results]
    merchant_pcts = [r.verdict.liability_split.merchant_pct        for r in ok_results]
    confidences   = [r.verdict.confidence                           for r in ok_results]

    headers = ["Run", "Primary Liable", "Consumer%", "Platform%", "Merchant%", "Confidence"]
    rows = []
    for i, r in enumerate(results, start=1):
        if r.ok:
            s = r.verdict.liability_split
            rows.append([str(i), r.verdict.primary_liable_party,
                         f"{s.consumer_pct:.1f}", f"{s.agent_platform_pct:.1f}",
                         f"{s.merchant_pct:.1f}", f"{r.verdict.confidence:.2f}"])
        else:
            rows.append([str(i), "ERROR", "—", "—", "—", "—"])

    def _stat_row(label: str, vals: list[float], fmt=".1f") -> list[str]:
        if len(vals) < 2:
            return [label, "—", "—", "—", "—", "—"]
        return [label, "—",
                f"{statistics.mean(consumer_pcts):{fmt}}" if label == "Mean"
                    else (f"{statistics.variance(consumer_pcts):.2f}" if label == "Variance"
                    else f"{statistics.stdev(consumer_pcts):.2f}"),
                f"{statistics.mean(platform_pcts):{fmt}}" if label == "Mean"
                    else (f"{statistics.variance(platform_pcts):.2f}" if label == "Variance"
                    else f"{statistics.stdev(platform_pcts):.2f}"),
                f"{statistics.mean(merchant_pcts):{fmt}}" if label == "Mean"
                    else (f"{statistics.variance(merchant_pcts):.2f}" if label == "Variance"
                    else f"{statistics.stdev(merchant_pcts):.2f}"),
                f"{statistics.mean(confidences):.2f}" if label == "Mean" else "—"]

    right = {0, 2, 3, 4, 5}
    print()
    _print_md_table(headers, rows, right_cols=right)

    if len(ok_results) >= 2:
        print(_sep(_widths(headers, rows)))  # divider before stats
        for label in ("Mean", "Variance", "Std Dev"):
            row = _stat_row(label, platform_pcts)
            w = _widths(headers, rows)
            print(_row(row, w, right_cols=right))

        print(f"\n  Platform % variance : {statistics.variance(platform_pcts):.2f}")
        print(f"  Platform % std dev  : {statistics.stdev(platform_pcts):.2f}")
        print(f"  Platform % range    : "
              f"{min(platform_pcts):.1f}% – {max(platform_pcts):.1f}%")
        print(f"  Confidence variance : {statistics.variance(confidences):.4f}")

    return results


# ===========================================================================
# Part 3 — Cost profile (one run each of DISP-001, DISP-002, DISP-003)
# ===========================================================================

def run_cost_profiling() -> list[RunResult]:
    print(f"\n{'='*72}")
    print(f"  PART 3 — COST PROFILE  (DISP-001, DISP-002, DISP-003 × 1 run each)")
    print(f"{'='*72}\n")

    cases = [CASE_1, CASE_2, CASE_3]
    results: list[RunResult] = []
    for case in cases:
        print(f"  Running {case.case_id} … ", end="", flush=True)
        r = run_case(case, "balanced")
        results.append(r)
        print(f"{'ok' if r.ok else 'FAILED'}  ({r.elapsed_s:.0f}s)")

    # Per-agent aggregated rows
    agent_order = ["intent", "policy", "arbitrator"]
    headers = ["Case ID", "Agent", "Input Tok", "Output Tok", "Cost ($)", ""]
    cost_rows: list[list[str]] = []
    grand_input = grand_output = grand_cost = 0.0

    for r in results:
        if not r.ok:
            cost_rows.append([r.case_id, "ERROR", "—", "—", "—", ""])
            continue

        case_input = case_output = case_cost = 0.0
        for agent in agent_order:
            agent_records = [t for t in r.tokens if t.agent == agent]
            inp  = sum(t.input_tokens  for t in agent_records)
            out  = sum(t.output_tokens for t in agent_records)
            cost = sum(t.cost_usd      for t in agent_records)
            cost_rows.append([r.case_id, agent,
                               f"{inp:,}", f"{out:,}", f"{cost:.4f}", ""])
            case_input  += inp
            case_output += out
            case_cost   += cost

        # Case subtotal
        cost_rows.append([r.case_id, "── TOTAL",
                          f"{case_input:,.0f}", f"{case_output:,.0f}",
                          f"{case_cost:.4f}", ""])
        grand_input  += case_input
        grand_output += case_output
        grand_cost   += case_cost

    right = {2, 3, 4}
    print()
    _print_md_table(headers[:5], [row[:5] for row in cost_rows],
                    right_cols=right, group_col=0)

    # Grand total
    w = _widths(headers[:5], [row[:5] for row in cost_rows])
    print(_sep(w))
    print(_row(["ALL CASES", "GRAND TOTAL",
                f"{grand_input:,.0f}", f"{grand_output:,.0f}",
                f"${grand_cost:.4f}"], w, right_cols=right))

    print(f"\n  Grand total input tokens  : {grand_input:,.0f}")
    print(f"  Grand total output tokens : {grand_output:,.0f}")
    print(f"  Estimated total cost      : ${grand_cost:.4f}")
    print(f"  Avg cost per case         : ${grand_cost / max(len(results), 1):.4f}")

    return results


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale tests for the dispute resolution pipeline."
    )
    parser.add_argument(
        "--part", choices=["all", "parallel", "consistency", "cost"],
        default="all",
        help="Which part to run (default: all)",
    )
    args = parser.parse_args()

    if args.part in ("all", "parallel"):
        run_parallel()

    if args.part in ("all", "consistency"):
        run_consistency()

    if args.part in ("all", "cost"):
        run_cost_profiling()

    print()


if __name__ == "__main__":
    main()
