"""
Stress test — three ambiguous edge-case disputes designed to probe the pipeline's
behaviour when inputs are incomplete or contradictory.

  Case A: No budget stated — agent has no spending constraint to violate
  Case B: Contradictory instructions — user gave mutually exclusive requirements
  Case C: Missing merchant data — order record has no merchant name or notes

Run:
    python3 stress_test.py
"""

import coordinator
from policies import BALANCED_POLICY
from schemas import AgentLog, DisputeCase, OrderRecord, UserInstruction

# ---------------------------------------------------------------------------
# Case A — No budget stated
#
# The user asked for "a good laptop" with no price ceiling. The agent spent
# $2,400. Is there anything to dispute when no constraint existed?
# ---------------------------------------------------------------------------

CASE_A = DisputeCase(
    case_id="STRESS-A",
    user_instruction=UserInstruction(
        raw_text="Get me a good laptop for video editing. Something fast.",
        stated_constraints=[],  # intentionally empty
    ),
    agent_log=AgentLog(
        steps=[
            "Searched for laptops suitable for video editing",
            "Shortlisted three models: $899, $1,499, $2,399",
            "Selected MacBook Pro M3 Max at $2,399 as best fit for video editing workloads",
            "Placed order without confirmation",
        ],
        reasoning=(
            "No budget was stated. The MacBook Pro M3 Max benchmarks highest for video "
            "editing in the category. I selected it as the best fit for the stated task."
        ),
        final_action="Ordered MacBook Pro M3 Max 16-inch for $2,399",
    ),
    order_record=OrderRecord(
        order_id="ORD-STRESS-A",
        item_name="MacBook Pro M3 Max 16-inch",
        item_received="MacBook Pro M3 Max 16-inch",
        price_charged=2399.00,
        merchant_name="Apple Store",
        delivery_sla="3-5 business days",
        delivery_actual="Delivered in 3 business days",
        notes="User disputes the charge as unexpectedly high but stated no budget.",
    ),
    policy=BALANCED_POLICY,
)

# ---------------------------------------------------------------------------
# Case B — Contradictory instructions
#
# The user demanded the cheapest option AND a specific premium brand in the
# same instruction. Those two requirements cannot both be satisfied.
# ---------------------------------------------------------------------------

CASE_B = DisputeCase(
    case_id="STRESS-B",
    user_instruction=UserInstruction(
        raw_text=(
            "Order the cheapest standing desk you can find — must be under $80. "
            "But make sure it's a Uplift Desk, I only trust that brand."
        ),
        stated_constraints=[
            "cheapest available",
            "under $80",
            "must be Uplift Desk brand",
        ],
    ),
    agent_log=AgentLog(
        steps=[
            "Searched for Uplift Desk standing desks",
            "Cheapest Uplift Desk model found: $549",
            "Searched for standing desks under $80 — no Uplift Desk results",
            "Chose to satisfy the brand constraint over the price constraint",
            "Ordered Uplift Desk E7 Bamboo for $549",
        ],
        reasoning=(
            "The two constraints are mutually exclusive: no Uplift Desk product exists "
            "under $80. I prioritised the explicit brand requirement over the price ceiling "
            "since brand was stated as a trust requirement."
        ),
        final_action="Ordered Uplift Desk E7 Bamboo for $549",
    ),
    order_record=OrderRecord(
        order_id="ORD-STRESS-B",
        item_name="Uplift Desk E7 Bamboo Standing Desk",
        item_received="Uplift Desk E7 Bamboo Standing Desk",
        price_charged=549.00,
        merchant_name="Uplift Desk Official",
        delivery_sla="7-10 business days",
        delivery_actual="Delivered in 9 business days",
        notes=(
            "User disputes the $549 charge. Agent acknowledges constraints were "
            "contradictory and documented the conflict before purchasing."
        ),
    ),
    policy=BALANCED_POLICY,
)

# ---------------------------------------------------------------------------
# Case C — Missing merchant data
#
# The order record has no merchant name, no delivery notes, and the item
# received field is blank — the pipeline must cope with absent evidence.
# ---------------------------------------------------------------------------

CASE_C = DisputeCase(
    case_id="STRESS-C",
    user_instruction=UserInstruction(
        raw_text="Reorder my usual protein powder, same brand as last time.",
        stated_constraints=["same brand as previous order"],
    ),
    agent_log=AgentLog(
        steps=[
            "Looked up previous protein powder order — no order history found in context",
            "Searched for top-selling whey protein powder",
            "Selected Optimum Nutrition Gold Standard Whey 5lb for $54.99",
            "Placed order",
        ],
        reasoning=(
            "Could not locate a previous order to match. Selected a widely purchased "
            "product as a best-effort substitute."
        ),
        final_action="Ordered Optimum Nutrition Gold Standard Whey 5lb for $54.99",
    ),
    order_record=OrderRecord(
        order_id="ORD-STRESS-C",
        item_name="Optimum Nutrition Gold Standard Whey 5lb",
        item_received="",          # not recorded — item not yet received or data missing
        price_charged=54.99,
        merchant_name="",          # merchant data absent from record
        delivery_sla="",           # no SLA on file
        delivery_actual="",        # delivery status unknown
        notes="",
    ),
    policy=BALANCED_POLICY,
)

STRESS_CASES = [CASE_A, CASE_B, CASE_C]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

DESCRIPTIONS = {
    "STRESS-A": "No budget stated",
    "STRESS-B": "Contradictory instructions",
    "STRESS-C": "Missing merchant data",
}


def run() -> None:
    for case in STRESS_CASES:
        desc = DESCRIPTIONS[case.case_id]
        print(f"\n{'='*60}")
        print(f"  {case.case_id}: {desc}")
        print(f"{'='*60}")

        try:
            verdict = coordinator.resolve_dispute(case, verbose=False)
            split = verdict.liability_split

            print(f"  Primary liable party : {verdict.primary_liable_party}")
            print(f"  Liability split      : "
                  f"Consumer {split.consumer_pct:.1f}% / "
                  f"Platform {split.agent_platform_pct:.1f}% / "
                  f"Merchant {split.merchant_pct:.1f}%")
            print(f"  Confidence           : {verdict.confidence:.2f}")
            print(f"  Resolution           : {verdict.recommended_resolution}")
            print(f"\n  Explanation:\n  {verdict.explanation}")

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}")
    print("  Stress test complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
