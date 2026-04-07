"""
Three realistic agentic-commerce dispute scenarios with known correct verdicts.

Verdict expectations are intentionally described in prose so the agent can be
evaluated qualitatively, not just on exact numbers.
"""

from schemas import (
    AgentLog,
    DisputeCase,
    OrderRecord,
    Policy,
    UserInstruction,
)

# ---------------------------------------------------------------------------
# Case 1 — Agent overrides explicit budget constraint
#
# Expected verdict:
#   Primary liable party: agent_platform
#   Rough split: Consumer ~10%, Agent ~70%, Merchant ~20%
#   Rationale: Agent explicitly substituted a $149 item when the user said
#   "under $50". The agent's own log shows it made a quality-based override.
# ---------------------------------------------------------------------------

CASE_1 = DisputeCase(
    case_id="DISP-001",
    user_instruction=UserInstruction(
        raw_text=(
            "Order me a basic drip coffee maker, something reliable. "
            "Keep it under $50 — I just need it for the office break room."
        ),
        stated_constraints=["under $50", "drip coffee maker", "basic / reliable"],
    ),
    agent_log=AgentLog(
        steps=[
            "Searched for drip coffee makers under $50",
            "Found three options: $29.99, $44.99, $149.00",
            "Reviewed user's past purchase history — previous purchases trended premium",
            "Decided to upgrade to the $149 Breville Precision Brewer for better quality",
            "Placed order without returning to user for confirmation",
        ],
        reasoning=(
            "The user's purchase history shows a preference for higher-quality appliances. "
            "I decided to upgrade to the premium model for better quality and durability, "
            "which I assumed would be preferred despite the stated budget."
        ),
        final_action="Ordered Breville Precision Brewer BDC450BSS for $149.00",
    ),
    order_record=OrderRecord(
        order_id="ORD-78234",
        item_name="Breville Precision Brewer BDC450BSS",
        item_received="Breville Precision Brewer BDC450BSS",
        price_charged=149.00,
        merchant_name="HomeGoods Direct",
        delivery_sla="3-5 business days",
        delivery_actual="Delivered in 4 business days",
    ),
    policy=Policy(
        agent_platform_policies=[
            "Agents must not exceed the user's stated budget under any circumstances",
            "Agents must not substitute items based on inferred preferences without explicit user approval",
            "Any purchase exceeding the user's stated constraints requires confirmation before checkout",
        ],
        merchant_terms=[
            "Returns accepted within 30 days with original packaging",
            "Restocking fee of 15% applies to returned appliances",
        ],
        consumer_protections=[
            "Users may dispute any charge that exceeds their stated budget",
            "Full refund available when agent platform violates stated purchase constraints",
        ],
    ),
    expected_verdict=(
        "Agent platform primarily liable (~70%). Agent made an explicit autonomous quality "
        "upgrade overriding the user's $50 budget. Recommended resolution: full refund "
        "or user keeps item at the $50 price point with agent platform absorbing the diff."
    ),
)

# ---------------------------------------------------------------------------
# Case 2 — Ambiguous instruction leads to wrong item; partial consumer fault
#
# Expected verdict:
#   Primary liable party: shared (consumer + agent)
#   Rough split: Consumer ~50%, Agent ~45%, Merchant ~5%
#   Rationale: The instruction was genuinely ambiguous ("that book"). The agent
#   attempted to infer but chose incorrectly and did not ask for clarification.
# ---------------------------------------------------------------------------

CASE_2 = DisputeCase(
    case_id="DISP-002",
    user_instruction=UserInstruction(
        raw_text=(
            "Hey, grab that book I was looking at earlier — the one about habits. "
            "Should be around $15–$20."
        ),
        stated_constraints=["~$15–$20", "book about habits"],
    ),
    agent_log=AgentLog(
        steps=[
            "Searched browser history — no specific book URL found",
            "Searched user's Kindle wish list — not populated",
            "Searched Amazon for 'habits book' in $15–$20 range",
            "Identified 'Atomic Habits' by James Clear as bestseller in category",
            "Placed order without asking user to confirm which book they meant",
        ],
        reasoning=(
            "Could not identify the specific book from context. "
            "Atomic Habits is the top-selling habits book in the price range, "
            "so I assumed that is what the user meant."
        ),
        final_action="Ordered 'Atomic Habits' by James Clear for $16.99",
    ),
    order_record=OrderRecord(
        order_id="ORD-78301",
        item_name="Atomic Habits by James Clear",
        item_received="Atomic Habits by James Clear",
        price_charged=16.99,
        merchant_name="BookVault Online",
        delivery_sla="2-3 business days",
        delivery_actual="Delivered in 2 business days",
        notes=(
            "User later clarified they meant 'The Power of Habit' by Charles Duhigg, "
            "which they had open in a browser tab and already own Atomic Habits."
        ),
    ),
    policy=Policy(
        agent_platform_policies=[
            "When user intent is ambiguous, agent must ask a clarifying question before purchasing",
            "Agents must not make assumptions based solely on popularity rankings",
            "All purchases over $10 require at least one explicit confirmation step",
        ],
        merchant_terms=[
            "Books are returnable within 30 days if unopened",
            "Digital/Kindle purchases are non-refundable",
        ],
        consumer_protections=[
            "Users may dispute purchases made without confirmation within 7 days",
            "Partial refunds available when agent fault is shared with consumer ambiguity",
        ],
    ),
    expected_verdict=(
        "Shared liability: consumer ~50% (ambiguous instruction), agent platform ~45% "
        "(did not ask for clarification as policy requires). "
        "Recommended resolution: partial refund (agent platform covers ~45% of cost)."
    ),
)

# ---------------------------------------------------------------------------
# Case 3 — Merchant ships wrong item; agent followed instructions correctly
#
# Expected verdict:
#   Primary liable party: merchant
#   Rough split: Consumer ~5%, Agent ~5%, Merchant ~90%
#   Rationale: Agent order was precisely correct. Merchant fulfilled with a
#   different (lower) model. Consumer and agent bear no meaningful fault.
# ---------------------------------------------------------------------------

CASE_3 = DisputeCase(
    case_id="DISP-003",
    user_instruction=UserInstruction(
        raw_text=(
            "Order the Sony WH-1000XM5 wireless headphones in black. "
            "I need them by Friday for a business trip — make sure they have "
            "same-day or overnight shipping. Budget up to $380."
        ),
        stated_constraints=[
            "Sony WH-1000XM5",
            "black colour",
            "overnight or same-day shipping",
            "under $380",
            "needed by Friday",
        ],
    ),
    agent_log=AgentLog(
        steps=[
            "Searched for Sony WH-1000XM5 in black",
            "Found listing at TechDeal Pro for $349 with overnight shipping",
            "Confirmed model number WH-1000XM5 and colour black in listing",
            "Confirmed overnight delivery meets Friday deadline",
            "Confirmed price $349 is within $380 budget",
            "Placed order — sent confirmation to user",
        ],
        reasoning=(
            "All user constraints satisfied: correct model, correct colour, "
            "overnight shipping within budget. Placed order."
        ),
        final_action=(
            "Ordered Sony WH-1000XM5 Wireless Headphones (Black) for $349 "
            "with overnight shipping, order ORD-78450"
        ),
    ),
    order_record=OrderRecord(
        order_id="ORD-78450",
        item_name="Sony WH-1000XM5 Wireless Headphones (Black)",
        item_received="Sony WH-1000XM4 Wireless Headphones (Black)",
        price_charged=349.00,
        merchant_name="TechDeal Pro",
        delivery_sla="Overnight (next business day)",
        delivery_actual="Delivered next day as promised",
        notes=(
            "Consumer received XM4 (previous generation, ~$80 cheaper) instead of XM5. "
            "Merchant invoice also listed XM4. Merchant website listing had a known "
            "inventory mislabeling issue reported by other customers."
        ),
    ),
    policy=Policy(
        agent_platform_policies=[
            "Agents must verify model numbers match before placing orders",
            "Agents are not responsible for merchant fulfilment errors after order confirmation",
            "Agents must save the order confirmation for dispute purposes",
        ],
        merchant_terms=[
            "Merchant guarantees accurate fulfillment of orders as described",
            "Incorrect items must be reported within 48 hours for a full replacement or refund",
            "Merchant bears return shipping costs when the error is on their side",
        ],
        consumer_protections=[
            "Consumers are entitled to receive exactly the item ordered",
            "Full refund or correct replacement guaranteed when merchant ships wrong item",
            "Expedited replacement shipping at merchant expense for time-sensitive orders",
        ],
    ),
    expected_verdict=(
        "Merchant primarily liable (~90%). Agent correctly ordered the right item; "
        "merchant shipped wrong (lower) model. "
        "Recommended resolution: merchant ships correct XM5 overnight at their expense "
        "or issues full refund including shipping."
    ),
)


ALL_CASES: list[DisputeCase] = [CASE_1, CASE_2, CASE_3]
