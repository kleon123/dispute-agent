"""
Tool implementations for the dispute resolution agent.

Each function performs real analysis and returns a JSON string that Claude
incorporates into its reasoning before calling the next tool.
"""

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Tool 1 — parse_intent
# ---------------------------------------------------------------------------

def parse_intent(
    user_instruction: str,
    agent_reasoning: str,
    agent_action: str,
    order_details: str,
) -> str:
    """
    Reconstructs the user's true intent and measures how well the agent's
    action aligned with it. Returns a structured JSON analysis.
    """
    findings: list[str] = []
    agent_fault_signals: list[str] = []
    consumer_fault_signals: list[str] = []

    instruction_lower = user_instruction.lower()
    reasoning_lower = agent_reasoning.lower()
    order_lower = order_details.lower()

    # --- Price constraint check ---
    price_limit: float | None = None
    limit_match = re.search(
        r"(?:under|below|less than|no more than|max(?:imum)?|budget[:\s]+)\s*\$?([\d,]+)",
        instruction_lower,
    )
    if limit_match:
        price_limit = float(limit_match.group(1).replace(",", ""))
        findings.append(f"User stated explicit price ceiling: ${price_limit:.2f}")

    order_prices = [
        float(p.replace(",", "")) for p in re.findall(r"\$([\d,]+(?:\.\d{1,2})?)", order_details)
    ]
    if price_limit and order_prices:
        max_charged = max(order_prices)
        if max_charged > price_limit:
            overage = max_charged - price_limit
            findings.append(
                f"VIOLATION: Order price ${max_charged:.2f} exceeds user's stated limit "
                f"of ${price_limit:.2f} by ${overage:.2f}"
            )
            agent_fault_signals.append("exceeded_budget_constraint")

    # --- Specific item / model check ---
    specificity_markers = [
        r"\bmodel\b", r"\bexact\b", r"\bspecific\b", r"\bsame\b",
        r"\bthis one\b", r"\bpart number\b", r"\bsku\b",
    ]
    is_specific_request = any(re.search(p, instruction_lower) for p in specificity_markers)
    if is_specific_request:
        findings.append("User requested a specific item — agent must match exactly, not substitute")

    # --- Ambiguity check ---
    ambiguity_phrases = [
        "something", "anything", "whatever", "you choose", "your pick",
        "the usual", "that thing", "you know what i mean",
    ]
    is_ambiguous = any(phrase in instruction_lower for phrase in ambiguity_phrases)
    if is_ambiguous:
        findings.append("User instruction contains ambiguous language — some interpretation by agent was required")
        consumer_fault_signals.append("ambiguous_instruction")

    # --- Autonomy override check ---
    override_phrases = [
        "better quality", "upgrade", "premium", "i decided", "i assumed",
        "thought the user would prefer", "similar but", "alternative",
    ]
    agent_overrode = any(phrase in reasoning_lower for phrase in override_phrases)
    if agent_overrode:
        findings.append(
            "CRITICAL: Agent reasoning suggests autonomous override of user preference"
        )
        agent_fault_signals.append("autonomous_preference_override")

    # --- Delivery constraint check ---
    delivery_keywords = ["urgent", "asap", "today", "tonight", "same.day", "overnight", "rush", "by tomorrow"]
    has_delivery_constraint = any(kw in instruction_lower for kw in delivery_keywords)
    if has_delivery_constraint:
        findings.append("User specified a delivery urgency constraint")

    # --- Intent alignment score (0–100) ---
    score = 100
    score -= len(agent_fault_signals) * 35
    score -= len(consumer_fault_signals) * 15
    score = max(0, min(100, score))

    if not findings:
        findings.append("No explicit misalignment detected; agent action appears consistent with user instruction")

    result = {
        "intent_alignment_score": score,
        "findings": findings,
        "agent_fault_signals": agent_fault_signals,
        "consumer_fault_signals": consumer_fault_signals,
        "price_limit_stated": price_limit,
        "is_specific_item_request": is_specific_request,
        "is_ambiguous_instruction": is_ambiguous,
        "agent_autonomy_override_detected": agent_overrode,
        "summary": (
            f"Alignment score {score}/100. "
            + (
                f"Agent fault indicators: {agent_fault_signals}. " if agent_fault_signals else ""
            )
            + (
                f"Consumer fault indicators: {consumer_fault_signals}." if consumer_fault_signals else ""
            )
        ),
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 2 — compare_to_policy
# ---------------------------------------------------------------------------

def compare_to_policy(
    order_summary: str,
    agent_behavior_summary: str,
    platform_policies: list[str],
    merchant_terms: list[str],
    consumer_protections: list[str],
) -> str:
    """
    Checks the order and agent behavior against all applicable policies.
    Returns a structured JSON report of violations and their implied liability.
    """
    violations: list[dict[str, str]] = []
    compliant_items: list[str] = []

    combined_text = (order_summary + " " + agent_behavior_summary).lower()

    # Violation detection heuristics per policy layer

    def check_policies(policies: list[str], party: str) -> None:
        for policy in policies:
            policy_lower = policy.lower()

            # Budget / authorization policies
            if any(kw in policy_lower for kw in ["budget", "spending limit", "authorized amount", "price cap"]):
                if any(kw in combined_text for kw in ["exceeded", "over budget", "above limit", "overspent"]):
                    violations.append({
                        "policy": policy,
                        "implicates": party,
                        "severity": "HIGH",
                        "detail": "Order appears to have violated a spending authorization rule",
                    })
                    return

            # Substitution / wrong item policies
            if any(kw in policy_lower for kw in ["no substitution", "exact item", "as specified", "no alternatives"]):
                if any(kw in combined_text for kw in ["substitut", "different model", "wrong item", "alternative", "replaced with"]):
                    violations.append({
                        "policy": policy,
                        "implicates": party,
                        "severity": "HIGH",
                        "detail": "Item substituted without user consent in violation of no-substitution policy",
                    })
                    return

            # Confirmation / consent policies
            if any(kw in policy_lower for kw in ["confirm", "consent", "approve", "explicit permission", "user must approve"]):
                if any(kw in combined_text for kw in ["without confirmation", "no approval", "skipped confirmation", "auto-approved"]):
                    violations.append({
                        "policy": policy,
                        "implicates": party,
                        "severity": "MEDIUM",
                        "detail": "Agent proceeded without required user confirmation step",
                    })
                    return

            # Return / refund merchant terms
            if any(kw in policy_lower for kw in ["no return", "all sales final", "non-refundable"]):
                if any(kw in combined_text for kw in ["refund", "return", "dispute"]):
                    violations.append({
                        "policy": policy,
                        "implicates": party,
                        "severity": "LOW",
                        "detail": "Merchant no-return policy may limit refund options",
                    })
                    return

            # Fulfillment accuracy
            if any(kw in policy_lower for kw in ["correct item", "accurate fulfillment", "ship as ordered", "described"]):
                if any(kw in combined_text for kw in ["wrong item", "mislabeled", "incorrect", "different from", "not what was ordered"]):
                    violations.append({
                        "policy": policy,
                        "implicates": party,
                        "severity": "HIGH",
                        "detail": "Merchant shipped an item that does not match what was ordered",
                    })
                    return

            compliant_items.append(f"[{party}] {policy}")

    check_policies(platform_policies, "agent_platform")
    check_policies(merchant_terms, "merchant")
    check_policies(consumer_protections, "consumer")

    high_violations = [v for v in violations if v["severity"] == "HIGH"]
    implicated_parties = list({v["implicates"] for v in violations})

    result = {
        "total_violations": len(violations),
        "high_severity_violations": len(high_violations),
        "violations": violations,
        "implicated_parties": implicated_parties,
        "compliant_policy_count": len(compliant_items),
        "summary": (
            f"{len(violations)} policy violation(s) found "
            f"({len(high_violations)} high-severity). "
            f"Parties implicated: {implicated_parties or ['none']}."
        ),
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 3 — assign_liability
# ---------------------------------------------------------------------------

def assign_liability(
    intent_analysis: str,
    policy_violations: str,
    case_summary: str,
) -> str:
    """
    Applies a weighted scoring rubric to the outputs of parse_intent and
    compare_to_policy, then returns a recommended liability split (percentages
    summing to 100) with per-party justification.
    """
    try:
        intent = json.loads(intent_analysis)
    except json.JSONDecodeError:
        intent = {}

    try:
        policy = json.loads(policy_violations)
    except json.JSONDecodeError:
        policy = {}

    # --- Scoring weights (raw points, normalised to 100% at end) ---
    scores: dict[str, float] = {
        "consumer": 0.0,
        "agent_platform": 0.0,
        "merchant": 0.0,
    }
    reasoning_parts: dict[str, list[str]] = {
        "consumer": [],
        "agent_platform": [],
        "merchant": [],
    }

    # Intent analysis signals
    if intent.get("agent_autonomy_override_detected"):
        scores["agent_platform"] += 40
        reasoning_parts["agent_platform"].append(
            "Agent made an autonomous decision that deviated from user preference (+40)"
        )

    if "exceeded_budget_constraint" in intent.get("agent_fault_signals", []):
        scores["agent_platform"] += 35
        reasoning_parts["agent_platform"].append(
            "Agent exceeded the user's explicit budget constraint (+35)"
        )

    if "ambiguous_instruction" in intent.get("consumer_fault_signals", []):
        scores["consumer"] += 25
        reasoning_parts["consumer"].append(
            "User instruction was ambiguous, requiring agent interpretation (+25)"
        )

    alignment_score = intent.get("intent_alignment_score", 100)
    if alignment_score < 50:
        scores["agent_platform"] += 20
        reasoning_parts["agent_platform"].append(
            f"Low intent alignment score ({alignment_score}/100) indicates substantial agent deviation (+20)"
        )
    elif alignment_score >= 85:
        # Agent followed intent well — any remaining fault likely lies elsewhere
        scores["agent_platform"] = max(0.0, scores["agent_platform"] - 10)
        reasoning_parts["agent_platform"].append(
            f"High intent alignment score ({alignment_score}/100) reduces agent culpability (-10)"
        )

    # Policy violation signals
    for violation in policy.get("violations", []):
        party = violation.get("implicates", "")
        severity = violation.get("severity", "LOW")
        weight = {"HIGH": 30, "MEDIUM": 15, "LOW": 5}.get(severity, 5)
        if party in scores:
            scores[party] += weight
            reasoning_parts[party].append(
                f"Policy violation ({severity}): \"{violation.get('policy', '')}\" (+{weight})"
            )

    # If no signals fired at all, assign a small base consumer share
    # (they initiated the transaction)
    total_raw = sum(scores.values())
    if total_raw == 0:
        scores = {"consumer": 10.0, "agent_platform": 45.0, "merchant": 45.0}
        for k in reasoning_parts:
            reasoning_parts[k].append("No clear fault signals — baseline split applied")
        total_raw = 100.0

    # Ensure every party has at least 5% if they have any signal
    for party in scores:
        if scores[party] > 0 and scores[party] < 5:
            scores[party] = 5.0

    # Normalise to 100
    total_raw = sum(scores.values())
    normalised = {k: round(v / total_raw * 100, 1) for k, v in scores.items()}

    # Fix rounding drift
    drift = 100.0 - sum(normalised.values())
    max_party = max(normalised, key=lambda k: normalised[k])
    normalised[max_party] = round(normalised[max_party] + drift, 1)

    primary = max(normalised, key=lambda k: normalised[k])
    primary_pct = normalised[primary]
    if primary_pct < 50:
        primary = "shared"

    result = {
        "liability_split": {
            "consumer_pct": normalised["consumer"],
            "agent_platform_pct": normalised["agent_platform"],
            "merchant_pct": normalised["merchant"],
        },
        "primary_liable_party": primary,
        "reasoning_per_party": reasoning_parts,
        "summary": (
            f"Primary liable party: {primary}. "
            f"Consumer {normalised['consumer']}% / "
            f"Agent platform {normalised['agent_platform']}% / "
            f"Merchant {normalised['merchant']}%."
        ),
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# API tool definitions (passed to Anthropic messages.create)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "parse_intent",
        "description": (
            "Reconstruct the user's true intent from their instruction and evaluate "
            "how well the agent's action aligned with it. Call this first to understand "
            "the gap between what was asked and what was done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_instruction": {
                    "type": "string",
                    "description": "The exact instruction the user gave to the agent",
                },
                "agent_reasoning": {
                    "type": "string",
                    "description": "The agent's stated reasoning for its final decision",
                },
                "agent_action": {
                    "type": "string",
                    "description": "The concrete action the agent ultimately executed",
                },
                "order_details": {
                    "type": "string",
                    "description": "Key details from the order record (item, price, delivery)",
                },
            },
            "required": ["user_instruction", "agent_reasoning", "agent_action", "order_details"],
        },
    },
    {
        "name": "compare_to_policy",
        "description": (
            "Compare the order and agent behavior against all applicable policies "
            "(platform rules, merchant terms, consumer protections) to identify violations "
            "and which party each implicates. Call this after parse_intent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_summary": {
                    "type": "string",
                    "description": "One-paragraph summary of what was ordered and what was received",
                },
                "agent_behavior_summary": {
                    "type": "string",
                    "description": "One-paragraph summary of how the agent behaved during the transaction",
                },
                "platform_policies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent platform policy statements",
                },
                "merchant_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of merchant terms of service statements",
                },
                "consumer_protections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of consumer protection rules",
                },
            },
            "required": [
                "order_summary",
                "agent_behavior_summary",
                "platform_policies",
                "merchant_terms",
                "consumer_protections",
            ],
        },
    },
    {
        "name": "assign_liability",
        "description": (
            "Apply a weighted scoring rubric to the intent analysis and policy violation "
            "results to produce a final liability split (percentages summing to 100) across "
            "consumer, agent platform, and merchant. Call this last."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_analysis": {
                    "type": "string",
                    "description": "JSON string returned by parse_intent",
                },
                "policy_violations": {
                    "type": "string",
                    "description": "JSON string returned by compare_to_policy",
                },
                "case_summary": {
                    "type": "string",
                    "description": "Brief narrative summary of the dispute for context",
                },
            },
            "required": ["intent_analysis", "policy_violations", "case_summary"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, inputs: dict[str, Any]) -> str:
    if name == "parse_intent":
        return parse_intent(**inputs)
    if name == "compare_to_policy":
        return compare_to_policy(**inputs)
    if name == "assign_liability":
        return assign_liability(**inputs)
    return json.dumps({"error": f"Unknown tool: {name}"})
