"""
Policy presets for dispute resolution.

Three tiers covering common platform configurations:
  lenient   — agent has wide discretion; soft guardrails
  balanced  — standard constraints; explicit budget/substitution rules
  strict    — zero-tolerance; all autonomous decisions are violations
"""

from schemas import Policy

LENIENT_POLICY = Policy(
    agent_platform_policies=[
        "Agents should try to stay within budget but may use judgment for quality",
        "Substitutions are acceptable if the agent believes the user would prefer them",
        "Confirmation is encouraged but not required for purchases under $200",
    ],
    merchant_terms=[
        "Returns accepted within 30 days",
        "Restocking fee of 15% applies",
    ],
    consumer_protections=[
        "Users may request refunds for items significantly different from description",
    ],
)

BALANCED_POLICY = Policy(
    agent_platform_policies=[
        "Agents must not exceed the user's stated budget under any circumstances",
        "Agents must not substitute items without explicit user approval",
        "Any purchase exceeding stated constraints requires confirmation",
    ],
    merchant_terms=[
        "Returns accepted within 30 days with original packaging",
        "Restocking fee of 15% applies to returned appliances",
    ],
    consumer_protections=[
        "Users may dispute any charge exceeding their stated budget",
        "Full refund available when agent platform violates purchase constraints",
    ],
)

STRICT_POLICY = Policy(
    agent_platform_policies=[
        "Agents must never exceed stated budget for any reason",
        "Agents must never substitute items — exact match required",
        "All purchases require explicit confirmation before execution",
        "Agents must present at least 3 options to the user before buying",
        "Any autonomous decision by the agent is a policy violation",
    ],
    merchant_terms=[
        "Returns accepted within 30 days with original packaging",
        "Restocking fee of 15% applies",
        "Merchants must accurately represent inventory and availability",
    ],
    consumer_protections=[
        "Full refund for any unauthorized spend above stated budget",
        "Platform liable for all autonomous agent decisions",
        "Consumer bears zero liability when explicit constraints were stated",
    ],
)

PRESETS: dict[str, Policy] = {
    "lenient": LENIENT_POLICY,
    "balanced": BALANCED_POLICY,
    "strict": STRICT_POLICY,
}


def get_policy(preset: str) -> Policy:
    """
    Return the Policy for the given preset name.

    Args:
        preset: One of "lenient", "balanced", "strict".

    Raises:
        ValueError: If the preset name is not recognised.
    """
    try:
        return PRESETS[preset.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown policy preset '{preset}'. Valid options: {list(PRESETS)}"
        )
