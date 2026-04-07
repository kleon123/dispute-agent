from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserInstruction:
    """What the user told the agent to do."""
    raw_text: str
    # Explicit constraints the user stated (e.g., "under $50", "same-day delivery")
    stated_constraints: list[str] = field(default_factory=list)


@dataclass
class AgentLog:
    """What the agent actually did and why."""
    steps: list[str]          # ordered list of actions the agent took
    reasoning: str            # agent's stated rationale for its final decision
    final_action: str         # the concrete action the agent ultimately executed


@dataclass
class OrderRecord:
    """The actual transaction record."""
    order_id: str
    item_name: str            # what the agent ordered
    item_received: str        # what the consumer actually received (may differ)
    price_charged: float      # in USD
    merchant_name: str
    delivery_sla: str         # promised delivery window
    delivery_actual: str      # what actually happened
    notes: str = ""


@dataclass
class Policy:
    """Applicable policies from all three parties."""
    agent_platform_policies: list[str]   # rules the agent platform imposes on its agents
    merchant_terms: list[str]            # merchant's own terms of service / fulfillment rules
    consumer_protections: list[str]      # platform or regulatory protections for the buyer


@dataclass
class DisputeCase:
    """A complete dispute case bundling all evidence."""
    case_id: str
    user_instruction: UserInstruction
    agent_log: AgentLog
    order_record: OrderRecord
    policy: Policy
    expected_verdict: Optional[str] = None  # for test cases only


@dataclass
class LiabilitySplit:
    """Percentage liability across the three parties. Must sum to 100."""
    consumer_pct: float
    agent_platform_pct: float
    merchant_pct: float

    def validate(self) -> bool:
        return abs(self.consumer_pct + self.agent_platform_pct + self.merchant_pct - 100.0) < 0.5


@dataclass
class Verdict:
    """Final dispute resolution output."""
    case_id: str
    primary_liable_party: str       # "consumer" | "agent_platform" | "merchant" | "shared"
    liability_split: LiabilitySplit
    explanation: str                # human-readable reasoning chain
    recommended_resolution: str     # concrete action (refund, partial credit, etc.)
    confidence: float               # 0.0–1.0
