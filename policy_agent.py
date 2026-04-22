"""
Policy Agent — Stage 2 of the dispute resolution pipeline.

Compares the transaction against platform policies, merchant terms, and consumer protections.
Takes the intent analysis from the IntentAgent and the full DisputeCase.
Returns a structured policy violation report JSON dict.
"""

import json
import os
import re

import anthropic
from dotenv import load_dotenv

from schemas import DisputeCase
from tools import TOOL_DEFINITIONS, dispatch_tool

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a policy compliance specialist for agentic commerce dispute resolution.

Your sole responsibility is to check the transaction and agent behaviour against all applicable
policies — platform rules, merchant terms, and consumer protections — and identify every violation.

Steps:
1. Review the intent analysis provided and the order/policy details in the dispute.
2. Call the compare_to_policy tool with the relevant inputs.
3. After receiving the tool result, output ONLY a JSON object (no surrounding prose) with this schema:

{
  "total_violations": <integer>,
  "high_severity_violations": <integer>,
  "violations": [
    {
      "policy": "<policy text>",
      "implicates": "<consumer|agent_platform|merchant>",
      "severity": "<HIGH|MEDIUM|LOW>",
      "detail": "<explanation>"
    }
  ],
  "implicated_parties": [<list of party strings>],
  "compliant_policy_count": <integer>,
  "key_risks": [<strings — most important risks for arbitration>],
  "summary": "<one-sentence summary of the policy analysis>"
}

Do not add commentary outside the JSON object."""

_POLICY_TOOL = next(t for t in TOOL_DEFINITIONS if t["name"] == "compare_to_policy")


def run(case: DisputeCase, intent_analysis: dict, verbose: bool = False) -> dict:
    """
    Identify policy violations for one dispute case.

    Args:
        case: The full dispute case (used for order details and policies).
        intent_analysis: The dict returned by intent_agent.run().

    Returns a dict matching the policy violation report schema above.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    order = case.order_record
    log = case.agent_log
    policy = case.policy

    # Derive order and agent summaries for the tool
    order_summary = (
        f"Order {order.order_id}: {order.item_name} ordered from {order.merchant_name}, "
        f"price charged ${order.price_charged:.2f}. "
        f"Item received: {order.item_received}. "
        f"Delivery promised: {order.delivery_sla}, actual: {order.delivery_actual}."
        + (f" Notes: {order.notes}" if order.notes else "")
    )
    agent_behavior_summary = (
        f"Agent steps: {'; '.join(log.steps)}. "
        f"Agent reasoning: {log.reasoning}. "
        f"Final action: {log.final_action}."
    )

    # Incorporate intent signals so the model can make a richer call
    intent_context = (
        f"Intent analysis summary: {intent_analysis.get('summary', 'N/A')}. "
        f"Alignment score: {intent_analysis.get('intent_alignment_score', 'N/A')}/100. "
        f"Agent fault signals: {intent_analysis.get('agent_fault_signals', [])}. "
        f"Consumer fault signals: {intent_analysis.get('consumer_fault_signals', [])}. "
        f"Budget violation detected: {intent_analysis.get('budget_violation_detected', False)}."
    )

    user_message = f"""Check this dispute for policy violations.

{intent_context}

Order summary: {order_summary}

Agent behaviour summary: {agent_behavior_summary}

Applicable policies:

Agent platform policies:
{chr(10).join(f"  • {p}" for p in policy.agent_platform_policies)}

Merchant terms:
{chr(10).join(f"  • {p}" for p in policy.merchant_terms)}

Consumer protections:
{chr(10).join(f"  • {p}" for p in policy.consumer_protections)}

Call compare_to_policy, then output your JSON violation report."""

    messages: list[dict] = [{"role": "user", "content": user_message}]

    if verbose:
        print("[PolicyAgent] Starting policy comparison...")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[_POLICY_TOOL],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    if verbose:
                        print(f"[PolicyAgent] Called tool: {block.name}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    text = re.sub(r"```(?:json)?\s*", "", block.text).strip().rstrip("`").strip()
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        match = re.search(r"\{.*\}", text, re.DOTALL)
                        if match:
                            data = json.loads(match.group())
                        else:
                            raise ValueError(f"PolicyAgent: cannot parse JSON from:\n{text}")
                    data.setdefault("key_risks", [v["detail"] for v in data.get("violations", [])
                                                  if v.get("severity") == "HIGH"])
                    if verbose:
                        print(f"[PolicyAgent] Done. Violations found: {data.get('total_violations')}")
                    return data

            raise ValueError("PolicyAgent produced no text output")

        raise RuntimeError(f"PolicyAgent unexpected stop_reason: {response.stop_reason}")
