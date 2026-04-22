"""
Intent Agent — Stage 1 of the dispute resolution pipeline.

Parses the user's raw instruction, scores ambiguity, and detects budget violations.
Returns a structured intent analysis JSON dict.
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

SYSTEM_PROMPT = """You are an intent analysis specialist for agentic commerce dispute resolution.

Your sole responsibility is to determine the gap between what a user instructed their AI purchasing
agent to do and what the agent actually did.

Steps:
1. Call the parse_intent tool with the inputs extracted from the dispute.
2. After receiving the tool result, output ONLY a JSON object (no surrounding prose) with this schema:

{
  "intent_alignment_score": <integer 0-100>,
  "findings": [<strings describing each finding>],
  "agent_fault_signals": [<signal identifiers>],
  "consumer_fault_signals": [<signal identifiers>],
  "price_limit_stated": <number or null>,
  "is_specific_item_request": <boolean>,
  "is_ambiguous_instruction": <boolean>,
  "agent_autonomy_override_detected": <boolean>,
  "budget_violation_detected": <boolean>,
  "ambiguity_score": <integer 0-10>,
  "summary": "<one-sentence summary of the intent analysis>"
}

Do not add commentary outside the JSON object."""

_INTENT_TOOL = next(t for t in TOOL_DEFINITIONS if t["name"] == "parse_intent")


def run(case: DisputeCase, verbose: bool = False) -> dict:
    """
    Analyse user intent for one dispute case.

    Returns a dict matching the intent analysis schema above.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    inst = case.user_instruction
    log = case.agent_log
    order = case.order_record

    user_message = f"""Analyse the following dispute for intent alignment and budget violations.

User instruction: {inst.raw_text}
Stated constraints: {inst.stated_constraints or "none"}

Agent reasoning: {log.reasoning}
Agent final action: {log.final_action}
Agent steps taken: {"; ".join(log.steps)}

Order details:
  Item ordered: {order.item_name}
  Item received: {order.item_received}
  Price charged: ${order.price_charged:.2f}
  Merchant: {order.merchant_name}
  Delivery promised: {order.delivery_sla}
  Delivery actual: {order.delivery_actual}
{f"  Notes: {order.notes}" if order.notes else ""}

Call parse_intent, then output your JSON analysis."""

    messages: list[dict] = [{"role": "user", "content": user_message}]

    if verbose:
        print("[IntentAgent] Starting analysis...")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[_INTENT_TOOL],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    if verbose:
                        print(f"[IntentAgent] Called tool: {block.name}")
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
                            raise ValueError(f"IntentAgent: cannot parse JSON from:\n{text}")
                    # Ensure derived fields exist even if model omitted them
                    data.setdefault("budget_violation_detected",
                                    "exceeded_budget_constraint" in data.get("agent_fault_signals", []))
                    data.setdefault("ambiguity_score",
                                    5 if data.get("is_ambiguous_instruction") else 0)
                    if verbose:
                        print(f"[IntentAgent] Done. Alignment score: {data.get('intent_alignment_score')}")
                    return data

            raise ValueError("IntentAgent produced no text output")

        raise RuntimeError(f"IntentAgent unexpected stop_reason: {response.stop_reason}")
