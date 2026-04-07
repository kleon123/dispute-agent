"""
Dispute resolution agent.

Runs a tool-use loop against the Anthropic API:
  1. parse_intent   — reconstruct what the user actually wanted
  2. compare_to_policy — find policy violations
  3. assign_liability  — compute the liability split
Then emits a structured Verdict.
"""

import json
import os
import re

import anthropic
from dotenv import load_dotenv

from schemas import DisputeCase, LiabilitySplit, Verdict
from tools import TOOL_DEFINITIONS, dispatch_tool

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are an expert dispute resolution agent for agentic commerce transactions.
A dispute arises when an AI purchasing agent buys something on a user's behalf and the user
contests the transaction.

Your job is to analyse the dispute and produce a structured verdict by calling three tools
in this exact order:

  1. parse_intent      — understand the gap between what the user wanted and what happened
  2. compare_to_policy — check the order and agent behaviour against all applicable policies
  3. assign_liability  — compute the final liability split

After all three tool calls are complete, output a verdict as a JSON object (and ONLY that
JSON object, with no surrounding prose) with this exact schema:

{
  "primary_liable_party": "<consumer|agent_platform|merchant|shared>",
  "liability_split": {
    "consumer_pct": <number>,
    "agent_platform_pct": <number>,
    "merchant_pct": <number>
  },
  "explanation": "<clear, concise explanation of your reasoning>",
  "recommended_resolution": "<concrete action: refund, partial credit, reshipment, etc.>",
  "confidence": <0.0–1.0>
}

Be precise and impartial. Your verdict will be used to settle real financial disputes."""


def _format_case(case: DisputeCase) -> str:
    """Serialise a DisputeCase into a readable message for the model."""
    inst = case.user_instruction
    log = case.agent_log
    order = case.order_record
    policy = case.policy

    return f"""## Dispute Case {case.case_id}

### User Instruction
{inst.raw_text}
Stated constraints: {inst.stated_constraints or "none"}

### Agent Log
Steps taken:
{chr(10).join(f"  {i+1}. {s}" for i, s in enumerate(log.steps))}

Agent reasoning: {log.reasoning}
Final action: {log.final_action}

### Order Record
- Order ID: {order.order_id}
- Item ordered: {order.item_name}
- Item received: {order.item_received}
- Price charged: ${order.price_charged:.2f}
- Merchant: {order.merchant_name}
- Delivery promised: {order.delivery_sla}
- Delivery actual: {order.delivery_actual}
{f"- Notes: {order.notes}" if order.notes else ""}

### Applicable Policies

Agent platform policies:
{chr(10).join(f"  • {p}" for p in policy.agent_platform_policies)}

Merchant terms:
{chr(10).join(f"  • {p}" for p in policy.merchant_terms)}

Consumer protections:
{chr(10).join(f"  • {p}" for p in policy.consumer_protections)}

Please analyse this dispute and return your verdict."""


def _parse_verdict(case_id: str, text: str) -> Verdict:
    """Extract the JSON verdict from the model's final message."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse verdict JSON from model output:\n{text}")

    split_data = data.get("liability_split", {})
    split = LiabilitySplit(
        consumer_pct=float(split_data.get("consumer_pct", 0)),
        agent_platform_pct=float(split_data.get("agent_platform_pct", 0)),
        merchant_pct=float(split_data.get("merchant_pct", 0)),
    )

    return Verdict(
        case_id=case_id,
        primary_liable_party=data.get("primary_liable_party", "unknown"),
        liability_split=split,
        explanation=data.get("explanation", ""),
        recommended_resolution=data.get("recommended_resolution", ""),
        confidence=float(data.get("confidence", 0.0)),
    )


def resolve_dispute(case: DisputeCase, verbose: bool = True) -> Verdict:
    """
    Run the full dispute resolution loop for one case.

    The agent calls parse_intent → compare_to_policy → assign_liability,
    then produces a final verdict.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages: list[dict] = [
        {"role": "user", "content": _format_case(case)}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Resolving dispute: {case.case_id}")
        print(f"{'='*60}")

    iteration = 0
    while True:
        iteration += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if verbose:
            print(f"\n[Turn {iteration}] stop_reason={response.stop_reason}")

        # Append the assistant's response to the conversation
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text block and parse verdict
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text
                    break

            if verbose:
                print("\n[Agent verdict text]")
                print(final_text)

            return _parse_verdict(case.case_id, final_text)

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                if verbose:
                    print(f"\n  → Calling tool: {tool_name}")
                    print(f"    Input keys: {list(tool_input.keys())}")

                result = dispatch_tool(tool_name, tool_input)

                if verbose:
                    result_preview = result[:300] + "..." if len(result) > 300 else result
                    print(f"    Result preview: {result_preview}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — surface the last text and bail
        for block in response.content:
            if hasattr(block, "text"):
                raise RuntimeError(
                    f"Unexpected stop_reason '{response.stop_reason}': {block.text}"
                )
        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")
