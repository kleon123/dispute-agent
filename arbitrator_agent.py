"""
Arbitrator Agent — Stage 3 of the dispute resolution pipeline.

Takes the intent analysis (Stage 1) and policy violation report (Stage 2) and produces
the final liability verdict: percentage split, explanation, recommended resolution, and
confidence score.
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
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a dispute arbitrator for agentic commerce transactions.

You will receive the outputs of two prior analysis stages:
  • Intent analysis  — alignment between what the user wanted and what the agent did
  • Policy report    — policy violations and which parties they implicate

Your sole responsibility is to synthesise these into a final, binding liability verdict.

Steps:
1. Call the assign_liability tool with the intent analysis and policy violations JSON strings.
2. After receiving the tool result, output ONLY a JSON object (no surrounding prose) with this schema:

{
  "primary_liable_party": "<consumer|agent_platform|merchant|shared>",
  "liability_split": {
    "consumer_pct": <number>,
    "agent_platform_pct": <number>,
    "merchant_pct": <number>
  },
  "explanation": "<clear, impartial reasoning chain referencing specific findings>",
  "recommended_resolution": "<concrete action: full refund, partial credit, reshipment, no action, etc.>",
  "confidence": <0.0–1.0>
}

Percentages must sum to 100. Be precise and impartial — your verdict settles a real financial dispute.
Do not add commentary outside the JSON object."""

_LIABILITY_TOOL = next(t for t in TOOL_DEFINITIONS if t["name"] == "assign_liability")


def run(
    case: DisputeCase,
    intent_analysis: dict,
    policy_report: dict,
    verbose: bool = False,
) -> Verdict:
    """
    Produce the final liability verdict for one dispute case.

    Args:
        case: The full dispute case (used for narrative context).
        intent_analysis: Dict returned by intent_agent.run().
        policy_report: Dict returned by policy_agent.run().

    Returns a Verdict dataclass instance.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    order = case.order_record
    inst = case.user_instruction

    case_summary = (
        f"Case {case.case_id}: User instructed agent to '{inst.raw_text}'. "
        f"Agent ordered {order.item_name} from {order.merchant_name} for ${order.price_charged:.2f}. "
        f"User received: {order.item_received}. "
        f"Intent alignment score: {intent_analysis.get('intent_alignment_score', 'N/A')}/100. "
        f"Policy violations: {policy_report.get('total_violations', 0)} "
        f"({policy_report.get('high_severity_violations', 0)} high-severity). "
        f"Implicated parties: {policy_report.get('implicated_parties', [])}."
    )

    user_message = f"""Produce the final liability verdict for this dispute.

Case summary: {case_summary}

Intent analysis (JSON):
{json.dumps(intent_analysis, indent=2)}

Policy violation report (JSON):
{json.dumps(policy_report, indent=2)}

Call assign_liability with the JSON strings above, then output your final verdict JSON."""

    messages: list[dict] = [{"role": "user", "content": user_message}]

    if verbose:
        print("[ArbitratorAgent] Starting liability assignment...")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[_LIABILITY_TOOL],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    if verbose:
                        print(f"[ArbitratorAgent] Called tool: {block.name}")
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
                            raise ValueError(f"ArbitratorAgent: cannot parse JSON from:\n{text}")

                    split_data = data.get("liability_split", {})
                    split = LiabilitySplit(
                        consumer_pct=float(split_data.get("consumer_pct", 0)),
                        agent_platform_pct=float(split_data.get("agent_platform_pct", 0)),
                        merchant_pct=float(split_data.get("merchant_pct", 0)),
                    )
                    verdict = Verdict(
                        case_id=case.case_id,
                        primary_liable_party=data.get("primary_liable_party", "unknown"),
                        liability_split=split,
                        explanation=data.get("explanation", ""),
                        recommended_resolution=data.get("recommended_resolution", ""),
                        confidence=float(data.get("confidence", 0.0)),
                    )
                    if verbose:
                        print(f"[ArbitratorAgent] Done. Primary liable: {verdict.primary_liable_party}, "
                              f"confidence: {verdict.confidence:.2f}")
                    return verdict

            raise ValueError("ArbitratorAgent produced no text output")

        raise RuntimeError(f"ArbitratorAgent unexpected stop_reason: {response.stop_reason}")
