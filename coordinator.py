"""
Coordinator — orchestrates the three-agent dispute resolution pipeline.

Pipeline: IntentAgent → PolicyAgent → ArbitratorAgent

Each agent runs independently and passes its structured output as input to the next.
The coordinator returns the final Verdict from the ArbitratorAgent.
"""

from schemas import DisputeCase, Verdict
import intent_agent
import policy_agent
import arbitrator_agent


def resolve_dispute(case: DisputeCase, verbose: bool = True) -> Verdict:
    """
    Run the full three-agent dispute resolution pipeline for one case.

    Stage 1 — IntentAgent:   parse user instruction, score ambiguity, detect budget violations
    Stage 2 — PolicyAgent:   compare transaction against platform policies, merchant terms,
                             and consumer protections
    Stage 3 — ArbitratorAgent: synthesise both analyses into a final liability verdict

    Returns a Verdict dataclass with the liability split and recommended resolution.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Dispute pipeline starting: {case.case_id}")
        print(f"{'='*60}")

    # --- Stage 1: Intent Analysis ---
    if verbose:
        print("\n[Stage 1/3] Intent Agent — parsing user intent...")
    intent_analysis = intent_agent.run(case, verbose=verbose)
    if verbose:
        print(f"  Alignment score: {intent_analysis.get('intent_alignment_score')}/100")
        print(f"  Budget violation: {intent_analysis.get('budget_violation_detected')}")
        print(f"  Agent fault signals: {intent_analysis.get('agent_fault_signals')}")

    # --- Stage 2: Policy Analysis ---
    if verbose:
        print("\n[Stage 2/3] Policy Agent — checking policy violations...")
    policy_report = policy_agent.run(case, intent_analysis, verbose=verbose)
    if verbose:
        print(f"  Violations found: {policy_report.get('total_violations')} "
              f"({policy_report.get('high_severity_violations')} high-severity)")
        print(f"  Implicated parties: {policy_report.get('implicated_parties')}")

    # --- Stage 3: Arbitration ---
    if verbose:
        print("\n[Stage 3/3] Arbitrator Agent — assigning liability...")
    verdict = arbitrator_agent.run(case, intent_analysis, policy_report, verbose=verbose)
    if verbose:
        split = verdict.liability_split
        print(f"\n{'='*60}")
        print(f"  VERDICT for {case.case_id}")
        print(f"  Primary liable party: {verdict.primary_liable_party}")
        print(f"  Split — Consumer {split.consumer_pct}% / "
              f"Platform {split.agent_platform_pct}% / "
              f"Merchant {split.merchant_pct}%")
        print(f"  Resolution: {verdict.recommended_resolution}")
        print(f"  Confidence: {verdict.confidence:.2f}")
        print(f"{'='*60}")

    return verdict
