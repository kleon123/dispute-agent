import uuid
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coordinator import resolve_dispute
from policies import get_policy
from schemas import (
    AgentLog,
    DisputeCase,
    OrderRecord,
    UserInstruction,
)

app = FastAPI(title="Dispute Resolution Agent")


class DisputeRequest(BaseModel):
    user_instruction: str
    budget_limit: float
    agent_action: str
    item_purchased: str
    purchase_amount: float
    merchant_name: str
    item_delivered: str
    price_charged: float
    policy_preset: str = "balanced"  # "lenient" | "balanced" | "strict"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/dispute")
def dispute(req: DisputeRequest):
    try:
        policy = get_policy(req.policy_preset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    case = DisputeCase(
        case_id=str(uuid.uuid4()),
        user_instruction=UserInstruction(
            raw_text=req.user_instruction,
            stated_constraints=[f"budget limit: ${req.budget_limit:.2f}"],
        ),
        agent_log=AgentLog(
            steps=[req.agent_action],
            reasoning="Submitted via API",
            final_action=req.agent_action,
        ),
        order_record=OrderRecord(
            order_id=str(uuid.uuid4()),
            item_name=req.item_purchased,
            item_received=req.item_delivered,
            price_charged=req.price_charged,
            merchant_name=req.merchant_name,
            delivery_sla="standard",
            delivery_actual=req.item_delivered,
        ),
        policy=policy,
    )

    verdict = resolve_dispute(case, verbose=False)
    return asdict(verdict)
