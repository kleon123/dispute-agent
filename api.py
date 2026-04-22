import uuid
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from coordinator import resolve_dispute
from schemas import (
    AgentLog,
    DisputeCase,
    OrderRecord,
    Policy,
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/dispute")
def dispute(req: DisputeRequest):
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
        policy=Policy(
            agent_platform_policies=[
                f"Agent must not exceed user's stated budget of ${req.budget_limit:.2f}",
                "Agent must purchase only the item the user requested",
            ],
            merchant_terms=[
                "Merchant is responsible for accurate item description and fulfillment",
            ],
            consumer_protections=[
                "Consumer is entitled to a refund if item received differs from item ordered",
                "Consumer is entitled to a refund if price charged exceeds authorised amount",
            ],
        ),
    )

    verdict = resolve_dispute(case, verbose=False)
    return asdict(verdict)
