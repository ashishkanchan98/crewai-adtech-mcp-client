import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    HealthResponse, QueryRequest, QueryResponse, TaskOutput, TicketResponse,
)
from app.config import settings
from app.db.database import get_db
from app.db.models import CrewTicket

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_query_context(req: QueryRequest) -> dict:
    return {
        "query":          req.query,
        "advertiser_id":  req.advertiser_id or "",
        "campaign_id":    req.campaign_id or "",
        "segment_id":     req.segment_id or "",
        "deal_id":        req.deal_id or "",
        "pixel_id":       req.pixel_id or "",
        "line_item_id":   req.line_item_id or "",
        "account_tier":   req.account_tier or "",
    }


def _determine_status(answer: str) -> str:
    keywords = ["escalat", "human review", "cannot resolve", "unable to determine",
                "PAUSE_NOW", "requires immediate"]
    return "ESCALATED" if any(k.lower() in answer.lower() for k in keywords) else "RESOLVED_BY_CREW"


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())


@router.get("/api/v1/crew/health-check", response_model=HealthResponse)
async def health_check(request: Request):
    crew_router = getattr(request.app.state, "crew_router", None)
    mcp_count = getattr(request.app.state, "mcp_tools_count", 0)
    return HealthResponse(
        status="UP",
        service="crewai-adtech-mcp-client",
        llm_provider=settings.llm_provider,
        kb_provider=settings.kb_provider,
        mcp_tools_loaded=mcp_count,
        mcp_server_url=settings.mcp_server_url,
        crews_available=crew_router.available_crews if crew_router else [],
    )


@router.post("/api/v1/crew/query", response_model=QueryResponse)
async def submit_query(
    body: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    crew_router = getattr(request.app.state, "crew_router", None)
    if crew_router is None:
        raise HTTPException(status_code=503, detail="CrewAI router not initialized")

    query_ctx = _build_query_context(body)
    logger.info(
        "Processing query: campaign=%s advertiser=%s",
        body.campaign_id, body.advertiser_id,
    )

    try:
        crew_result = await crew_router.run(body.query, query_ctx)
    except Exception as exc:
        logger.error("Crew execution failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crew execution error: {exc}")

    status = _determine_status(crew_result.final_answer)
    ticket_id = str(uuid.uuid4())

    ticket = CrewTicket(
        id=ticket_id,
        query=body.query,
        advertiser_id=body.advertiser_id,
        campaign_id=body.campaign_id,
        use_case=crew_result.use_case,
        agents_used=",".join(crew_result.agents_used),
        answer=crew_result.final_answer,
        status=status,
        created_at=datetime.utcnow(),
    )
    db.add(ticket)
    await db.commit()

    logger.info(
        "Ticket %s saved  use_case=%s  status=%s  agents=%s",
        ticket_id, crew_result.use_case, status, crew_result.agents_used,
    )

    return QueryResponse(
        ticket_id=ticket_id,
        use_case=crew_result.use_case,
        use_case_label=crew_result.use_case_label,
        agents_used=crew_result.agents_used,
        task_outputs=[
            TaskOutput(agent=t["agent"], role=t["role"], output=t["output"])
            for t in crew_result.task_outputs
        ],
        answer=crew_result.final_answer,
        status=status,
    )


@router.get("/api/v1/crew/tickets", response_model=list[TicketResponse])
async def list_tickets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(CrewTicket).order_by(CrewTicket.created_at.desc()).limit(100)
    )
    tickets = result.scalars().all()
    return [
        TicketResponse(
            id=t.id,
            query=t.query,
            advertiser_id=t.advertiser_id,
            campaign_id=t.campaign_id,
            use_case=t.use_case,
            agents_used=t.agents_used.split(",") if t.agents_used else [],
            answer=t.answer,
            status=t.status,
            created_at=t.created_at,
        )
        for t in tickets
    ]


@router.get("/api/v1/crew/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CrewTicket).where(CrewTicket.id == ticket_id))
    ticket = result.scalar_one_or_none()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return TicketResponse(
        id=ticket.id,
        query=ticket.query,
        advertiser_id=ticket.advertiser_id,
        campaign_id=ticket.campaign_id,
        use_case=ticket.use_case,
        agents_used=ticket.agents_used.split(",") if ticket.agents_used else [],
        answer=ticket.answer,
        status=ticket.status,
        created_at=ticket.created_at,
    )
