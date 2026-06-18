from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


class QueryRequest(BaseModel):
    query: str = Field(..., description="The support question or issue description")
    advertiser_id: Optional[str] = Field(None, alias="advertiserId")
    campaign_id: Optional[str] = Field(None, alias="campaignId")
    segment_id: Optional[str] = Field(None, alias="segmentId")
    deal_id: Optional[str] = Field(None, alias="dealId")
    pixel_id: Optional[str] = Field(None, alias="pixelId")
    line_item_id: Optional[str] = Field(None, alias="lineItemId")
    account_tier: Optional[str] = Field(None, alias="accountTier")

    model_config = {"populate_by_name": True}


class TaskOutput(BaseModel):
    agent: str
    role: str
    output: str


class QueryResponse(BaseModel):
    ticket_id: Optional[str] = Field(None, alias="ticketId")
    use_case: Optional[str] = Field(None, alias="useCase")
    use_case_label: Optional[str] = Field(None, alias="useCaseLabel")
    agents_used: list[str] = Field(default_factory=list, alias="agentsUsed")
    task_outputs: list[TaskOutput] = Field(default_factory=list, alias="taskOutputs")
    answer: Optional[str] = None
    status: str = "RESOLVED_BY_CREW"

    model_config = {"populate_by_name": True}


class TicketResponse(BaseModel):
    id: str
    query: str
    advertiser_id: Optional[str] = Field(None, alias="advertiserId")
    campaign_id: Optional[str] = Field(None, alias="campaignId")
    use_case: Optional[str] = Field(None, alias="useCase")
    agents_used: list[str] = Field(default_factory=list, alias="agentsUsed")
    answer: str
    status: str
    created_at: datetime = Field(..., alias="createdAt")

    model_config = {"populate_by_name": True}

    @field_serializer("created_at")
    def serialize_dt(self, v: datetime) -> str:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class HealthResponse(BaseModel):
    status: str
    service: str
    llm_provider: str = Field(..., alias="llmProvider")
    kb_provider: str = Field(..., alias="kbProvider")
    mcp_tools_loaded: int = Field(..., alias="mcpToolsLoaded")
    mcp_server_url: str = Field(..., alias="mcpServerUrl")
    crews_available: list[str] = Field(..., alias="crewsAvailable")

    model_config = {"populate_by_name": True}
