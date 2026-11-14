from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    ticket_id: str
    description: str = Field(min_length=5)


class TriageResponse(BaseModel):
    ticket_id: str
    resolver_group: str
    confidence: float
    requires_human_review: bool