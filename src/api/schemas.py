from pydantic import BaseModel, Field


# ============================================================
# INCOMING TICKET REQUEST
# ============================================================

class TicketRequest(BaseModel):
    ticket_id: str = Field(min_length=1)
    description: str = Field(min_length=5)
    incident_id: str | None = None


# ============================================================
# TRIAGE RESPONSE
# ============================================================

class TriageResponse(BaseModel):
    ticket_id: str

    # Incident information
    incident_id: str | None = None

    # Resolver group prediction
    resolver_group: str
    resolver_confidence: float = Field(ge=0.0, le=1.0)

    # Priority prediction
    priority: str = Field(pattern=r"^P[1-4]$")
    priority_confidence: float = Field(ge=0.0, le=1.0)

    # Human review decision
    requires_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)

    # Workflow status
    status: str = "Pending"

    # Incident correlation
    triage_reused: bool = False
    original_ticket_id: str | None = None

    # Performance measurements
    processing_time_seconds: float | None = None
    processing_time_ms: float | None = None