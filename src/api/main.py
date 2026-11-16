
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import TicketRequest
from src.services.ticket_service import process_ticket
from src.services.itsm_integration import send_to_itsm
from src.config.logger import get_logger

logger = get_logger("main.py")

app = FastAPI(title="Ethan Global AI Triage API")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "application": "Ethan Global AI Triage API",
        "status": "running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# TICKET TRIAGE
# ============================================================

@app.post("/triage")
async def insert_ticket(ticket: TicketRequest):

    try:
        logger.info(
            "Starting ticket triage: %s",
            ticket.ticket_id
        )

        # ----------------------------------------
        # 1. Run AI triage
        # ----------------------------------------

        new_ticket = process_ticket(ticket)

        # ----------------------------------------
        # 2. Send result to ITSM
        # ----------------------------------------

        if new_ticket.get("triage_reused", False):

            itsm_result = {
                "status": "Skipped",
                "reason": "Existing incident triage was reused."
            }

        elif new_ticket.get("status") == "Triage In Progress":

            itsm_result = {
                "status": "Skipped",
                "reason": "Incident is currently being processed."
            }

        else:

            # IMPORTANT: send_to_itsm is asynchronous.
            itsm_result = await send_to_itsm(new_ticket)

        # ----------------------------------------
        # 3. Log integration result
        # ----------------------------------------

        logger.info(
            "ITSM integration status for ticket %s: %s",
            ticket.ticket_id,
            itsm_result.get("status")
        )

        logger.info(
            "Ticket triage completed: %s",
            ticket.ticket_id
        )

        # ----------------------------------------
        # 4. Return results
        # ----------------------------------------

        return {
            "message": "Ticket triage completed",
            "ticket": new_ticket,
            "itsm": itsm_result
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Ticket triage failed: %s",
            ticket.ticket_id
        )

        raise HTTPException(
            status_code=500,
            detail="Ticket triage failed."
        )
