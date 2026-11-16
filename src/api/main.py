from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import TicketRequest
from src.services.ticket_service import handle_ticket_triage
from src.config.logger import get_logger

logger = get_logger("main.py")

app = FastAPI(title="Ethan Global AI Triage API")


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


@app.get("/")
def home():
    return {
        "application": "Ethan Global AI Triage API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/triage")
async def insert_ticket(ticket: TicketRequest):

    try:
        logger.info("Starting ticket triage: %s",ticket.ticket_id)

        result = await handle_ticket_triage(ticket)

        logger.info("Ticket triage completed: %s",ticket.ticket_id)

        return result

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