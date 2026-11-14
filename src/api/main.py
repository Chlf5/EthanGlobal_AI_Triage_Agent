
from fastapi import FastAPI
from src.api.schemas import TicketRequest
from src.services.ticket_service import process_ticket
from src.config.logger import get_logger

logger = get_logger("main.py")

app = FastAPI(title="Ethan Global AI Triage API")


@app.post("/tickets")
def insert_ticket(ticket: TicketRequest):
    try:
        logger.info(f"start insert_ticket")
        
        new_ticket = process_ticket(ticket)

        logger.info(f"end insert_ticket")

        return {
            "message": "Ticket created successfully",
            "ticket": new_ticket
        }
    except Exception as ex:
        logger.error(f"insert_ticket {ex}")

