from src.api.schemas import TicketRequest
from src.config.logger import get_logger
from src.services.predict_ticket import predict_ticket_untrained,predict_ticket_trained

logger = get_logger("ticket_service.py")

def process_ticket(ticket: TicketRequest):
    try:
        description = ticket.description.strip()
        untrained_prediction = predict_ticket_untrained(description)
        resolver_prediction = predict_ticket_trained(description)

        return {
            "ticket_id": ticket.ticket_id,
            "description": description,
            "untrained_prediction" : untrained_prediction,
            "resolver_prediction" : resolver_prediction,
            "status": "Processed"
        }

    except Exception:
        logger.exception("process_ticket failed")
        raise