
import time

from src.api.schemas import TicketRequest
from src.config.logger import get_logger
from src.services.predict_ticket import predict_ticket_trained

from src.services.incident_store import (
    reserve_incident,
    save_incident_result,
    mark_incident_failed
)

logger = get_logger("ticket_service.py")

RESOLVER_THRESHOLD = 0.7000
PRIORITY_THRESHOLD = 0.7000


def finalize_response(result: dict, start_time: float) -> dict:
    """
    Add service processing time to the response.
    """
    elapsed_seconds = time.perf_counter() - start_time

    result["processing_time_seconds"] = round(elapsed_seconds, 6)
    result["processing_time_ms"] = round(elapsed_seconds * 1000, 2)

    logger.info(
        "Ticket %s processed in %.2f ms",
        result.get("ticket_id"),
        result["processing_time_ms"]
    )

    return result


def generate_prediction_result(ticket: TicketRequest, description: str):

    resolver_prediction = predict_ticket_trained(description)

    resolver_confidence = resolver_prediction["resolver_confidence"]
    priority_confidence = resolver_prediction["priority_confidence"]

    resolver_requires_review = (
        resolver_confidence < RESOLVER_THRESHOLD
    )

    priority_requires_review = (
        priority_confidence < PRIORITY_THRESHOLD
    )

    requires_human_review = (
        resolver_requires_review or priority_requires_review
    )

    return {
        "ticket_id": ticket.ticket_id,
        "description": description,
        "resolver_prediction": resolver_prediction,
        "requires_human_review": requires_human_review,
        "review_reasons": [
            reason
            for condition, reason in [
                (resolver_requires_review, "Low resolver confidence"),
                (priority_requires_review, "Low priority confidence")
            ]
            if condition
        ],
        "status": (
            "Pending Human Review"
            if requires_human_review else "Processed"
        )
    }


def process_ticket(ticket: TicketRequest):

    start_time = time.perf_counter()
    incident_id = ticket.incident_id

    try:
        description = ticket.description.strip()

        if not description:
            raise ValueError("Ticket description cannot be empty.")

        # ------------------------------------------------
        # CASE 1: Standalone ticket
        # ------------------------------------------------

        if not incident_id:

            result = generate_prediction_result(ticket, description)

            result["triage_reused"] = False

            return finalize_response(result, start_time)

        # ------------------------------------------------
        # CASE 2: Reserve incident
        # ------------------------------------------------

        incident_status, existing_result = reserve_incident(incident_id)

        # ------------------------------------------------
        # CASE 3: Incident already completed
        # ------------------------------------------------

        if incident_status == "COMPLETED":

            logger.info(
                "Reusing existing triage for incident %s",
                incident_id
            )

            result = {
                **existing_result,
                "ticket_id": ticket.ticket_id,
                "description": description,
                "original_ticket_id": existing_result["ticket_id"],
                "incident_id": incident_id,
                "triage_reused": True
            }

            return finalize_response(result, start_time)

        # ------------------------------------------------
        # CASE 4: Incident currently being processed
        # ------------------------------------------------

        if incident_status == "PROCESSING":

            logger.info(
                "Incident %s is already being processed",
                incident_id
            )

            result = {
                "ticket_id": ticket.ticket_id,
                "incident_id": incident_id,
                "resolver_prediction": None,
                "requires_human_review": False,
                "review_reasons": [],
                "status": "Triage In Progress",
                "triage_reused": True
            }

            return finalize_response(result, start_time)

        # ------------------------------------------------
        # CASE 5: New incident - run AI prediction
        # ------------------------------------------------

        try:

            result = generate_prediction_result(ticket, description)

            result["incident_id"] = incident_id
            result["triage_reused"] = False

            # Add timing before saving so the stored result
            # also contains the original processing time.
            finalize_response(result, start_time)

            # Save prediction for future correlated tickets.
            save_incident_result(incident_id, result)

            return result

        except Exception:

            mark_incident_failed(incident_id)
            raise

    except Exception:

        logger.exception("process_ticket failed")
        raise
