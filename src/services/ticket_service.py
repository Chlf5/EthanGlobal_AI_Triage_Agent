
import time

from src.api.schemas import TicketRequest
from src.config.logger import get_logger
from src.config.enums import (
    TRESHOLD_RESOLVER,
    TRESHOLD_PRIORITY
)

from src.services.predict_ticket import predict_ticket_trained

from src.services.incident_store import (
    reserve_incident,
    save_incident_result,
    mark_incident_failed
)

from src.services.itsm_integration import send_to_itsm
from src.services.semantic_duplicate_service import find_semantic_duplicates


logger = get_logger("ticket_service.py")

RESOLVER_THRESHOLD = TRESHOLD_RESOLVER
PRIORITY_THRESHOLD = TRESHOLD_PRIORITY

def build_triage_response(
    message: str,
    ticket: dict,
    duplicate_result: dict,
    itsm_result: dict
):
    return {
        "message": message,
        "ticket": ticket,
        "semantic_duplicate_check": duplicate_result,
        # "itsm": itsm_result
    }

def finalize_response(result: dict, start_time: float) -> dict:

    elapsed_seconds = time.perf_counter() - start_time

    result["processing_time_seconds"] = round(elapsed_seconds, 6)
    result["processing_time_ms"] = round(elapsed_seconds * 1000, 2)

    logger.info(
        "Ticket %s processed in %.2f ms",
        result.get("ticket_id"),
        result["processing_time_ms"]
    )

    return result

def generate_prediction_result(
    ticket: TicketRequest,
    description: str
):

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

    result = {
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
            if requires_human_review
            else "Processed"
        )
    }

    return result


# ============================================================
# PROCESS TICKET
# ============================================================

def process_ticket(ticket: TicketRequest):

    start_time = time.perf_counter()
    incident_id = ticket.incident_id

    try:

        description = ticket.description.strip()

        if not description:
            raise ValueError("Ticket description cannot be empty.")

        if not incident_id:

            result = generate_prediction_result(ticket,description)
            result["triage_reused"] = False

        else:

            incident_status, existing_result = reserve_incident(incident_id)
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

            elif incident_status == "PROCESSING":

                logger.info(
                    "Incident %s is already being processed",
                    incident_id
                )

                result = {
                    "ticket_id": ticket.ticket_id,
                    "description": description,
                    "incident_id": incident_id,
                    "resolver_prediction": None,
                    "requires_human_review": False,
                    "review_reasons": [],
                    "status": "Triage In Progress",
                    "triage_reused": True
                }

            else:

                try:

                    result = generate_prediction_result(
                        ticket,
                        description
                    )

                    result["incident_id"] = incident_id
                    result["triage_reused"] = False

                except Exception:

                    mark_incident_failed(incident_id)
                    raise

        result = finalize_response(result, start_time)

        # Save newly generated incident prediction
        if incident_id and not result.get("triage_reused"):

            save_incident_result(
                incident_id,
                result
            )

    except Exception:

        logger.exception("process_ticket failed")
        raise

    return result


async def handle_ticket_triage(ticket: TicketRequest):

    logger.info("Starting ticket triage | ticket_id=%s",ticket.ticket_id)

    # Default values for the unified response
    message = "Ticket triage completed"
    ticket_result = None
    itsm_result = {"status": "Skipped","reason": "Ticket was not sent to ITSM."}


    duplicate_result = find_semantic_duplicates(
        description=ticket.description,
        ticket_id=ticket.ticket_id,
        incident_id=getattr(ticket, "incident_id", None)
    )

    if duplicate_result.get("possible_duplicate"):

        logger.warning(
            "Semantic duplicate detected | ticket_id=%s",
            ticket.ticket_id
        )

        message = "Duplicate ticket detected"

        ticket_result = ticket.model_dump()

        itsm_result = {
            "status": "Skipped",
            "reason": "Duplicate ticket detected. AI triage was not executed."
        }

    else:

        ticket_result = process_ticket(ticket)

        # Add duplicate-check information to ticket result
        ticket_result["semantic_duplicate_check"] = duplicate_result

        if ticket_result.get("triage_reused"):

            itsm_result = {"status": "Skipped","reason": "Existing incident triage reused."}

        elif ticket_result.get("status") == "Triage In Progress":

            itsm_result = {"status": "Skipped","reason": "Incident triage is already in progress."}

        else:

            itsm_result = await send_to_itsm(ticket_result)

    # --------------------------------------------------------
    # STEP 6: SINGLE JSON RESPONSE CONSTRUCTION
    # --------------------------------------------------------

    response = build_triage_response(
        message=message,
        ticket=ticket_result,
        duplicate_result=duplicate_result,
        itsm_result=itsm_result
    )

    logger.info("Ticket triage completed | ticket_id=%s | message=%s",ticket.ticket_id,message)

    return response
