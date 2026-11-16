
import os

import httpx


# ============================================================
# CONFIGURATION
# ============================================================

ITSM_API_URL = os.getenv(
    "ITSM_API_URL",
    "http://127.0.0.1:8001"
)

ITSM_TIMEOUT = 10.0

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# SEND AI RESULT TO MOCK ITSM
# ============================================================

async def send_to_itsm(ticket: dict) -> dict:

    if not ticket:
        return {
            "status": "Skipped",
            "reason": "Empty ticket result."
        }

    if ticket.get("triage_reused"):
        return {
            "status": "Skipped",
            "reason": "Existing incident triage was reused."
        }

    ticket_id = ticket.get("ticket_id")
    incident_id = ticket.get("incident_id")

    if not ticket_id or not incident_id:
        return {
            "status": "Failed",
            "reason": "Missing ticket_id or incident_id."
        }

    # --------------------------------------------------------
    # Extract nested model predictions
    # --------------------------------------------------------

    prediction = ticket.get("resolver_prediction")

    if not isinstance(prediction, dict):
        return {
            "status": "Failed",
            "reason": "Missing or invalid resolver_prediction object."
        }

    resolver_group = prediction.get("resolver_group")
    resolver_confidence = prediction.get("resolver_confidence")

    priority = prediction.get("priority")
    priority_confidence = prediction.get("priority_confidence")

    if not resolver_group or not priority:
        return {
            "status": "Failed",
            "reason": "Model prediction is missing resolver group or priority."
        }

    try:
        resolver_confidence = float(resolver_confidence)
        priority_confidence = float(priority_confidence)
    except (TypeError, ValueError):
        return {
            "status": "Failed",
            "reason": "Invalid confidence values returned by the model."
        }

    # --------------------------------------------------------
    # Determine human validation requirement
    # --------------------------------------------------------

    requires_human_review = (
        resolver_confidence < CONFIDENCE_THRESHOLD
        or priority_confidence < CONFIDENCE_THRESHOLD
    )

    review_reasons = []

    if resolver_confidence < CONFIDENCE_THRESHOLD:
        review_reasons.append("Low resolver confidence")

    if priority_confidence < CONFIDENCE_THRESHOLD:
        review_reasons.append("Low priority confidence")

    # --------------------------------------------------------
    # Build AI result payload
    # --------------------------------------------------------

    payload = {
        "ticket_id": ticket_id,
        "incident_id": incident_id,
        "resolver_group": resolver_group,
        "priority": priority,
        "resolver_confidence": resolver_confidence,
        "priority_confidence": priority_confidence,
        "review_reasons": review_reasons
    }

    # --------------------------------------------------------
    # Send to AI callback endpoint
    # IMPORTANT: NOT /update
    # --------------------------------------------------------

    url = (
        f"{ITSM_API_URL}/incidents/"
        f"{incident_id}/ai-result"
    )

    try:
        async with httpx.AsyncClient(
            timeout=ITSM_TIMEOUT
        ) as client:

            response = await client.post(
                url,
                json=payload
            )

            response.raise_for_status()

            result = response.json()

        return {
            "status": "Success",
            "automatic_assignment": not requires_human_review,
            "message": result.get("message"),
            "response": result
        }

    except httpx.TimeoutException:
        return {
            "status": "Failed",
            "reason": "ITSM request timed out."
        }

    except httpx.HTTPStatusError as exc:
        return {
            "status": "Failed",
            "reason": exc.response.text,
            "http_status": exc.response.status_code
        }

    except httpx.RequestError as exc:
        return {
            "status": "Failed",
            "reason": str(exc)
        }
