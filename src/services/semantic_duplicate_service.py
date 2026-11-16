import httpx
import torch

from sentence_transformers import SentenceTransformer, util
from src.config.enums import ITSM_TICKETS_URL,MODEL_NAME_SIM,SIMILARITY_THRESHOLD,TOP_K,REQUEST_TIMEOUT

from src.config.logger import get_logger

logger = get_logger("semantic_duplicate_service")

# ============================================================
# LOAD MODEL
# ============================================================

logger.info("Loading semantic similarity model: %s", MODEL_NAME_SIM)

model = SentenceTransformer(MODEL_NAME_SIM)

logger.info("Semantic similarity model loaded successfully.")


# ============================================================
# RETRIEVE REFERENCE TICKETS
# ============================================================

def get_reference_tickets():

    logger.info("Retrieving reference tickets from ITSM: %s", ITSM_TICKETS_URL)

    try:
        response = httpx.get(
            ITSM_TICKETS_URL,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        tickets = data.get("tickets", [])

        logger.info("Retrieved %d tickets from ITSM.", len(tickets))

        valid_tickets = []

        for ticket in tickets:

            description = ticket.get("description")

            if not description or not description.strip():
                logger.warning(
                    "Skipping ticket %s: empty description.",
                    ticket.get("ticket_id", "UNKNOWN")
                )
                continue

            valid_tickets.append({
                "ticket_id": str(ticket.get("ticket_id", "")),
                "incident_id": str(ticket.get("incident_id", "")),
                "description": description.strip(),
                "client_id": str(ticket.get("client_id", ""))
            })

        logger.info(
            "Valid reference tickets available for comparison: %d",
            len(valid_tickets)
        )

        return valid_tickets

    except httpx.HTTPError:

        logger.exception("Failed to retrieve reference tickets from ITSM.")

        return []


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_ticket_embeddings(tickets):

    if not tickets:
        logger.warning("No tickets available for embedding generation.")
        return None

    descriptions = [
        ticket["description"]
        for ticket in tickets
    ]

    logger.info(
        "Generating embeddings for %d reference tickets.",
        len(descriptions)
    )

    embeddings = model.encode(
        descriptions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    )

    logger.info(
        "Reference embeddings generated. Shape: %s",
        tuple(embeddings.shape)
    )

    return embeddings


# ============================================================
# SEMANTIC DUPLICATE DETECTION
# ============================================================

def find_semantic_duplicates(
    description: str,
    ticket_id: str = None,
    incident_id: str = None
):

    logger.info("=" * 60)
    logger.info("STARTING SEMANTIC DUPLICATE CHECK")
    logger.info("New ticket ID: %s", ticket_id)
    logger.info("New incident ID: %s", incident_id)
    logger.info("New ticket description: %s", description)
    logger.info("Similarity threshold: %.2f", SIMILARITY_THRESHOLD)

    if not description or not description.strip():

        logger.warning("Duplicate check stopped: empty description.")

        return {
            "possible_duplicate": False,
            "threshold": SIMILARITY_THRESHOLD,
            "matches": [],
            "message": "Ticket description is empty."
        }

    # 1. Retrieve reference tickets

    reference_tickets = get_reference_tickets()

    if not reference_tickets:

        logger.warning("No reference tickets available.")

        return {
            "possible_duplicate": False,
            "threshold": SIMILARITY_THRESHOLD,
            "matches": [],
            "message": "No reference tickets available."
        }

    # 2. Exclude current ticket and same incident

    original_count = len(reference_tickets)

    reference_tickets = [
        ticket for ticket in reference_tickets
        if ticket["ticket_id"] != str(ticket_id)
        and (
            not incident_id
            or ticket["incident_id"] != str(incident_id)
        )
    ]

    logger.info(
        "Reference tickets after exclusions: %d (removed %d)",
        len(reference_tickets),
        original_count - len(reference_tickets)
    )

    if not reference_tickets:

        logger.warning("No eligible tickets remain for comparison.")

        return {
            "possible_duplicate": False,
            "threshold": SIMILARITY_THRESHOLD,
            "matches": [],
            "message": "No other tickets available for comparison."
        }

    # 3. Generate embeddings

    ticket_embeddings = generate_ticket_embeddings(reference_tickets)

    new_embedding = model.encode(
        description.strip(),
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    logger.info("New ticket embedding generated.")

    # 4. Calculate cosine similarity

    similarities = util.cos_sim(
        new_embedding,
        ticket_embeddings
    )[0]

    logger.info("Cosine similarity calculation completed.")

    # 5. Display every comparison

    for ticket, score in zip(reference_tickets, similarities.tolist()):

        logger.info(
            "Compared with ticket=%s | incident=%s | similarity=%.4f | description=%s",
            ticket["ticket_id"],
            ticket["incident_id"],
            score,
            ticket["description"]
        )

    # 6. Retrieve top matches

    top_results = torch.topk(
        similarities,
        k=min(TOP_K, len(reference_tickets))
    )

    matches = []

    for score, index in zip(
        top_results.values.tolist(),
        top_results.indices.tolist()
    ):

        similarity = float(score)

        ticket = reference_tickets[index]

        if similarity >= SIMILARITY_THRESHOLD:

            logger.warning(
                "POSSIBLE DUPLICATE DETECTED | ticket=%s | similarity=%.4f",
                ticket["ticket_id"],
                similarity
            )

            matches.append({
                "ticket_id": ticket["ticket_id"],
                "incident_id": ticket["incident_id"],
                "description": ticket["description"],
                "similarity": round(similarity, 4)
            })

        else:

            logger.info(
                "Not a duplicate | ticket=%s | similarity=%.4f | threshold=%.2f",
                ticket["ticket_id"],
                similarity,
                SIMILARITY_THRESHOLD
            )

    # 7. Final result

    possible_duplicate = len(matches) > 0

    logger.info(
        "SEMANTIC CHECK COMPLETED | possible_duplicate=%s | matches=%d",
        possible_duplicate,
        len(matches)
    )

    logger.info("=" * 60)

    return {
        "message": "Duplicate detected. Previous AI result reused.",
        "ticket": existing_ticket,
        "semantic_duplicate_check": duplicate_result,
        "itsm": {
            "status": "Skipped",
            "reason": "Previous processed ticket reused."
        }
    
    }