
import json
import sqlite3

from src.config.paths import PROCESSED_DATA_DIR

DB_PATH = PROCESSED_DATA_DIR / "incident_triage.db"


def initialize_database():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incident_triage (
                incident_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result_json TEXT
            )
        """)


def reserve_incident(incident_id: str):
    """
    Reserve an incident before running the model.
    Only one request can reserve a given incident.
    """
    initialize_database()

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("BEGIN IMMEDIATE")

        cursor = conn.execute("""
            INSERT OR IGNORE INTO incident_triage
                (incident_id, status, result_json)
            VALUES (?, 'PROCESSING', NULL)
        """, (incident_id,))

        if cursor.rowcount == 1:
            return "RESERVED", None

        row = conn.execute("""
            SELECT status, result_json
            FROM incident_triage
            WHERE incident_id = ?
        """, (incident_id,)).fetchone()

        status, result_json = row

        # Allow retry if the previous prediction failed.
        if status == "FAILED":
            conn.execute("""
                UPDATE incident_triage
                SET status = 'PROCESSING', result_json = NULL
                WHERE incident_id = ?
            """, (incident_id,))

            return "RESERVED", None

        result = json.loads(result_json) if result_json else None

        return status, result


def save_incident_result(incident_id: str, result: dict):
    initialize_database()

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            UPDATE incident_triage
            SET status = 'COMPLETED', result_json = ?
            WHERE incident_id = ?
        """, (json.dumps(result), incident_id))


def mark_incident_failed(incident_id: str):
    initialize_database()

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            UPDATE incident_triage
            SET status = 'FAILED'
            WHERE incident_id = ?
        """, (incident_id,))
