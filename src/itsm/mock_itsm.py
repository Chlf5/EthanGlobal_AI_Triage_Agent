
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(title="Ethan Global - Mock ITSM")

AI_API_URL = "http://127.0.0.1:8000/triage"

CONFIDENCE_THRESHOLD = 0.70
MAX_UPDATES_PER_MINUTE = 10
AI_TIMEOUT_SECONDS = 30


# ============================================================
# IN-MEMORY DATABASE
# ============================================================

# All tickets submitted to the ITSM system.
TICKETS = {}

# Incident-level assignments.
INCIDENTS = {}

# Audit history.
AUDIT_LOG = []


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class AITriageRequest(BaseModel):
    ticket_id: str
    description: str = Field(min_length=5)
    incident_id: Optional[str] = None


class AIIncidentResult(BaseModel):
    ticket_id: str
    incident_id: str

    resolver_group: str
    priority: str = Field(pattern=r"^P[1-4]$")

    resolver_confidence: float = Field(ge=0, le=1)
    priority_confidence: float = Field(ge=0, le=1)

    review_reasons: list[str] = Field(default_factory=list)


class IncidentUpdate(BaseModel):
    incident_id: str
    ticket_id: Optional[str] = None

    resolver_group: str
    priority: str = Field(pattern=r"^P[1-4]$")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def current_time():
    return datetime.now(timezone.utc)


def clean_audit_log():
    cutoff = current_time() - timedelta(minutes=1)

    AUDIT_LOG[:] = [
        entry for entry in AUDIT_LOG
        if datetime.fromisoformat(entry["timestamp"]) > cutoff
    ]


def check_update_limit():
    clean_audit_log()

    if len(AUDIT_LOG) >= MAX_UPDATES_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Maximum of 10 manual updates per minute reached."
        )


def needs_human_validation(resolver_confidence, priority_confidence):
    return (
        resolver_confidence < CONFIDENCE_THRESHOLD
        or priority_confidence < CONFIDENCE_THRESHOLD
    )


def get_review_reasons(resolver_confidence, priority_confidence):
    reasons = []

    if resolver_confidence < CONFIDENCE_THRESHOLD:
        reasons.append("Low resolver group confidence")

    if priority_confidence < CONFIDENCE_THRESHOLD:
        reasons.append("Low priority confidence")

    return reasons


def get_ticket_category(ticket):
    if ticket.get("requires_human_review"):
        return "Needs Human Validation"

    return "Automatically Assigned"


def save_ticket(ticket):
    TICKETS[ticket["ticket_id"]] = ticket


def normalize_ai_ticket(data):
    """
    Supports the AI API response containing either:
    - resolver_group / priority
    - resolver_prediction / priority_prediction
    """

    ticket = data.get("ticket", data)

    if not isinstance(ticket, dict):
        return None

    resolver = ticket.get("resolver_group")

    if resolver is None:
        resolver = ticket.get("resolver_prediction")

    if isinstance(resolver, dict):
        resolver = resolver.get("label") or resolver.get("prediction")

    priority = ticket.get("priority")

    if priority is None:
        priority = ticket.get("priority_prediction")

    if isinstance(priority, dict):
        priority = priority.get("label") or priority.get("prediction")

    resolver_confidence = ticket.get("resolver_confidence")

    if resolver_confidence is None:
        resolver_confidence = ticket.get("confidence", 0)

    priority_confidence = ticket.get("priority_confidence", 0)

    try:
        resolver_confidence = float(resolver_confidence)
        priority_confidence = float(priority_confidence)
    except (TypeError, ValueError):
        return None

    if not resolver or priority not in {"P1", "P2", "P3", "P4"}:
        return None

    ticket_id = ticket.get("ticket_id")

    if not ticket_id:
        return None

    incident_id = (
        ticket.get("incident_id")
        or ticket_id
    )

    requires_review = needs_human_validation(
        resolver_confidence,
        priority_confidence
    )

    return {
        "ticket_id": ticket_id,
        "incident_id": incident_id,
        "description": ticket.get("description", ""),
        "resolver_group": resolver,
        "priority": priority,
        "resolver_confidence": resolver_confidence,
        "priority_confidence": priority_confidence,
        "requires_human_review": requires_review,
        "review_reasons": get_review_reasons(
            resolver_confidence,
            priority_confidence
        ),
        "status": (
            "Pending Human Review"
            if requires_review
            else "Assigned"
        ),
        "triage_reused": ticket.get("triage_reused", False),
        "created_at": ticket.get(
            "created_at",
            current_time().isoformat()
        ),
        "last_updated": current_time().isoformat()
    }


def store_ai_result(result: AIIncidentResult):
    """
    Save the AI recommendation.

    High confidence:
        Automatically assign the incident.

    Low confidence:
        Store recommendation and wait for human validation.
    """

    needs_review = needs_human_validation(
        result.resolver_confidence,
        result.priority_confidence
    )

    reasons = get_review_reasons(
        result.resolver_confidence,
        result.priority_confidence
    )

    now = current_time().isoformat()

    existing = INCIDENTS.get(result.incident_id, {})

    if needs_review:

        incident = {
            **existing,
            "incident_id": result.incident_id,
            "ticket_id": result.ticket_id,

            # Keep existing actual assignment, if one exists.
            "assignment_group": existing.get("assignment_group"),
            "priority": existing.get("priority"),

            # Store AI recommendations separately.
            "recommended_assignment_group": result.resolver_group,
            "recommended_priority": result.priority,

            "resolver_confidence": result.resolver_confidence,
            "priority_confidence": result.priority_confidence,

            "status": "Pending Human Review",
            "review_reasons": reasons,
            "last_updated": now
        }

        INCIDENTS[result.incident_id] = incident

        return {
            "status": "Success",
            "automatic_assignment": False,
            "message": "AI recommendation saved. Human validation required.",
            "incident": incident
        }

    incident = {
        **existing,
        "incident_id": result.incident_id,
        "ticket_id": result.ticket_id,

        "assignment_group": result.resolver_group,
        "priority": result.priority,

        "recommended_assignment_group": result.resolver_group,
        "recommended_priority": result.priority,

        "resolver_confidence": result.resolver_confidence,
        "priority_confidence": result.priority_confidence,

        "status": "Assigned",
        "review_reasons": [],
        "last_updated": now
    }

    INCIDENTS[result.incident_id] = incident

    AUDIT_LOG.append({
        "timestamp": now,
        "incident_id": result.incident_id,
        "ticket_id": result.ticket_id,
        "action": "Automatic AI Assignment",
        "assignment_group": result.resolver_group,
        "priority": result.priority,
        "performed_by": "AI System"
    })

    return {
        "status": "Success",
        "automatic_assignment": True,
        "message": "High-confidence AI assignment completed.",
        "incident": incident
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():

    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Ethan Global - Mock ITSM</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f3f5f9;
            margin: 0;
            padding: 25px;
            color: #222;
        }

        h1, h2 {
            margin-top: 0;
        }

        .container {
            max-width: 1400px;
            margin: auto;
        }

        .section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px #00000010;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat {
            background: white;
            padding: 20px;
            border-radius: 10px;
        }

        .stat strong {
            display: block;
            font-size: 27px;
            margin-top: 8px;
        }

        input, textarea, select {
            width: 100%;
            padding: 10px;
            margin: 7px 0 14px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }

        button {
            padding: 10px 16px;
            border: 0;
            border-radius: 5px;
            cursor: pointer;
            background: #2457c5;
            color: white;
            margin: 3px;
        }

        button:hover {
            opacity: .85;
        }

        button.secondary {
            background: #555;
        }

        button.success {
            background: #16834a;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 850px;
        }

        th, td {
            text-align: left;
            padding: 11px;
            border-bottom: 1px solid #eee;
        }

        th {
            background: #f5f6f8;
        }

        .table-container {
            overflow-x: auto;
        }

        .badge {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
        }

        .review {
            background: #fff0d5;
            color: #8a5200;
        }

        .assigned {
            background: #d9f5e5;
            color: #12653b;
        }

        .message {
            padding: 12px;
            margin-top: 12px;
            border-radius: 5px;
            display: none;
        }

        .error {
            background: #ffe0e0;
            color: #9b1515;
        }

        .info {
            background: #e6efff;
            color: #174b9c;
        }

        .tabs {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }

        .tabs button.active {
            background: #142d63;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        @media(max-width: 750px) {
            .stats {
                grid-template-columns: 1fr 1fr;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <h1>Ethan Global - Mock ITSM</h1>
    <p>AI ticket triage and human validation dashboard</p>

    <div class="stats">
        <div class="stat">
            Total Tickets
            <strong id="totalTickets">0</strong>
        </div>

        <div class="stat">
            Needs Human Validation
            <strong id="reviewCount">0</strong>
        </div>

        <div class="stat">
            Automatically Assigned
            <strong id="assignedCount">0</strong>
        </div>

        <div class="stat">
            Total Incidents
            <strong id="incidentCount">0</strong>
        </div>
    </div>


    <!-- SUBMIT TICKET -->

    <div class="section">
        <h2>Submit Ticket for AI Triage</h2>

        <form id="triageForm">

            <div class="form-grid">
                <div>
                    <label>Ticket ID</label>
                    <input id="ticketId" required
                           placeholder="INC1001">
                </div>

                <div>
                    <label>Incident ID</label>
                    <input id="incidentId"
                           placeholder="GROUP100">
                </div>
            </div>

            <label>Ticket Description</label>
            <textarea id="description" rows="4"
                      minlength="5" required
                      placeholder="Describe the IT issue..."></textarea>

            <button type="submit">Run AI Triage</button>
        </form>

        <div id="triageMessage" class="message"></div>
    </div>


    <!-- TICKET CATEGORIES -->

    <div class="section">
        <h2>All Tickets</h2>

        <div class="tabs">
            <button class="active" onclick="loadTickets('all', this)">
                All Tickets
            </button>

            <button onclick="loadTickets('review', this)">
                Needs Human Validation
            </button>

            <button onclick="loadTickets('assigned', this)">
                Automatically Assigned
            </button>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Ticket ID</th>
                        <th>Incident ID</th>
                        <th>Resolver Group</th>
                        <th>Resolver Confidence</th>
                        <th>Priority</th>
                        <th>Priority Confidence</th>
                        <th>Status</th>
                        <th>Action</th>
                    </tr>
                </thead>

                <tbody id="ticketTable"></tbody>
            </table>
        </div>
    </div>


    <!-- HUMAN VALIDATION -->

    <div class="section" id="manualSection" style="display:none">
        <h2>Human Validation and Assignment</h2>

        <p id="reviewInformation"></p>

        <form id="manualForm">

            <label>Ticket ID</label>
            <input id="manualTicketId" readonly>

            <label>Incident ID</label>
            <input id="manualIncidentId" readonly>

            <label>Resolver Group</label>
            <input id="manualResolverGroup" required>

            <label>Priority</label>
            <select id="manualPriority">
                <option value="P1">P1 - Critical</option>
                <option value="P2">P2 - High</option>
                <option value="P3">P3 - Medium</option>
                <option value="P4">P4 - Low</option>
            </select>

            <button type="submit" class="success">
                Confirm and Apply Assignment
            </button>

            <button type="button" class="secondary"
                    onclick="closeManualForm()">
                Cancel
            </button>
        </form>

        <div id="manualMessage" class="message"></div>
    </div>


    <!-- INCIDENTS -->

    <div class="section">
        <h2>Incident Records</h2>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Incident ID</th>
                        <th>Assignment Group</th>
                        <th>Priority</th>
                        <th>AI Recommendation</th>
                        <th>Status</th>
                        <th>Last Updated</th>
                    </tr>
                </thead>

                <tbody id="incidentTable"></tbody>
            </table>
        </div>
    </div>


    <!-- AUDIT -->

    <div class="section">
        <h2>Audit Log</h2>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Ticket</th>
                        <th>Incident</th>
                        <th>Action</th>
                        <th>Assignment Group</th>
                        <th>Priority</th>
                        <th>Performed By</th>
                    </tr>
                </thead>

                <tbody id="auditTable"></tbody>
            </table>
        </div>
    </div>

</div>


<script>

let currentCategory = "all";


function showMessage(elementId, message, type = "info") {
    const element = document.getElementById(elementId);

    element.style.display = "block";
    element.className = "message " + type;
    element.textContent = message;
}


function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[char]);
}


async function api(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        },
        ...options
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            JSON.stringify(data.detail ?? data)
        );
    }

    return data;
}


// ============================================================
// LOAD STATISTICS
// ============================================================

async function loadStats() {
    try {
        const data = await api("/stats");

        document.getElementById("totalTickets").textContent =
            data.total_tickets;

        document.getElementById("reviewCount").textContent =
            data.needs_human_validation;

        document.getElementById("assignedCount").textContent =
            data.automatically_assigned;

        document.getElementById("incidentCount").textContent =
            data.total_incidents;

    } catch (error) {
        console.error(error);
    }
}


// ============================================================
// LOAD TICKETS BY CATEGORY
// ============================================================

async function loadTickets(category = currentCategory, button = null) {
    currentCategory = category;

    if (button) {
        document.querySelectorAll(".tabs button").forEach(item => {
            item.classList.remove("active");
        });

        button.classList.add("active");
    }

    try {
        const data = await api("/tickets");

        let tickets = data.tickets || [];

        if (category === "review") {
            tickets = tickets.filter(
                ticket => ticket.requires_human_review
            );
        }

        if (category === "assigned") {
            tickets = tickets.filter(
                ticket => !ticket.requires_human_review
            );
        }

        const tbody = document.getElementById("ticketTable");

        tbody.innerHTML = "";

        if (tickets.length === 0) {
            tbody.innerHTML =
                "<tr><td colspan='8'>No tickets found.</td></tr>";
            return;
        }

        tickets.forEach(ticket => {
            const review = ticket.requires_human_review;

            const badge = review
                ? '<span class="badge review">Needs Human Validation</span>'
                : '<span class="badge assigned">Automatically Assigned</span>';

            const action = review
                ? `<button onclick="openManualForm('${escapeHtml(ticket.ticket_id)}')">Validate</button>`
                : `<button class="secondary" onclick="viewTicket('${escapeHtml(ticket.ticket_id)}')">View</button>`;

            tbody.innerHTML += `
                <tr>
                    <td>${escapeHtml(ticket.ticket_id)}</td>
                    <td>${escapeHtml(ticket.incident_id)}</td>
                    <td>${escapeHtml(ticket.resolver_group)}</td>
                    <td>${(ticket.resolver_confidence * 100).toFixed(2)}%</td>
                    <td>${escapeHtml(ticket.priority)}</td>
                    <td>${(ticket.priority_confidence * 100).toFixed(2)}%</td>
                    <td>${badge}</td>
                    <td>${action}</td>
                </tr>
            `;
        });

    } catch (error) {
        console.error(error);
    }
}


// ============================================================
// OPEN HUMAN VALIDATION FORM
// ============================================================

async function openManualForm(ticketId) {
    try {
        const ticket = await api(
            `/tickets/${encodeURIComponent(ticketId)}`
        );

        document.getElementById("manualSection").style.display = "block";

        document.getElementById("manualTicketId").value =
            ticket.ticket_id;

        document.getElementById("manualIncidentId").value =
            ticket.incident_id;

        document.getElementById("manualResolverGroup").value =
            ticket.resolver_group;

        document.getElementById("manualPriority").value =
            ticket.priority;

        document.getElementById("reviewInformation").textContent =
            "Resolver confidence: " +
            (ticket.resolver_confidence * 100).toFixed(2) +
            "%. Priority confidence: " +
            (ticket.priority_confidence * 100).toFixed(2) +
            "%. Review reasons: " +
            (ticket.review_reasons || []).join(", ");

        document.getElementById("manualSection").scrollIntoView({
            behavior: "smooth"
        });

    } catch (error) {
        alert(error.message);
    }
}


function closeManualForm() {
    document.getElementById("manualSection").style.display = "none";
}


async function viewTicket(ticketId) {
    try {
        const ticket = await api(
            `/tickets/${encodeURIComponent(ticketId)}`
        );

        alert(
            "Ticket: " + ticket.ticket_id +
            "\\nIncident: " + ticket.incident_id +
            "\\nResolver: " + ticket.resolver_group +
            "\\nPriority: " + ticket.priority +
            "\\nResolver confidence: " +
            (ticket.resolver_confidence * 100).toFixed(2) + "%" +
            "\\nPriority confidence: " +
            (ticket.priority_confidence * 100).toFixed(2) + "%" +
            "\\nStatus: " + ticket.status
        );

    } catch (error) {
        alert(error.message);
    }
}


// ============================================================
// SUBMIT TICKET TO AI API
// ============================================================

document.getElementById("triageForm").addEventListener(
    "submit",
    async function(event) {
        event.preventDefault();

        const payload = {
            ticket_id: document.getElementById("ticketId").value.trim(),
            incident_id: document.getElementById("incidentId").value.trim() || null,
            description: document.getElementById("description").value.trim()
        };

        try {
            const data = await api("/ai-triage", {
                method: "POST",
                body: JSON.stringify(payload)
            });

            const ticket = data.ticket || {};

            if (data.itsm && data.itsm.status === "Failure") {
                showMessage(
                    "triageMessage",
                    "AI triage completed, but the ITSM callback failed: " +
                    (data.itsm.message || "Unknown error"),
                    "error"
                );
            } else {
                showMessage(
                    "triageMessage",
                    "Ticket submitted for AI triage. Refreshing ticket records.",
                    "info"
                );
            }

            await refreshDashboard();

            if (ticket.ticket_id) {
                const saved = await api(
                    `/tickets/${encodeURIComponent(ticket.ticket_id)}`
                );

                if (saved.requires_human_review) {
                    await openManualForm(saved.ticket_id);
                }
            }

        } catch (error) {
            showMessage("triageMessage", error.message, "error");
        }
    }
);


// ============================================================
// MANUAL ASSIGNMENT
// ============================================================

document.getElementById("manualForm").addEventListener(
    "submit",
    async function(event) {
        event.preventDefault();

        const incidentId =
            document.getElementById("manualIncidentId").value.trim();

        const payload = {
            incident_id: incidentId,
            ticket_id: document.getElementById("manualTicketId").value,
            resolver_group:
                document.getElementById("manualResolverGroup").value.trim(),
            priority:
                document.getElementById("manualPriority").value
        };

        if (!incidentId) {
            showMessage(
                "manualMessage",
                "A valid incident ID is required.",
                "error"
            );
            return;
        }

        try {
            const result = await api(
                `/incidents/${encodeURIComponent(incidentId)}/update`,
                {
                    method: "POST",
                    body: JSON.stringify(payload)
                }
            );

            showMessage(
                "manualMessage",
                result.message || "Manual assignment completed.",
                "info"
            );

            closeManualForm();
            await refreshDashboard();

        } catch (error) {
            showMessage("manualMessage", error.message, "error");
        }
    }
);


// ============================================================
// LOAD INCIDENTS
// ============================================================

async function loadIncidents() {
    try {
        const data = await api("/incidents");

        const tbody = document.getElementById("incidentTable");

        tbody.innerHTML = "";

        if (!data.incidents.length) {
            tbody.innerHTML =
                "<tr><td colspan='6'>No incidents found.</td></tr>";
            return;
        }

        data.incidents.forEach(item => {
            tbody.innerHTML += `
                <tr>
                    <td>${escapeHtml(item.incident_id)}</td>
                    <td>${escapeHtml(item.assignment_group || "Not assigned")}</td>
                    <td>${escapeHtml(item.priority || "Not assigned")}</td>
                    <td>
                        ${escapeHtml(item.recommended_assignment_group || "-")}
                        /
                        ${escapeHtml(item.recommended_priority || "-")}
                    </td>
                    <td>${escapeHtml(item.status)}</td>
                    <td>${escapeHtml(item.last_updated || "-")}</td>
                </tr>
            `;
        });

    } catch (error) {
        console.error(error);
    }
}


// ============================================================
// LOAD AUDIT LOG
// ============================================================

async function loadAudit() {
    try {
        const data = await api("/audit");

        const tbody = document.getElementById("auditTable");

        tbody.innerHTML = "";

        data.audit.forEach(item => {
            tbody.innerHTML += `
                <tr>
                    <td>${escapeHtml(item.timestamp)}</td>
                    <td>${escapeHtml(item.ticket_id || "-")}</td>
                    <td>${escapeHtml(item.incident_id)}</td>
                    <td>${escapeHtml(item.action)}</td>
                    <td>${escapeHtml(item.assignment_group || "-")}</td>
                    <td>${escapeHtml(item.priority || "-")}</td>
                    <td>${escapeHtml(item.performed_by)}</td>
                </tr>
            `;
        });

    } catch (error) {
        console.error(error);
    }
}


// ============================================================
// REFRESH DASHBOARD
// ============================================================

async function refreshDashboard() {
    await Promise.all([
        loadStats(),
        loadTickets(currentCategory),
        loadIncidents(),
        loadAudit()
    ]);
}


refreshDashboard();

</script>

</body>
</html>
""")


# ============================================================
# AI TRIAGE PROXY
# ============================================================

@app.post("/ai-triage")
async def ai_triage(request: AITriageRequest):

    payload = request.model_dump()

    try:
        async with httpx.AsyncClient(
            timeout=AI_TIMEOUT_SECONDS
        ) as client:

            response = await client.post(
                AI_API_URL,
                json=payload
            )

            response.raise_for_status()

            data = response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI API request timed out."
        )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to AI API: {str(exc)}"
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text
        )

    # Save the ticket returned by the AI API.
    normalized = normalize_ai_ticket(data)

    if normalized:
        existing = TICKETS.get(normalized["ticket_id"], {})

        # Do not overwrite an already completed manual assignment.
        if existing.get("status") == "Assigned - Human Validated":
            normalized.update({
                "requires_human_review": False,
                "status": "Assigned - Human Validated"
            })

        save_ticket({
            **existing,
            **normalized
        })

    return data


# ============================================================
# AI CALLBACK: AUTOMATIC ASSIGNMENT OR HUMAN REVIEW
# ============================================================

@app.post("/incidents/{incident_id}/ai-result")
async def receive_ai_result(
    incident_id: str,
    result: AIIncidentResult
):

    if incident_id != result.incident_id:
        raise HTTPException(
            status_code=400,
            detail="Incident ID in URL does not match request body."
        )

    result_data = store_ai_result(result)

    # Update ticket record.
    existing = TICKETS.get(result.ticket_id, {})

    needs_review = not result_data["automatic_assignment"]

    ticket = {
        **existing,
        "ticket_id": result.ticket_id,
        "incident_id": result.incident_id,
        "resolver_group": result.resolver_group,
        "priority": result.priority,
        "resolver_confidence": result.resolver_confidence,
        "priority_confidence": result.priority_confidence,
        "requires_human_review": needs_review,
        "review_reasons": get_review_reasons(
            result.resolver_confidence,
            result.priority_confidence
        ),
        "status": (
            "Pending Human Review"
            if needs_review
            else "Assigned"
        ),
        "last_updated": current_time().isoformat()
    }

    save_ticket(ticket)

    return result_data


# ============================================================
# MANUAL INCIDENT UPDATE
# ============================================================

@app.post("/incidents/{incident_id}/update")
async def update_incident(
    incident_id: str,
    update: IncidentUpdate
):

    if incident_id != update.incident_id:
        raise HTTPException(
            status_code=400,
            detail="Incident ID in URL does not match request body."
        )

    check_update_limit()

    incident = INCIDENTS.get(incident_id)

    if not incident:
        raise HTTPException(
            status_code=404,
            detail="Incident not found. Submit and triage the ticket first."
        )

    now = current_time().isoformat()

    incident.update({
        "assignment_group": update.resolver_group,
        "priority": update.priority,
        "status": "Assigned - Human Validated",
        "review_reasons": [],
        "last_updated": now
    })

    INCIDENTS[incident_id] = incident

    # Mark all tickets belonging to this incident as validated.
    for ticket in TICKETS.values():

        if ticket.get("incident_id") == incident_id:
            ticket["requires_human_review"] = False
            ticket["status"] = "Assigned - Human Validated"
            ticket["review_reasons"] = []
            ticket["last_updated"] = now

    AUDIT_LOG.append({
        "timestamp": now,
        "incident_id": incident_id,
        "ticket_id": update.ticket_id,
        "action": "Manual Human Validation",
        "assignment_group": update.resolver_group,
        "priority": update.priority,
        "performed_by": "Human User"
    })

    return {
        "status": "Success",
        "message": "Human validation completed. Incident assignment updated.",
        "incident": incident
    }


# ============================================================
# GET ALL TICKETS
# ============================================================

@app.get("/tickets")
async def get_tickets():

    tickets = list(TICKETS.values())

    return {
        "total": len(tickets),
        "tickets": tickets
    }


# ============================================================
# GET TICKETS NEEDING HUMAN VALIDATION
# ============================================================

@app.get("/tickets/needs-human-validation")
async def get_tickets_needing_validation():

    tickets = [
        ticket for ticket in TICKETS.values()
        if ticket.get("requires_human_review")
    ]

    return {
        "total": len(tickets),
        "tickets": tickets
    }


# ============================================================
# GET AUTOMATICALLY ASSIGNED TICKETS
# ============================================================

@app.get("/tickets/automatically-assigned")
async def get_automatically_assigned_tickets():

    tickets = [
        ticket for ticket in TICKETS.values()
        if not ticket.get("requires_human_review")
    ]

    return {
        "total": len(tickets),
        "tickets": tickets
    }


# ============================================================
# GET ONE TICKET
# ============================================================

@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):

    ticket = TICKETS.get(ticket_id)

    if not ticket:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found."
        )

    return ticket


# ============================================================
# GET ALL INCIDENTS
# ============================================================

@app.get("/incidents")
async def get_incidents():

    return {
        "total": len(INCIDENTS),
        "incidents": list(INCIDENTS.values())
    }


# ============================================================
# GET ONE INCIDENT
# ============================================================

@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):

    incident = INCIDENTS.get(incident_id)

    if not incident:
        raise HTTPException(
            status_code=404,
            detail="Incident not found."
        )

    return incident


# ============================================================
# AUDIT LOG
# ============================================================

@app.get("/audit")
async def get_audit():

    return {
        "total": len(AUDIT_LOG),
        "audit": AUDIT_LOG
    }


# ============================================================
# STATISTICS
# ============================================================

@app.get("/stats")
async def get_stats():

    tickets = list(TICKETS.values())

    needs_review = sum(
        1 for ticket in tickets
        if ticket.get("requires_human_review")
    )

    automatically_assigned = sum(
        1 for ticket in tickets
        if not ticket.get("requires_human_review")
    )

    return {
        "total_tickets": len(tickets),
        "needs_human_validation": needs_review,
        "automatically_assigned": automatically_assigned,
        "total_incidents": len(INCIDENTS),
        "total_audit_entries": len(AUDIT_LOG),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "maximum_manual_updates_per_minute": MAX_UPDATES_PER_MINUTE
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "Ethan Global Mock ITSM",
        "tickets": len(TICKETS),
        "incidents": len(INCIDENTS)
    }
