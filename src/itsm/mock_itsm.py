
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.config.enums import TRESHOLD_PRIORITY,TRESHOLD_RESOLVER


# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(title="Ethan Global - Mock ITSM")

AI_API_URL = "http://127.0.0.1:8000/triage"

CONFIDENCE_THRESHOLD = TRESHOLD_PRIORITY
MAX_UPDATES_PER_MINUTE = 10
AI_TIMEOUT_SECONDS = 30

VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}


# ============================================================
# IN-MEMORY DATABASE
# ============================================================

TICKETS = {}
INCIDENTS = {}
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

    # Optional for compatibility with older AI callback payloads.
    # The description is preserved if the ticket already exists.
    description: Optional[str] = None

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
        entry
        for entry in AUDIT_LOG
        if datetime.fromisoformat(entry["timestamp"]) > cutoff
    ]


def check_update_limit():
    clean_audit_log()

    manual_updates = sum(
        1 for entry in AUDIT_LOG
        if entry.get("action") == "Manual Human Validation"
    )

    if manual_updates >= MAX_UPDATES_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Maximum of 10 manual updates per minute reached."
        )


def needs_human_validation(
    resolver_confidence,
    priority_confidence
):
    return (
        resolver_confidence < CONFIDENCE_THRESHOLD
        or priority_confidence < CONFIDENCE_THRESHOLD
    )


def get_review_reasons(
    resolver_confidence,
    priority_confidence
):
    reasons = []

    if resolver_confidence < CONFIDENCE_THRESHOLD:
        reasons.append("Low resolver group confidence")

    if priority_confidence < CONFIDENCE_THRESHOLD:
        reasons.append("Low priority confidence")

    return reasons


def save_ticket(ticket):
    """
    Save a ticket while preserving its description
    if a later update does not contain one.
    """

    ticket_id = ticket["ticket_id"]

    existing = TICKETS.get(ticket_id, {})

    if not ticket.get("description"):
        ticket["description"] = existing.get("description", "")

    TICKETS[ticket_id] = ticket

    return ticket


# ============================================================
# NORMALIZE AI RESPONSE
# ============================================================

def normalize_ai_ticket(data, request):

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="AI API returned an invalid response."
        )

    ticket = data.get("ticket")

    if not isinstance(ticket, dict):
        raise HTTPException(
            status_code=502,
            detail="AI API did not return a valid ticket object."
        )

    semantic_duplicate_check = data.get("semantic_duplicate_check", {})

    if not isinstance(semantic_duplicate_check, dict):
        raise HTTPException(
            status_code=502,
            detail="AI API returned an invalid semantic duplicate check."
        )

    if semantic_duplicate_check.get("possible_duplicate", False):
        raise HTTPException(
            status_code=409,
            detail="Same ticket already exists."
        )

    prediction = ticket.get("resolver_prediction")

    if not isinstance(prediction, dict):
        raise HTTPException(
            status_code=502,
            detail="AI API did not return resolver_prediction."
        )

    resolver = prediction.get("resolver_group")
    priority = prediction.get("priority")

    try:
        resolver_confidence = float(
            prediction.get("resolver_confidence")
        )

        priority_confidence = float(
            prediction.get("priority_confidence")
        )

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="AI API returned invalid confidence values."
        )

    if not resolver:
        raise HTTPException(
            status_code=502,
            detail="AI API did not return a resolver group."
        )

    if priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid priority returned: {priority}"
        )

    if not 0 <= resolver_confidence <= 1:
        raise HTTPException(
            status_code=502,
            detail="Invalid resolver confidence."
        )

    if not 0 <= priority_confidence <= 1:
        raise HTTPException(
            status_code=502,
            detail="Invalid priority confidence."
        )

    ticket_id = ticket.get("ticket_id") or request.ticket_id

    incident_id = (
        ticket.get("incident_id")
        or request.incident_id
        or ticket_id
    )

    description = (
        ticket.get("description")
        or request.description
    )

    if not description:
        raise HTTPException(
            status_code=502,
            detail="Ticket description is missing."
        )

    return {
        "ticket_id": ticket_id,
        "incident_id": incident_id,
        "description": description,
        "resolver_group": resolver,
        "priority": priority,
        "resolver_confidence": resolver_confidence,
        "priority_confidence": priority_confidence,
        "requires_human_review": needs_human_validation(
            resolver_confidence,
            priority_confidence
        ),
        "review_reasons": get_review_reasons(
            resolver_confidence,
            priority_confidence
        ),
        "triage_reused": ticket.get("triage_reused", False),
        "created_at": ticket.get(
            "created_at",
            current_time().isoformat()
        )
    }


# ============================================================
# INCIDENT MANAGEMENT
# ============================================================

def store_ai_result(result: AIIncidentResult):

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

    # Preserve existing human-validated assignments.
    if existing.get("status") == "Assigned - Human Validated":

        incident = {
            **existing,
            "ticket_id": result.ticket_id,
            "recommended_assignment_group": result.resolver_group,
            "recommended_priority": result.priority,
            "resolver_confidence": result.resolver_confidence,
            "priority_confidence": result.priority_confidence,
            "last_updated": now
        }

        INCIDENTS[result.incident_id] = incident

        return {
            "status": "Success",
            "automatic_assignment": False,
            "message": "Existing human-validated assignment preserved.",
            "incident": incident
        }

    if needs_review:

        incident = {
            **existing,
            "incident_id": result.incident_id,
            "ticket_id": result.ticket_id,

            "assignment_group": existing.get("assignment_group"),
            "priority": existing.get("priority"),

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

        .container {
            max-width: 1600px;
            margin: auto;
        }

        .section, .stat {
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

        button.secondary { background: #555; }
        button.success { background: #16834a; }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 1100px;
        }

        th, td {
            text-align: left;
            padding: 11px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }

        th {
            background: #f5f6f8;
            white-space: nowrap;
        }

        .table-container { overflow-x: auto; }

        .description-cell {
            min-width: 260px;
            max-width: 400px;
            white-space: normal;
            overflow-wrap: anywhere;
            line-height: 1.5;
        }

        .ticket-description {
            padding: 12px 15px;
            margin: 7px 0 14px;
            background: #f5f7fa;
            border-left: 4px solid #2457c5;
            border-radius: 5px;
            line-height: 1.6;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            min-height: 35px;
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

        .error { background: #ffe0e0; color: #9b1515; }
        .info { background: #e6efff; color: #174b9c; }

        .tabs {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }

        .tabs button.active { background: #142d63; }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        @media(max-width: 750px) {
            body { padding: 12px; }
            .stats { grid-template-columns: 1fr 1fr; }
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>

<body>
<div class="container">

    <h1>Ethan Global - Mock ITSM</h1>
    <p>AI ticket triage and human validation dashboard</p>

    <div class="stats">
        <div class="stat">Total Tickets
            <strong id="totalTickets">0</strong>
        </div>
        <div class="stat">Needs Human Validation
            <strong id="reviewCount">0</strong>
        </div>
        <div class="stat">Automatically Assigned
            <strong id="assignedCount">0</strong>
        </div>
        <div class="stat">Total Incidents
            <strong id="incidentCount">0</strong>
        </div>
    </div>

    <div class="section">
        <h2>Submit Ticket for AI Triage</h2>

        <form id="triageForm">
            <div class="form-grid">
                <div>
                    <label>Ticket ID</label>
                    <input id="ticketId" required placeholder="TKT1001">
                </div>
                <div>
                    <label>Incident ID</label>
                    <input id="incidentId" placeholder="INC1001">
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
                        <th>Ticket Description</th>
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

    <div class="section" id="manualSection" style="display:none">
        <h2>Human Validation and Assignment</h2>
        <p id="reviewInformation"></p>

        <form id="manualForm">
            <label>Ticket ID</label>
            <input id="manualTicketId" readonly>

            <label>Incident ID</label>
            <input id="manualIncidentId" readonly>

            <label>Ticket Description</label>
            <div id="manualDescription" class="ticket-description"></div>

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
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        }
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            typeof data.detail === "string"
                ? data.detail
                : JSON.stringify(data.detail ?? data)
        );
    }

    return data;
}


// ============================================================
// STATISTICS
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
// TICKETS
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
                "<tr><td colspan='9'>No tickets found.</td></tr>";
            return;
        }

        tickets.forEach(ticket => {
            const review = ticket.requires_human_review;

            const badge = review
                ? '<span class="badge review">Needs Human Validation</span>'
                : '<span class="badge assigned">Assigned</span>';

            const action = review
                ? `<button data-action="validate"
                    data-ticket-id="${escapeHtml(ticket.ticket_id)}">Validate</button>`
                : `<button class="secondary" data-action="view"
                    data-ticket-id="${escapeHtml(ticket.ticket_id)}">View</button>`;

            const resolverConfidence =
                (Number(ticket.resolver_confidence || 0) * 100).toFixed(2);

            const priorityConfidence =
                (Number(ticket.priority_confidence || 0) * 100).toFixed(2);

            const description =
                ticket.description || "No description available.";

            tbody.innerHTML += `
                <tr>
                    <td>${escapeHtml(ticket.ticket_id)}</td>
                    <td>${escapeHtml(ticket.incident_id)}</td>
                    <td class="description-cell">${escapeHtml(description)}</td>
                    <td>${escapeHtml(ticket.resolver_group)}</td>
                    <td>${resolverConfidence}%</td>
                    <td>${escapeHtml(ticket.priority)}</td>
                    <td>${priorityConfidence}%</td>
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
// TICKET TABLE ACTIONS
// ============================================================

document.getElementById("ticketTable").addEventListener(
    "click",
    async function(event) {
        const button = event.target.closest("button[data-action]");

        if (!button) return;

        const ticketId = button.dataset.ticketId;

        if (button.dataset.action === "validate") {
            await openManualForm(ticketId);
        }

        if (button.dataset.action === "view") {
            await viewTicket(ticketId);
        }
    }
);


// ============================================================
// HUMAN VALIDATION FORM
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

        document.getElementById("manualDescription").textContent =
            ticket.description || "No description available.";

        document.getElementById("manualResolverGroup").value =
            ticket.resolver_group || "";

        document.getElementById("manualPriority").value =
            ticket.priority || "P4";

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


// ============================================================
// VIEW TICKET
// ============================================================

async function viewTicket(ticketId) {
    try {
        const ticket = await api(
            `/tickets/${encodeURIComponent(ticketId)}`
        );

        alert(
            "Ticket: " + ticket.ticket_id +
            "\\nIncident: " + ticket.incident_id +
            "\\nDescription: " + (ticket.description || "Not available") +
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
// SUBMIT TICKET FOR AI TRIAGE
// ============================================================

document.getElementById("triageForm").addEventListener(
    "submit",
    async function(event) {
        event.preventDefault();

        const payload = {
            ticket_id: document.getElementById("ticketId").value.trim(),
            incident_id:
                document.getElementById("incidentId").value.trim() || null,
            description:
                document.getElementById("description").value.trim()
        };

        try {
            const data = await api("/ai-triage", {
                method: "POST",
                body: JSON.stringify(payload)
            });

            const ticket = data.ticket || {};

            if (data.itsm?.automatic_assignment) {
                showMessage(
                    "triageMessage",
                    "AI triage completed. Ticket automatically assigned.",
                    "info"
                );
            } else {
                showMessage(
                    "triageMessage",
                    "AI triage completed. Human validation is required.",
                    "info"
                );
            }

            document.getElementById("triageForm").reset();

            await refreshDashboard();

            if (ticket.ticket_id && ticket.requires_human_review) {
                await openManualForm(ticket.ticket_id);
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

        const ticketId =
            document.getElementById("manualTicketId").value;

        const payload = {
            incident_id: incidentId,
            ticket_id: ticketId,
            resolver_group:
                document.getElementById("manualResolverGroup").value.trim(),
            priority:
                document.getElementById("manualPriority").value
        };

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
// INCIDENTS
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
// AUDIT LOG
// ============================================================

async function loadAudit() {
    try {
        const data = await api("/audit");
        const tbody = document.getElementById("auditTable");

        tbody.innerHTML = "";

        if (!data.audit.length) {
            tbody.innerHTML =
                "<tr><td colspan='7'>No audit records found.</td></tr>";
            return;
        }

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

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI API returned {exc.response.status_code}: "
                   f"{exc.response.text}"
        )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to AI API: {str(exc)}"
        )

    normalized = normalize_ai_ticket(data, request)

    existing_ticket = TICKETS.get(
        normalized["ticket_id"],
        {}
    )

    # Create the incident result.
    result = AIIncidentResult(
        ticket_id=normalized["ticket_id"],
        incident_id=normalized["incident_id"],
        description=normalized["description"],
        resolver_group=normalized["resolver_group"],
        priority=normalized["priority"],
        resolver_confidence=normalized["resolver_confidence"],
        priority_confidence=normalized["priority_confidence"],
        review_reasons=normalized["review_reasons"]
    )

    # Store incident and AI recommendation.
    itsm_result = store_ai_result(result)

    human_validated = (
        INCIDENTS.get(result.incident_id, {}).get("status")
        == "Assigned - Human Validated"
    )

    needs_review = (
        needs_human_validation(
            result.resolver_confidence,
            result.priority_confidence
        )
        and not human_validated
    )

    # Preserve the description and existing ticket fields.
    ticket_record = {
        **existing_ticket,
        **normalized,
        "description": normalized["description"],
        "requires_human_review": needs_review,
        "review_reasons": (
            get_review_reasons(
                result.resolver_confidence,
                result.priority_confidence
            )
            if needs_review
            else []
        ),
        "status": (
            "Assigned - Human Validated"
            if human_validated
            else "Pending Human Review"
            if needs_review
            else "Assigned"
        ),
        "last_updated": current_time().isoformat()
    }

    save_ticket(ticket_record)

    return {
        "status": "Success",
        "message": "AI triage completed successfully.",
        "ticket": ticket_record,
        "itsm": itsm_result,
        "ai_response": data
    }


# ============================================================
# AI CALLBACK ENDPOINT
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

    # Save or update incident.
    result_data = store_ai_result(result)

    existing = TICKETS.get(result.ticket_id, {})

    human_validated = (
        INCIDENTS.get(incident_id, {}).get("status")
        == "Assigned - Human Validated"
    )

    needs_review = (
        needs_human_validation(
            result.resolver_confidence,
            result.priority_confidence
        )
        and not human_validated
    )

    # Preserve existing description if callback does not contain it.
    description = (
        result.description
        or existing.get("description")
        or ""
    )

    ticket = {
        **existing,
        "ticket_id": result.ticket_id,
        "incident_id": result.incident_id,
        "description": description,
        "resolver_group": result.resolver_group,
        "priority": result.priority,
        "resolver_confidence": result.resolver_confidence,
        "priority_confidence": result.priority_confidence,
        "requires_human_review": needs_review,
        "review_reasons": (
            get_review_reasons(
                result.resolver_confidence,
                result.priority_confidence
            )
            if needs_review
            else []
        ),
        "status": (
            "Assigned - Human Validated"
            if human_validated
            else "Pending Human Review"
            if needs_review
            else "Assigned"
        ),
        "last_updated": current_time().isoformat()
    }

    save_ticket(ticket)

    return {
        "status": "Success",
        "message": "AI result received and ticket updated.",
        "ticket": ticket,
        "incident": result_data["incident"]
    }


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

    # Update every ticket belonging to this incident.
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
        ticket
        for ticket in TICKETS.values()
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
        ticket
        for ticket in TICKETS.values()
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
