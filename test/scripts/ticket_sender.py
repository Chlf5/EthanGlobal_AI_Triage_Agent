
import json
import random
import time
import requests
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# CONFIGURATION
# ==========================================

MAX_TICKETS = 100

API_URL = "http://127.0.0.1:8000/triage"

OUTPUT_JSON = Path("test/scripts/test_tickets.json")
OUTPUT_EXCEL = Path("test/scripts/ethan_api_test_report.xlsx")

TIMEOUT = 60

random.seed(42)


# ==========================================
# TICKET TEMPLATES
# ==========================================

TICKET_TEMPLATES = {

    "Network Operations": {
        "priority": ["P2", "P3"],
        "descriptions": [
            "Employees cannot connect to the company Wi-Fi. Authentication fails on multiple devices.",
            "The office network is experiencing intermittent connectivity issues.",
            "The VPN connection fails when employees attempt to access internal resources.",
            "Network switches are unreachable and several departments have lost network access.",
            "Users are experiencing slow internet connectivity across the office.",
            "The company DNS server is not resolving internal hostnames.",
            "Multiple employees cannot access shared network drives.",
            "The firewall appears to be blocking legitimate internal network traffic."
        ]
    },

    "Desktop Support": {
        "priority": ["P3", "P4"],
        "descriptions": [
            "My laptop is running very slowly and applications take several minutes to open.",
            "The computer screen flickers randomly during normal usage.",
            "My keyboard and mouse stopped responding after restarting the computer.",
            "The laptop battery is not charging even when connected to power.",
            "The computer keeps restarting unexpectedly.",
            "An employee cannot log in to their workstation.",
            "The monitor displays a black screen after starting the computer.",
            "The laptop webcam and microphone are not detected."
        ]
    },

    "Application Support": {
        "priority": ["P2", "P3"],
        "descriptions": [
            "The company application crashes whenever users attempt to open it.",
            "Users receive an error when trying to submit application forms.",
            "The application dashboard is not loading correctly.",
            "Employees cannot access the document management application.",
            "The application freezes when generating reports.",
            "Users are unable to save changes in the application."
        ]
    },

    "CRM Support": {
        "priority": ["P2", "P3"],
        "descriptions": [
            "The CRM system is unavailable and sales employees cannot access customer records.",
            "Customer information is not loading in the CRM dashboard.",
            "The CRM application fails when creating new customer accounts.",
            "Sales representatives cannot update customer opportunities.",
            "CRM reports are showing incorrect information."
        ]
    },

    "IT Security Operations": {
        "priority": ["P1", "P2"],
        "descriptions": [
            "Multiple suspicious login attempts were detected against employee accounts.",
            "An employee reported a suspicious email containing a potentially malicious attachment.",
            "The endpoint security system detected possible malware activity.",
            "An account appears to have been accessed without authorization.",
            "Several computers are reporting security policy violations."
        ]
    },

    "Database Administration": {
        "priority": ["P1", "P2"],
        "descriptions": [
            "The production database is unavailable and applications cannot retrieve records.",
            "Database queries are taking significantly longer than usual.",
            "The database server is reporting repeated connection failures.",
            "A database backup job failed and requires investigation.",
            "Users are receiving errors when saving information to the database."
        ]
    }
}


# ==========================================
# GENERATE TICKETS
# ==========================================

def generate_tickets():

    tickets = []

    resolver_groups = list(TICKET_TEMPLATES.keys())

    for index in range(1, MAX_TICKETS + 1):

        resolver_group = random.choice(resolver_groups)

        template = TICKET_TEMPLATES[resolver_group]

        description = random.choice(template["descriptions"])
        priority = random.choice(template["priority"])

        tickets.append({
            "ticket_id": f"TEST-{index:04d}",
            "description": description,
            "actual_resolver": resolver_group,
            "actual_priority": priority
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(tickets, file, indent=4, ensure_ascii=False)

    print(f"\nGenerated {len(tickets)} tickets.")

    return tickets


# ==========================================
# SEND TICKET TO API
# ==========================================

def send_ticket(ticket):

    payload = {
        "ticket_id": ticket["ticket_id"],
        "description": ticket["description"]
    }

    start_time = time.perf_counter()

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        elapsed = time.perf_counter() - start_time

        api_data = response.json()

        returned_ticket = api_data.get("ticket", {})
        resolver = returned_ticket.get("resolver_prediction", {})

        status = returned_ticket.get("status", "UNKNOWN")

        if status != "Processed":
            raise ValueError(f"Unexpected API status: {status}")

        return {
            "ticket_id": ticket["ticket_id"],
            "description": ticket["description"],

            "actual_resolver": ticket["actual_resolver"],
            "predicted_resolver": resolver.get("resolver_group"),
            "resolver_confidence": resolver.get("resolver_confidence"),

            "actual_priority": ticket["actual_priority"],
            "predicted_priority": resolver.get("priority"),
            "priority_confidence": resolver.get("priority_confidence"),

            "response_time_seconds": round(elapsed, 4),
            "status": "SUCCESS",
            "api_message": api_data.get("message"),
            "error": None
        }

    except (requests.exceptions.RequestException, ValueError, KeyError) as error:

        elapsed = time.perf_counter() - start_time

        return {
            "ticket_id": ticket["ticket_id"],
            "description": ticket["description"],

            "actual_resolver": ticket["actual_resolver"],
            "predicted_resolver": None,
            "resolver_confidence": None,

            "actual_priority": ticket["actual_priority"],
            "predicted_priority": None,
            "priority_confidence": None,

            "response_time_seconds": round(elapsed, 4),
            "status": "FAILED",
            "api_message": None,
            "error": str(error)
        }


# ==========================================
# CLASSIFICATION STATISTICS
# ==========================================

def calculate_statistics(df, actual_column, predicted_column):

    valid = df.dropna(
        subset=[actual_column, predicted_column]
    )

    if valid.empty:
        return pd.DataFrame()

    report = classification_report(
        valid[actual_column],
        valid[predicted_column],
        output_dict=True,
        zero_division=0
    )

    rows = []

    for label, metrics in report.items():

        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue

        rows.append({
            "Class": label,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1-score"],
            "Support": metrics["support"]
        })

    rows.append({
        "Class": "OVERALL",
        "Precision": None,
        "Recall": None,
        "F1 Score": report.get("weighted avg", {}).get("f1-score"),
        "Support": len(valid)
    })

    return pd.DataFrame(rows)


# ==========================================
# GENERATE EXCEL REPORT
# ==========================================

def generate_report(results):

    df = pd.DataFrame(results)

    successful = df[df["status"] == "SUCCESS"]
    failed = df[df["status"] == "FAILED"]

    resolver_valid = successful.dropna(
        subset=["actual_resolver", "predicted_resolver"]
    )

    priority_valid = successful.dropna(
        subset=["actual_priority", "predicted_priority"]
    )

    resolver_accuracy = None
    priority_accuracy = None

    if not resolver_valid.empty:

        resolver_accuracy = accuracy_score(
            resolver_valid["actual_resolver"],
            resolver_valid["predicted_resolver"]
        )

    if not priority_valid.empty:

        priority_accuracy = accuracy_score(
            priority_valid["actual_priority"],
            priority_valid["predicted_priority"]
        )

    response_times = successful["response_time_seconds"].dropna()

    summary = pd.DataFrame([
        {"Metric": "Total Tickets", "Value": len(df)},
        {"Metric": "Successful Requests", "Value": len(successful)},
        {"Metric": "Failed Requests", "Value": len(failed)},
        {"Metric": "Resolver Evaluation Count", "Value": len(resolver_valid)},
        {"Metric": "Priority Evaluation Count", "Value": len(priority_valid)},
        {"Metric": "Resolver Accuracy", "Value": resolver_accuracy},
        {"Metric": "Priority Accuracy", "Value": priority_accuracy},
        {
            "Metric": "Average Response Time (seconds)",
            "Value": response_times.mean() if not response_times.empty else None
        },
        {
            "Metric": "P95 Response Time (seconds)",
            "Value": response_times.quantile(0.95)
            if not response_times.empty else None
        }
    ])

    resolver_stats = calculate_statistics(
        df,
        "actual_resolver",
        "predicted_resolver"
    )

    priority_stats = calculate_statistics(
        df,
        "actual_priority",
        "predicted_priority"
    )

    errors = failed[
        ["ticket_id", "description", "error"]
    ]

    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

        df.to_excel(
            writer,
            sheet_name="Test Results",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        resolver_stats.to_excel(
            writer,
            sheet_name="Resolver Statistics",
            index=False
        )

        priority_stats.to_excel(
            writer,
            sheet_name="Priority Statistics",
            index=False
        )

        errors.to_excel(
            writer,
            sheet_name="API Errors",
            index=False
        )

    # ======================================
    # PRINT SUMMARY
    # ======================================

    print("\n========== API TEST SUMMARY ==========")

    print(f"Total tickets: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if resolver_accuracy is not None:
        print(f"Resolver Accuracy: {resolver_accuracy:.2%}")

    if priority_accuracy is not None:
        print(f"Priority Accuracy: {priority_accuracy:.2%}")

    if not response_times.empty:

        print(
            f"Average Response Time: "
            f"{response_times.mean():.4f} seconds"
        )

        print(
            f"P95 Response Time: "
            f"{response_times.quantile(0.95):.4f} seconds"
        )

    print(f"\nExcel report saved: {OUTPUT_EXCEL.resolve()}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    print("\n========== ETHAN AI TICKET TEST ==========")

    # Step 1: Generate tickets
    tickets = generate_tickets()

    results = []

    print("\nStarting API requests...\n")

    # Step 2: Send tickets to API
    for index, ticket in enumerate(tickets, start=1):

        result = send_ticket(ticket)

        results.append(result)

        print(
            f"[{index}/{MAX_TICKETS}] "
            f"{ticket['ticket_id']} - {result['status']}"
        )

    # Step 3: Generate Excel report
    generate_report(results)

    print("\nTesting completed.")


if __name__ == "__main__":
    main()
