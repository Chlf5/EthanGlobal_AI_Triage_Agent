
import json
import uuid
import time
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TICKETS = 100

API_URL = "http://127.0.0.1:8000/triage"

DATASET_PATH = Path("data/raw/ethan_tickets_10000.csv")

OUTPUT_JSON = Path("test/scripts/test_tickets.json")
OUTPUT_EXCEL = Path("test/scripts/ethan_api_test_report.xlsx")

REQUEST_TIMEOUT = 60
REQUEST_DELAY = 0.1

DIFFICULTY_DISTRIBUTION = {
    "Easy": 0.10,
    "Medium": 0.20,
    "Hard": 0.70
}

RUN_ID = uuid.uuid4().hex[:8].upper()
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RANDOM_STATE = int(RUN_ID, 16)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("ETHAN GLOBAL - AI TRIAGE API TEST")
print("=" * 80)

print(f"Run ID:        {RUN_ID}")
print(f"Timestamp:     {RUN_TIMESTAMP}")
print(f"API URL:       {API_URL}")
print(f"Maximum tests: {MAX_TICKETS}")
print("Distribution:  10% Easy | 20% Medium | 70% Hard")

print("=" * 80)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(DATASET_PATH)

    # Detect description column.
    if "description" in df.columns:
        description_column = "description"

    elif "text" in df.columns:
        description_column = "text"

    else:
        raise ValueError(
            f"Dataset must contain 'description' or 'text'. "
            f"Available columns: {df.columns.tolist()}"
        )

    required_columns = ["resolver_group", "priority"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing dataset column: {column}"
            )

    # Create a consistent description column.
    df["description"] = (
        df[description_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["resolver_group"] = (
        df["resolver_group"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["priority"] = (
        df["priority"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Remove invalid records.
    df = df[
        (df["description"].str.len() >= 5)
        & (df["resolver_group"].str.len() > 0)
        & (df["priority"].isin(["P1", "P2", "P3", "P4"]))
    ].copy()

    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No valid tickets found in dataset.")

    print(f"\nDataset loaded: {len(df)} tickets")
    print(f"Resolver groups: {df['resolver_group'].nunique()}")
    print(f"Priority classes: {df['priority'].nunique()}")
    print(f"Original description column: {description_column}")

    print("\nSample description:")
    print(df["description"].iloc[0])

    return df


# ============================================================
# CALCULATE DESCRIPTION COMPLEXITY
# ============================================================

TECHNICAL_TERMS = [
    "server", "database", "network", "firewall", "vpn",
    "dns", "dhcp", "domain controller", "authentication",
    "authorization", "certificate", "encryption", "endpoint",
    "configuration", "integration", "synchronization",
    "infrastructure", "deployment", "application", "api",
    "sql", "driver", "firmware", "router", "switch",
    "latency", "timeout", "permission", "access denied",
    "service unavailable", "memory", "cpu", "disk space",
    "backup", "replication", "load balancer", "active directory",
    "identity provider", "virtual machine", "container",
    "kubernetes", "storage array", "failover", "cluster",
    "ssl", "tls", "proxy", "certificate authority",
    "single sign-on", "sso", "mfa", "ldap",
    "microservice", "api gateway", "data pipeline",
    "message queue", "cloud infrastructure", "operating system",
    "power supply", "raid controller"
]

COMPLEXITY_INDICATORS = [
    "multiple", "several", "intermittent", "simultaneously",
    "across departments", "all users", "entire department",
    "production", "business critical", "outage",
    "after the update", "after deployment",
    "following the migration", "root cause", "dependency",
    "cascading", "recurring", "unpredictable", "inconsistent",
    "partial failure", "degraded performance", "data corruption",
    "synchronization failure", "service disruption",
    "authentication errors", "connection failures",
    "while", "although", "however", "in addition",
    "as a result", "at the same time", "without warning",
    "across multiple systems"
]

IMPACT_INDICATORS = [
    "critical", "urgent", "business impact", "financial impact",
    "security breach", "data loss", "data integrity",
    "unauthorized access", "ransomware", "compliance",
    "customer impact", "system-wide", "enterprise-wide",
    "mission critical", "complete outage", "unavailable",
    "failed transactions", "production environment",
    "business operations", "sensitive information",
    "service interruption", "multiple departments",
    "customer records"
]

INVESTIGATION_INDICATORS = [
    "investigate", "diagnose", "troubleshoot",
    "identify the root cause", "logs indicate",
    "monitoring shows", "correlation", "configuration mismatch",
    "dependency failure", "rollback", "recovery", "failover",
    "cross-system", "upstream", "downstream", "analyze logs",
    "review configurations", "trace the issue",
    "isolate the problem", "restore service",
    "compare configurations", "check dependencies"
]


def calculate_complexity(description):

    text = description.lower()
    word_count = len(text.split())

    technical_count = sum(
        term in text for term in TECHNICAL_TERMS
    )

    indicator_count = sum(
        term in text for term in COMPLEXITY_INDICATORS
    )

    impact_count = sum(
        term in text for term in IMPACT_INDICATORS
    )

    investigation_count = sum(
        term in text for term in INVESTIGATION_INDICATORS
    )

    score = 0

    if word_count >= 15:
        score += 1
    if word_count >= 30:
        score += 2
    if word_count >= 50:
        score += 2
    if word_count >= 80:
        score += 2

    score += min(technical_count * 2, 12)
    score += min(indicator_count * 2, 10)
    score += min(impact_count * 3, 12)
    score += min(investigation_count * 2, 8)

    return {
        "word_count": word_count,
        "technical_terms_count": technical_count,
        "complexity_indicators_count": indicator_count,
        "impact_indicators_count": impact_count,
        "investigation_indicators_count": investigation_count,
        "complexity_score": score
    }


# ============================================================
# ASSIGN DIFFICULTY
# ============================================================

def assign_difficulty(df):

    complexity = df["description"].apply(
        calculate_complexity
    )

    complexity_df = pd.DataFrame(
        complexity.tolist(),
        index=df.index
    )

    df = pd.concat([df, complexity_df], axis=1)

    df["difficulty"] = pd.qcut(
        df["complexity_score"].rank(method="first"),
        q=3,
        labels=["Easy", "Medium", "Hard"]
    )

    print("\nComplexity score distribution:")

    print(
        df.groupby("difficulty", observed=False)
        ["complexity_score"]
        .agg(["min", "mean", "max", "count"])
    )

    return df


# ============================================================
# GENERATE TEST TICKETS
# ============================================================

def generate_test_tickets(df):

    sample_size = min(MAX_TICKETS, len(df))

    if sample_size < 3:
        raise ValueError("At least 3 valid tickets are required.")

    targets = {
        level: int(sample_size * percentage)
        for level, percentage in DIFFICULTY_DISTRIBUTION.items()
    }

    targets["Hard"] += sample_size - sum(targets.values())

    print("\nTarget distribution:")
    print(targets)

    selected = []
    remaining = df.copy()

    for level in ["Easy", "Medium", "Hard"]:

        candidates = remaining[
            remaining["difficulty"].astype(str) == level
        ]

        requested = targets[level]
        available = min(requested, len(candidates))

        if available > 0:

            sample = candidates.sample(
                n=available,
                replace=False,
                random_state=RANDOM_STATE + len(selected)
            )

            selected.append(sample)

            remaining = remaining.drop(index=sample.index)

    selected_df = (
        pd.concat(selected)
        if selected
        else pd.DataFrame()
    )

    missing = sample_size - len(selected_df)

    if missing > 0:

        additional = remaining.sample(
            n=missing,
            replace=False,
            random_state=RANDOM_STATE
        )

        selected.append(additional)

    sample = (
        pd.concat(selected)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    tickets = []

    for index, row in sample.iterrows():

        ticket = {
            "ticket_id": f"TEST-{RUN_ID}-{index + 1:04d}",

            "incident_id": f"INC-{RUN_ID}-{index + 1:06d}",

            # Always preserve the original description.
            "description": str(row["description"]),

            "actual_resolver_group": row["resolver_group"],
            "actual_priority": row["priority"],

            "difficulty": str(row["difficulty"]),

            "word_count": int(row["word_count"]),
            "complexity_score": int(row["complexity_score"]),

            "technical_terms_count": int(
                row["technical_terms_count"]
            ),

            "complexity_indicators_count": int(
                row["complexity_indicators_count"]
            ),

            "impact_indicators_count": int(
                row["impact_indicators_count"]
            ),

            "investigation_indicators_count": int(
                row["investigation_indicators_count"]
            )
        }

        tickets.append(ticket)

    print(f"\nGenerated {len(tickets)} tickets.")

    print("\nFirst generated ticket:")
    print(json.dumps(tickets[0], indent=4, ensure_ascii=False))

    print("\nActual difficulty distribution:")
    print(
        pd.Series(
            [ticket["difficulty"] for ticket in tickets]
        ).value_counts()
    )

    return tickets


# ============================================================
# SEND TICKET TO AI API
# ============================================================

def send_ticket(ticket):

    # Preserve the original description in the result.
    original_description = ticket["description"]

    payload = {
        "ticket_id": ticket["ticket_id"],
        "description": original_description,
        "incident_id": ticket["incident_id"]
    }

    result = {
        **ticket,

        "description": original_description,

        "predicted_resolver_group": None,
        "predicted_priority": None,

        "resolver_confidence": None,
        "priority_confidence": None,

        "ticket_status": None,
        "requires_human_review": None,
        "review_reasons": [],

        "semantic_duplicate": None,
        "semantic_duplicate_matches": [],

        "itsm_status": None,
        "automatic_assignment": None,
        "itsm_incident_id": None,

        "response_time_seconds": None,
        "http_status_code": None,

        "api_status": "FAILED",
        "error": None
    }

    response = None
    start_time = time.perf_counter()

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        elapsed = time.perf_counter() - start_time

        result["response_time_seconds"] = round(elapsed, 4)
        result["http_status_code"] = response.status_code

        response.raise_for_status()

        api_data = response.json()

        if not isinstance(api_data, dict):
            raise ValueError("API returned an invalid JSON structure.")

        # ----------------------------------------------------
        # Extract ticket object
        # ----------------------------------------------------

        returned_ticket = api_data.get("ticket")

        if not isinstance(returned_ticket, dict):
            raise ValueError(
                "Missing or invalid ticket object in API response."
            )

        # Use returned description only if it is non-empty.
        # Otherwise keep the original dataset description.
        returned_description = returned_ticket.get("description")

        if isinstance(returned_description, str):
            if returned_description.strip():
                result["description"] = returned_description.strip()

        prediction = returned_ticket.get("resolver_prediction")

        if not isinstance(prediction, dict):
            raise ValueError(
                "Missing resolver_prediction object."
            )

        predicted_resolver = prediction.get("resolver_group")
        predicted_priority = prediction.get("priority")

        if not predicted_resolver:
            raise ValueError("Resolver group prediction is missing.")

        if predicted_priority not in {"P1", "P2", "P3", "P4"}:
            raise ValueError(
                f"Invalid priority: {predicted_priority}"
            )

        # ----------------------------------------------------
        # Extract confidence
        # ----------------------------------------------------

        resolver_confidence = float(
            prediction.get("resolver_confidence")
        )

        priority_confidence = float(
            prediction.get("priority_confidence")
        )

        if not 0 <= resolver_confidence <= 1:
            raise ValueError("Invalid resolver confidence.")

        if not 0 <= priority_confidence <= 1:
            raise ValueError("Invalid priority confidence.")

        result["predicted_resolver_group"] = predicted_resolver
        result["predicted_priority"] = predicted_priority

        result["resolver_confidence"] = resolver_confidence
        result["priority_confidence"] = priority_confidence

        result["ticket_status"] = returned_ticket.get(
            "status", "UNKNOWN"
        )

        result["requires_human_review"] = returned_ticket.get(
            "requires_human_review", False
        )

        result["review_reasons"] = returned_ticket.get(
            "review_reasons", []
        )

        # ----------------------------------------------------
        # Extract semantic duplicate information
        # ----------------------------------------------------

        duplicate_data = api_data.get(
            "semantic_duplicate_check", {}
        )

        if isinstance(duplicate_data, dict):

            result["semantic_duplicate"] = duplicate_data.get(
                "possible_duplicate"
            )

            result["semantic_duplicate_matches"] = (
                duplicate_data.get("matches", [])
            )

        # ----------------------------------------------------
        # Extract ITSM response
        # ----------------------------------------------------

        itsm_data = api_data.get("itsm", {})

        if not isinstance(itsm_data, dict):
            itsm_data = {}

        itsm_response = itsm_data.get("response", {})

        if not isinstance(itsm_response, dict):
            itsm_response = {}

        itsm_incident = itsm_response.get("incident", {})

        if not isinstance(itsm_incident, dict):
            itsm_incident = {}

        result["itsm_status"] = (
            itsm_data.get("status")
            or itsm_response.get("status")
            or "UNKNOWN"
        )

        result["automatic_assignment"] = (
            itsm_data.get("automatic_assignment")
            if "automatic_assignment" in itsm_data
            else itsm_response.get("automatic_assignment")
        )

        result["itsm_incident_id"] = (
            itsm_incident.get("incident_id")
            or returned_ticket.get("incident_id")
            or ticket["incident_id"]
        )

        result["api_status"] = "SUCCESS"

        # Retain full response for troubleshooting.
        result["api_response"] = api_data

    except requests.exceptions.ConnectionError:

        result["error"] = (
            "Connection error. Check whether the AI API is running."
        )

    except requests.exceptions.Timeout:

        result["error"] = "API request timed out."

    except requests.exceptions.HTTPError as exc:

        response_text = (
            response.text[:1000]
            if response is not None
            else "No response body"
        )

        result["error"] = (
            f"HTTP error: {exc}. Response: {response_text}"
        )

    except (ValueError, TypeError, KeyError) as exc:

        result["error"] = str(exc)

    except requests.exceptions.RequestException as exc:

        result["error"] = f"Request error: {exc}"

    except Exception as exc:

        result["error"] = f"Unexpected error: {exc}"

    # Ensure description is never lost.
    if not result.get("description"):
        result["description"] = original_description

    return result


# ============================================================
# RUN TESTS
# ============================================================

def run_tests(tickets):

    results = []

    print("\nStarting API tests...\n")

    for index, ticket in enumerate(tickets, start=1):

        result = send_ticket(ticket)
        results.append(result)

        if result["api_status"] == "SUCCESS":

            print(
                f"[{index}/{len(tickets)}] "
                f"[{ticket['difficulty'].upper():6}] "
                f"{ticket['ticket_id']} | "
                f"Resolver: {result['predicted_resolver_group']} "
                f"({result['resolver_confidence']:.2%}) | "
                f"Priority: {result['predicted_priority']} "
                f"({result['priority_confidence']:.2%}) | "
                f"Duplicate: {result['semantic_duplicate']} | "
                f"Human Review: {result['requires_human_review']}"
            )

        else:

            print(
                f"[{index}/{len(tickets)}] "
                f"[{ticket['difficulty'].upper():6}] "
                f"{ticket['ticket_id']} | FAILED | "
                f"{result['error']}"
            )

        time.sleep(REQUEST_DELAY)

    return results


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def build_classification_report(
    successful,
    actual_column,
    predicted_column
):

    if successful.empty:
        return pd.DataFrame()

    report = classification_report(
        successful[actual_column],
        successful[predicted_column],
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose()


# ============================================================
# CATEGORY STATISTICS
# ============================================================

def build_category_statistics(
    successful,
    actual_column,
    predicted_column,
    confidence_column,
    display_column
):

    if successful.empty:
        return pd.DataFrame()

    working = successful.copy()

    working["is_correct"] = (
        working[actual_column] == working[predicted_column]
    )

    stats = (
        working.groupby(actual_column)
        .agg(
            Total_Tickets=("ticket_id", "count"),
            Correct_Predictions=("is_correct", "sum"),
            Average_Confidence=(confidence_column, "mean")
        )
        .reset_index()
        .rename(columns={actual_column: display_column})
    )

    stats["Accuracy"] = (
        stats["Correct_Predictions"] / stats["Total_Tickets"]
    )

    return stats


# ============================================================
# DIFFICULTY STATISTICS
# ============================================================

def build_difficulty_statistics(successful):

    if successful.empty:
        return pd.DataFrame()

    return (
        successful.groupby("difficulty", observed=False)
        .agg(
            Total_Tickets=("ticket_id", "count"),

            Average_Complexity_Score=(
                "complexity_score", "mean"
            ),

            Resolver_Accuracy=("resolver_correct", "mean"),
            Priority_Accuracy=("priority_correct", "mean"),

            Average_Resolver_Confidence=(
                "resolver_confidence", "mean"
            ),

            Average_Priority_Confidence=(
                "priority_confidence", "mean"
            ),

            Human_Review_Count=(
                "requires_human_review", "sum"
            ),

            Average_Response_Time=(
                "response_time_seconds", "mean"
            )
        )
        .reset_index()
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(results):

    df = pd.DataFrame(results)

    successful = df[
        df["api_status"] == "SUCCESS"
    ].copy()

    failed = df[
        df["api_status"] == "FAILED"
    ].copy()

    # --------------------------------------------------------
    # Accuracy calculations
    # --------------------------------------------------------

    resolver_accuracy = None
    priority_accuracy = None

    if not successful.empty:

        successful["resolver_correct"] = (
            successful["actual_resolver_group"]
            == successful["predicted_resolver_group"]
        )

        successful["priority_correct"] = (
            successful["actual_priority"]
            == successful["predicted_priority"]
        )

        resolver_accuracy = accuracy_score(
            successful["actual_resolver_group"],
            successful["predicted_resolver_group"]
        )

        priority_accuracy = accuracy_score(
            successful["actual_priority"],
            successful["predicted_priority"]
        )

    # --------------------------------------------------------
    # Classification reports
    # --------------------------------------------------------

    resolver_report = build_classification_report(
        successful,
        "actual_resolver_group",
        "predicted_resolver_group"
    )

    priority_report = build_classification_report(
        successful,
        "actual_priority",
        "predicted_priority"
    )

    # --------------------------------------------------------
    # Operational statistics
    # --------------------------------------------------------

    total_tickets = len(df)
    successful_count = len(successful)
    failed_count = len(failed)

    human_review_count = (
        int(
            successful["requires_human_review"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if not successful.empty else 0
    )

    automatic_assignment_count = (
        int(
            successful["automatic_assignment"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if not successful.empty else 0
    )

    average_response_time = (
        successful["response_time_seconds"].mean()
        if not successful.empty else 0
    )

    max_response_time = (
        successful["response_time_seconds"].max()
        if not successful.empty else 0
    )

    min_response_time = (
        successful["response_time_seconds"].min()
        if not successful.empty else 0
    )

    duplicate_count = (
        int(successful["semantic_duplicate"].fillna(False).sum())
        if not successful.empty else 0
    )

    summary = {
        "Run ID": RUN_ID,
        "Run Timestamp": RUN_TIMESTAMP,
        "Total Tickets": total_tickets,
        "Successful API Requests": successful_count,
        "Failed API Requests": failed_count,

        "API Success Rate": (
            successful_count / total_tickets
            if total_tickets else 0
        ),

        "Resolver Accuracy": resolver_accuracy,
        "Priority Accuracy": priority_accuracy,

        "Semantic Duplicates Detected": duplicate_count,

        "Automatically Assigned Tickets": automatic_assignment_count,
        "Tickets Requiring Human Review": human_review_count,

        "Average Response Time (Seconds)": average_response_time,
        "Maximum Response Time (Seconds)": max_response_time,
        "Minimum Response Time (Seconds)": min_response_time
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    # --------------------------------------------------------
    # Resolver and priority statistics
    # --------------------------------------------------------

    resolver_stats = build_category_statistics(
        successful,
        "actual_resolver_group",
        "predicted_resolver_group",
        "resolver_confidence",
        "Resolver Group"
    )

    priority_stats = build_category_statistics(
        successful,
        "actual_priority",
        "predicted_priority",
        "priority_confidence",
        "Priority"
    )

    difficulty_stats = build_difficulty_statistics(successful)

    # --------------------------------------------------------
    # API errors
    # --------------------------------------------------------

    error_columns = [
        "ticket_id",
        "incident_id",
        "difficulty",
        "complexity_score",
        "description",
        "error"
    ]

    errors_df = (
        failed[error_columns].copy()
        if not failed.empty
        else pd.DataFrame(columns=error_columns)
    )

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    json_results = []

    for result in results:

        clean_result = {
            key: value
            for key, value in result.items()
            if key != "api_response"
        }

        json_results.append(clean_result)

    json_output = {
        "run_id": RUN_ID,
        "run_timestamp": RUN_TIMESTAMP,
        "total_tickets": total_tickets,
        "successful_requests": successful_count,
        "failed_requests": failed_count,

        "difficulty_distribution": (
            df["difficulty"].value_counts().to_dict()
        ),

        "results": json_results
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_output,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # --------------------------------------------------------
    # EXPORT EXCEL
    # --------------------------------------------------------

    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)

    excel_results = df.drop(
        columns=["api_response"],
        errors="ignore"
    ).copy()

    # Keep description explicitly as a report column.
    if "description" not in excel_results.columns:
        excel_results["description"] = ""

    excel_results["review_reasons"] = (
        excel_results["review_reasons"].apply(
            lambda value: ", ".join(value)
            if isinstance(value, list)
            else str(value)
        )
    )

    excel_results["semantic_duplicate_matches"] = (
        excel_results["semantic_duplicate_matches"].apply(
            lambda value: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (list, dict))
            else str(value)
        )
    )

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        excel_results.to_excel(
            writer,
            sheet_name="Test Results",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        difficulty_stats.to_excel(
            writer,
            sheet_name="Difficulty Statistics",
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

        resolver_report.to_excel(
            writer,
            sheet_name="Resolver Classification"
        )

        priority_report.to_excel(
            writer,
            sheet_name="Priority Classification"
        )

        errors_df.to_excel(
            writer,
            sheet_name="API Errors",
            index=False
        )

    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(summary):

    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)

    for metric, value in summary.items():

        if isinstance(value, float):

            if "Accuracy" in metric or "Rate" in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")

        else:
            print(f"{metric}: {value}")

    print("\nGenerated files:")
    print(f"JSON:  {OUTPUT_JSON.resolve()}")
    print(f"Excel: {OUTPUT_EXCEL.resolve()}")

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # 1. Load dataset.
        df = load_dataset()

        # 2. Calculate complexity and difficulty.
        df = assign_difficulty(df)

        # 3. Generate test tickets.
        tickets = generate_test_tickets(df)

        # 4. Send tickets to AI API.
        results = run_tests(tickets)

        # 5. Generate reports.
        summary = generate_report(results)

        # 6. Print final summary.
        print_summary(summary)

    except Exception as exc:

        print("\nTEST EXECUTION FAILED")
        print(f"Error: {str(exc)}")

        raise


if __name__ == "__main__":
    main()
