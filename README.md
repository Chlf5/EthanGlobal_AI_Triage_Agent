# Ethan Global – AI-Powered IT Ticket Triage Agent

## 1. Project Overview

The **Ethan Global AI Triage Agent** is an AI-powered system designed to automate the classification of IT support tickets.

When an employee reports an IT problem, the system analyzes the ticket description and automatically predicts:

- **Resolver Group:** The IT team responsible for solving the problem.
- **Priority:** How urgent the ticket is.

The objective is to reduce manual work, improve ticket assignment, and help IT support teams respond to problems more efficiently.

## 2. Key Features

- **Automatic Ticket Classification:** Predicts the appropriate IT resolver group.
- **Priority Prediction:** Classifies tickets into four priority levels.
- **Multitask AI Model:** Uses one shared BERT-based model to perform two prediction tasks.
- **Confidence Scores:** Provides confidence scores for both predictions.
- **Human Review:** Identifies predictions that may require manual verification.
- **REST API:** Makes AI predictions available through FastAPI.
- **ITSM Integration Demo:** Demonstrates how predictions can be transferred to a mock IT Service Management system.

## 3. How It Works

The system follows a simple workflow:

1. An employee submits an IT support ticket.
2. The API receives the ticket description.
3. The AI model analyzes the text.
4. The model predicts the resolver group.
5. The model predicts the ticket priority.
6. The system returns both predictions and their confidence scores.
7. The ticket information can be forwarded to the ITSM system.

### Example

**Input:**

"My laptop cannot connect to the company Wi-Fi. I have restarted it several times, but the problem continues."

**Expected AI Output:**

```json
{
  "ticket_id": "INC-1001",
  "resolver_group": "Network Operations",
  "resolver_confidence": 0.95,
  "priority": "High",
  "priority_confidence": 0.92,
  "requires_human_review": false
}
```

*Example output only. Confidence values are illustrative.*

## 4. AI Model Architecture

The project uses a multitask BERT-based classification architecture.

Instead of training two completely independent models, the system uses:

- **Shared BERT Encoder:** Understands and extracts information from ticket descriptions.
- **Resolver Classification Head:** Predicts one of the 24 IT resolver groups.
- **Priority Classification Head:** Predicts one of the 4 priority classes.

Both classification heads use the same shared text representation.

This approach allows the model to perform two related classification tasks within one architecture.

## 5. Dataset

The model was trained using a synthetic IT support ticket dataset.

| Dataset Information | Value |
|---|---:|
| Total Tickets | 10,000 |
| Resolver Groups | 24 |
| Priority Classes | 4 |
| Clients | 3 |
| Missing Values | 0 |
| Duplicate Ticket IDs | 0 |
| Unique Descriptions | 9,952 |
| Conflicting Labels | 0 |

The dataset includes different types of IT incidents, such as:

- Network connectivity problems
- Hardware failures
- Software issues
- Application errors
- Security incidents
- Account and access problems

**Note:** The dataset contains synthetic data created for model development and demonstration.

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| BERT | Natural language understanding |
| Hugging Face Transformers | Model training and implementation |
| Hugging Face Datasets | Dataset preparation |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Model evaluation |
| FastAPI | REST API development |
| Uvicorn | API server |

## 7. Project Architecture

```text
Employee
   |
   v
IT Support Ticket
   |
   v
FastAPI Application
   |
   v
BERT Shared Encoder
   |
   +----------------------+
   |                      |
   v                      v
Resolver Classifier   Priority Classifier
   |                      |
   v                      v
Resolver Group        Priority Level
   |                      |
   +----------+-----------+
              |
              v
       Confidence Scores
              |
              v
       Human Review Check
              |
              v
         JSON Response
              |
              v
        Mock ITSM System
```

## 8. API Endpoints

The FastAPI application provides endpoints for interacting with the AI system.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Checks API health |
| `/triage` | POST | Analyzes and classifies an IT ticket |

### API Documentation

Once the API is running, interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 9. Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Required Python dependencies
- Trained model files

### Step 1 – Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

### Step 2 – Navigate to the Project

```bash
cd EthanGlobal_AI_Triage_Agent
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Start the API

```bash
uvicorn src.api.main:app --reload
```

### Step 5 – Open API Documentation

Navigate to:

```text
http://127.0.0.1:8000/docs
```

You can now submit test tickets and inspect the AI predictions.

**Important:** The trained model files must be available in their configured model directories before running predictions.

## 10. Model Evaluation

The project includes model evaluation using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-class classification reports

These metrics help evaluate how effectively the model classifies resolver groups and predicts ticket priorities.

Actual evaluation results should be reported separately for each prediction task.

## 11. Project Goals

The main goals of this project are:

1. Automate IT support ticket classification.
2. Improve ticket assignment accuracy.
3. Identify high-priority incidents.
4. Reduce manual ticket triage.
5. Demonstrate AI integration with ITSM workflows.
6. Provide a foundation for future intelligent IT support automation.

## 12. Future Improvements

Potential future enhancements include:

- Integration with a real ITSM platform.
- Continuous model improvement using real ticket data.
- Advanced confidence-based human review.
- Model monitoring and performance tracking.
- Support for additional ticket categories.
- Automated ticket routing and assignment.

## 13. Author

**Charbel Fakhry**

AI Engineering | Software Engineering | Python | Machine Learning | Backend Development

---

**Project:** Ethan Global – AI-Powered IT Ticket Triage Agent

**Purpose:** Intelligent automation of IT support ticket classification and prioritization.