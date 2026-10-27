# 🛡️ DataGuard AI
### An Autonomous AI Agent for Data Quality and Intelligent Data Cleaning

DataGuard AI is an autonomous data quality application designed to help users detect, understand, and resolve data quality issues with minimal technical effort. It analyzes datasets, identifies common data problems, explains findings in simple language, and proposes corrections that users can review before applying.

Built with Python and Streamlit, the project combines structured agent workflows, data quality checks, remediation recommendations, and visual analytics to make data cleaning more transparent and accessible.

## ✨ Features

- 📂 **CSV and Excel Upload:** Analyze supported datasets through an interactive interface.
- 🔍 **Automated Data Profiling:** Inspect dataset structure and quality.
- 🚨 **Issue Detection:** Identify missing values, duplicate records, and other data inconsistencies.
- 🤖 **Agent-Based Workflow:** Organize profiling, investigation, and remediation into structured steps.
- 💬 **Simple Explanations:** Help users understand detected problems without unnecessary technical jargon.
- ✅ **Human-in-the-Loop Fixes:** Review, approve, reject, skip, or revise proposed corrections where supported.
- 📊 **Visual Analytics:** Explore data quality summaries and issue statistics.
- 🔄 **Before-and-After Comparison:** Review the effects of data cleaning operations.
- 🧾 **Audit and Version History:** Track remediation actions and review previous states where supported.
- 📥 **Export:** Download cleaned datasets and quality reports.

## 🎯 Problem Statement

Poor-quality data can affect business decisions, analytics, and machine learning results. Finding and correcting missing values, duplicates, and inconsistencies often requires manual effort and technical expertise.

DataGuard AI addresses this problem by bringing automated detection, understandable explanations, and user-reviewed corrections into one workflow.

## 💡 How It Works

1. **Upload:** The user uploads a CSV or Excel dataset.
2. **Profile:** The application inspects the dataset and its quality.
3. **Detect:** The agent identifies potential data quality issues.
4. **Explain:** Findings are presented in understandable language.
5. **Recommend:** The application proposes possible corrections.
6. **Review:** The user decides whether to apply each proposed fix.
7. **Validate:** The dataset can be checked again after remediation.
8. **Export:** The user exports the cleaned data and available quality reports.

## 🏗️ Project Architecture

```text
DataGuard-AI/
├── agent/
│   ├── auditor.py
│   ├── investigator.py
│   ├── orchestrator.py
│   ├── profiler.py
│   ├── quality_engine.py
│   ├── remediation_engine.py
│   └── session_manager.py
├── data/
│   ├── generator.py
│   ├── ground_truth.json
│   └── sample_customers.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEMO_SCRIPT.md
│   └── SPONSORS.md
├── integrations/
│   ├── meterless_engine.py
│   ├── meterless_nightshift.py
│   ├── nemotron.py
│   └── zetaris_client.py
├── models/
├── tests/
├── zetaris-platform/
├── app.py
├── requirements.txt
└── README.md
```

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Pandas | Data processing and manipulation |
| Pytest | Automated testing |
| Zetaris | Data querying and platform integration |
| Meterless | Agent workflow and state-management integration |
| NVIDIA Nemotron | Optional AI reasoning integration, subject to API configuration |

## 🤝 Sponsor Technology Integration

### Zetaris

DataGuard AI includes a Zetaris client and supporting platform configuration intended to connect SQL-based data querying with the data quality workflow. This enables the application to retrieve and inspect records through the configured data platform.

Live connectivity and query execution should be verified in the demonstration environment.

### Meterless

The project includes a Meterless integration module and a local Markovian workflow engine. The local engine supports structured state handoffs and checkpoints across agent-processing steps.

The local engine and Meterless Nightshift are separate components. Nightshift requires successful live verification before it can be presented as operational.

### NVIDIA Nemotron

The project includes an NVIDIA Nemotron integration module for API-based AI reasoning and explanations. Live inference requires a valid NVIDIA API key and successful request verification.

Until live inference has been tested, this integration should be described as implemented in code but not verified in operation.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- Git
- A terminal or command-line environment

### 1. Clone the Repository

```bash
git clone https://github.com/aimakh/DataGuard-AI.git
cd DataGuard-AI
```

### 2. Create a Virtual Environment

**Linux / WSL**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

If `.env.example` is provided, copy it to `.env` and configure the variables required by your environment.

**Linux / WSL**

```bash
cp .env.example .env
```

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

Add valid API credentials only when needed. Never commit actual API keys, passwords, or private credentials.

### 5. Start the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 🧪 Testing

Run the automated tests from the project root:

```bash
pytest -q
```

Tests involving external services may require additional configuration, credentials, or running services. Live sponsor integrations should be tested separately.

## 🔐 Responsible Data Handling

- Review proposed changes before applying them.
- Validate the output before using it for important decisions.
- Keep API keys and credentials out of source control.
- Avoid uploading confidential or sensitive data to untrusted environments.
- Treat automated recommendations as suggestions that may require human review.

## 🔮 Future Roadmap

- Expand support for additional file formats and data sources.
- Improve data quality rules and configurable validation.
- Strengthen AI-generated explanations and remediation suggestions.
- Verify and improve live sponsor integrations.
- Add more comprehensive monitoring and reporting.
- Improve deployment and collaborative review capabilities.

## 🏆 Hackathon

**Developed for the Open Agent Hackathon 2026.**

DataGuard AI explores how autonomous agents can make data quality workflows more understandable, transparent, and efficient while keeping users in control of important corrections.

## 👥 Contributors

@muhibkhan98



A license has not yet been specified. Add a license after confirming the licensing terms for the project and its third-party components.
