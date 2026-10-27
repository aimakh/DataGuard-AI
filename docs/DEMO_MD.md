# DataGuard AI — Hackathon Judging & Live Demonstration Script

Follow this 5-minute walkthrough to test and demonstrate all features of DataGuard AI.

---

## Prerequisites
1. Ensure the Python environment is activated:
   ```bash
   source .venv/bin/activate
   ```
2. Launch the Streamlit Cockpit:
   ```bash
   streamlit run app.py --server.headless=true --server.port=8501
   ```
3. Open `http://localhost:8501` in your browser.

---

## Live Demonstration Flow

### Step 1: Sponsor Integrations Inspection (Sidebar)
- Direct the judge's attention to the sidebar **Ecosystem Integrations**:
  - **Zetaris Network Platform**: Shows `ONLINE (Port 3000/8888)` with active Spark federation proxy.
  - **Meterless Nightshift**: Shows `CONNECTED (Port 8088)` local V1 runtime.
  - **Markovian Engine**: Shows `ACTIVE (O(1) Memory)` constant-memory reasoning.
  - **NVIDIA Nemotron**: Shows `REASONING READY` empirical synthesis.

### Step 2: Ingest Data Source
- Select **Benchmark Dataset (Ground Truth)** to load the 503-record benchmark containing 8 injected anomaly classes.
- (Optional) Select **Live Zetaris SQL Query** to execute a federated query directly against the containerized Zetaris platform.

### Step 3: Trigger Autonomous Reliability Audit
- Click the prominent **⚡ Run Autonomous Reliability Audit** button.
- Observe the orchestrator executing the 10-state pipeline:
  `INGEST` -> `PROFILE` -> `DETECT` -> `TRIAGE` -> `INVESTIGATE` -> `RECOMMEND` -> `VALIDATE` -> `REVIEW`.

### Step 4: Examine Health Score & Rules (Tab 1)
- Note the **Composite Quality Score**: ~`42.5 / 100` (Grade: `D-`).
- Point out the 8 deterministic rules table showing exact pass/fail criteria and weighted completeness, uniqueness, and validity scores.

### Step 5: Inspect Quality Findings (Tab 2)
- Browse the 8 defect categories: missing values, exact duplicates, invalid bounds (e.g. negative ages), inconsistent date formats (`YYYY-MM-DD` vs `MM/DD/YYYY`), unnormalized categories (`'Gold'` vs `'gold'`), and semantic contradictions (`churn_date < signup_date`).
- Expand any finding to reveal exact row indices and violated constraints.

### Step 6: Review Markovian & Nemotron Reasoning (Tab 3)
- Highlight the **Meterless Markovian Engine Telemetry**:
  - 3 execution chunks with bounded carryover state.
  - **62.5% Token Savings** vs unchunked cumulative context.
- Review the **NVIDIA Nemotron Empirical Root-Cause Synthesis**:
  - Identified upstream ETL schema divergences, unstandardized client parsers, and async race conditions.

### Step 7: Human-in-the-Loop Remediation Approval (Tab 4)
- **CRITICAL REQUIREMENT**: Show that DataGuard AI **never** modifies data automatically.
- Review the projected score delta: `42.5` -> `87.5 / 100` (`+45.0` improvement).
- Review individual proposed actions with safety levels (`SAFE`, `CAUTION`).
- Uncheck an action or keep all checked.
- Click **🚀 Authorize & Apply Selected Fixes**.

### Step 8: Cryptographic Audit Certificate & Export (Tab 5)
- Inspect the sealed **Audit Certificate**:
  - `VERIFIED PASSED` indicator.
  - Score increase confirmed.
  - Source SHA-256 vs Remediated SHA-256 cryptographic fingerprints.
- Click **📥 Download Audit Report (.md)** to view the markdown compliance record.
- Click **📥 Download Remediated Dataset (.csv)** to obtain the clean, production-ready dataset.
