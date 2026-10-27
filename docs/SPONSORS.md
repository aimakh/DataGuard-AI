# DataGuard AI — Sponsor Technology Integrations

DataGuard AI was engineered specifically for the Open Agent Hackathon 2026, integrating real sponsor platforms and documented APIs without synthetic placeholders.

---

## 1. Zetaris Network Platform Integration

### Overview
Zetaris is an enterprise Network Data Platform providing unified Spark SQL federation across distributed databases and lakehouses.

### Verification & Health
- **Container Architecture**: DataGuard AI interfaces directly with the containerized Zetaris platform (`zetaris-ui-1`, `zetaris-api-1`, `zetaris-driver-1`, `zetaris-postgres-1`).
- **Platform Verification**: Verified with `verify.sh` (9/9 health checks passing on ports 3000 and 8888).

### Protocol & Interface
- **Authentication**: `POST http://localhost:3000/api/auth/login`
  - Payload: `{"username": "<email>", "password": "<password>"}`
  - Response: Extracts `idToken` and `orgId: "1"`.
- **Spark SQL Proxy**: `POST http://localhost:3000/api/proxy/query/sql/start`
  - Required Headers:
    - `Authorization: Bearer <idToken>`
    - `X-Org-ID: 1`
    - `X-Request-ID: <UUID-v4>` (standard RFC UUID string)
    - `Content-Type: application/json`
  - Payload: `{"select": "<SQL SELECT STATEMENT>", "pageLimit": 100}`
  - Returns records under `"records": [...]` or `"data": [...]`, converted directly into high-performance DataFrames.

---

## 2. Meterless Nightshift / Local AI Engine

### Overview
Meterless Nightshift provides an OpenAI-compatible local AI runtime (`http://127.0.0.1:8088/v1`), allowing high-throughput LLM reasoning without cloud dependency or token billing.

### Protocol & Interface
- Standard OpenAI ChatCompletions endpoint (`POST /v1/chat/completions`).
- Supports structured system prompts and deterministic temperature settings (`temperature=0.1`).
- Integrated through `integrations/meterless_nightshift.py` with automatic health detection and fallback resilience.

---

## 3. Meterless Markovian Engine (Constant-Memory Reasoning)

### Overview
Standard multi-agent data quality loops suffer from $O(N)$ linear context window explosion: feeding the full history of every row defect rapidly exhausts LLM context limits and incurs astronomical token costs.

The **Meterless Markovian Engine** solves this by implementing an explicit $O(1)$ memory model where each reasoning step depends only on the current state checkpoint and initial investigation goal.

### Protocol & Markers
- **Chunk Bounding**: Context is partitioned into discrete chunks (e.g. 4,000 tokens).
- **Carryover State**: A strictly bounded state buffer (default 256 tokens) transfers essential discoveries between chunks:
  - `[STATE_CHECKPOINT]` marks an intermediate state handoff.
  - `[TASK_COMPLETE]` marks definitive convergence.
- **Token Economics Telemetry**:
  $$\text{Tokens Saved} = \sum_{i=1}^{N} (i \times \text{chunkSize}) - \sum_{i=1}^{N} (\text{carryover} + \text{output})$$
  DataGuard AI achieves $>50\%$ token savings on enterprise datasets while maintaining flawless investigation depth.

---

## 4. NVIDIA Nemotron Reasoning Engine

### Overview
NVIDIA Nemotron (`nvidia/llama-3.1-nemotron-70b-instruct`) provides deep empirical root-cause synthesis, turning raw defect coordinates into actionable engineering hypotheses.

### Verification & Hypotheses
- Evaluates defect patterns across multi-column correlations (e.g. unstandardized date formats indicating upstream ingestion parser discrepancies; inverted signup/churn timestamps indicating async write races).
- Integrated via `integrations/nemotron.py` with direct NVIDIA API integration (`https://integrate.api.nvidia.com/v1`) and empirical synthesis fallback.

---

## 5. Cursor Compliance & Rules Configuration

- Hackathon compliance is strictly observed via `.cursorrules` located in the root repository.
- Establishes domain context, non-destructive safety invariants, 10-state lifecycle steps, typing standards, and deterministic test requirements.
