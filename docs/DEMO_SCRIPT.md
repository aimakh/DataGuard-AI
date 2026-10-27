
# DataGuard AI — Live Demonstration Walkthrough (5 Minutes)

### Step 1: Ingest Dataset & Declare Objective (Sidebar)
1. Select **Preset Datasets** -> **Benchmark Dataset (Ground Truth)** or upload any CSV/Excel file.
2. Select Intended Dataset Goal: **Machine learning** (or Business reporting / Data analysis).
3. Click **⚡ Initialize Reliability Agent**.

### Step 2: Sequential Issue Review (The Core Feature)
1. Observe the top banner: `Issue 1 of 8 | 0 fixed | 0 rejected | 8 remaining`.
2. Inspect the active issue card:
   - **What I Found**: Plain English defect summary + actual evidence data.
   - **Why It Matters**: Contextual explanation for Machine Learning.
   - **My Suggested Fix**: Recommended method (e.g. Impute Missing Numeric Values with Median).
   - **Other Options**: Choose from Mean, Mode, Constant, or Row Dropping.
3. Review the **Preview Before vs After** table.
4. Click **✅ Approve & Apply Fix**:
   - Watch the score increase!
   - Active version increments to `v1`.

### Step 3: Natural Language Revision Assistant
1. In the chat box at the bottom, type:
   `"Use mean instead of median"` or `"Keep duplicates but flag them"`
2. Watch DataGuard AI interpret the request, update the proposal, and provide a fresh preview.
3. Type `"Undo last change"` to test instant version rollback!

### Step 4: Explore My Data (Visualization Studio)
1. Switch sidebar view to **📊 Explore My Data**.
2. Inspect the **Missingness by Column** and **Feature Distributions**.
3. View **🔄 Before vs After Comparisons**: select a column to see side-by-side distribution overlays showing how the approved fix shifted the data.
4. View the **📈 Reliability Score Progression** chart across `v0 -> v1 -> v2`.

### Step 5: Export Cleaned Data & Governance Audit
1. Switch to **📥 Export & Certification**.
2. Read the **"Your Dataset Is Ready for Review"** executive summary.
3. Download the Cleaned CSV, Cleaned Excel, Markdown Report, and Cryptographic SHA-256 JSON Audit Trail.
