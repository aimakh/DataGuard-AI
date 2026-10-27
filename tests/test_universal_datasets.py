import io
import pytest
import pandas as pd
import numpy as np

from agent.profiler import DatasetProfiler
from agent.quality_engine import QualityEngine
from agent.remediation_engine import RemediationEngine
from agent.orchestrator import DataGuardOrchestrator
from agent.auditor import Auditor
from models.state import AgentStep, WorkflowStatus
from models.findings import QualityCategory, Severity


# ==============================================================================
# 1. FILE INGESTION: CSV, EXCEL, AND EDGE CASES
# ==============================================================================

def test_csv_and_excel_ingestion(tmp_path):
    df_sample = pd.DataFrame({
        "sku": ["SKU-001", "SKU-002", "SKU-003"],
        "price": [19.99, 29.50, 45.00],
        "in_stock": [True, False, True]
    })
    
    # 1. Test CSV loading
    csv_file = tmp_path / "products.csv"
    df_sample.to_csv(csv_file, index=False)
    loaded_csv, meta_csv = DatasetProfiler.load_dataset(csv_file)
    assert len(loaded_csv) == 3
    assert list(loaded_csv.columns) == ["sku", "price", "in_stock"]
    assert meta_csv["format"] == "csv"
    assert meta_csv["rows"] == 3

    # 2. Test Excel loading (.xlsx)
    xlsx_file = tmp_path / "products.xlsx"
    df_sample.to_excel(xlsx_file, index=False, engine="openpyxl")
    loaded_xlsx, meta_xlsx = DatasetProfiler.load_dataset(xlsx_file)
    assert len(loaded_xlsx) == 3
    assert list(loaded_xlsx.columns) == ["sku", "price", "in_stock"]
    assert meta_xlsx["format"] == "excel"

    # 3. Test BytesIO buffer loading
    buf = io.BytesIO()
    df_sample.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    loaded_buf, meta_buf = DatasetProfiler.load_dataset(buf, file_name="buffer_products.xlsx")
    assert len(loaded_buf) == 3
    assert meta_buf["format"] == "excel"


def test_malformed_and_empty_files_handled_gracefully(tmp_path):
    # Empty CSV
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse|empty|no columns"):
        DatasetProfiler.load_dataset(empty_csv)

    # Corrupted encoding fallback (Latin-1 chars)
    latin1_csv = tmp_path / "latin1.csv"
    latin1_csv.write_bytes("id,city\n1,München\n2,São Paulo\n".encode("latin-1"))
    df_latin1, meta = DatasetProfiler.load_dataset(latin1_csv)
    assert len(df_latin1) == 2
    assert "city" in df_latin1.columns


# ==============================================================================
# 2. UNIVERSAL DATASET 1: RETAIL E-COMMERCE TRANSACTIONS
# ==============================================================================

def test_retail_ecommerce_quality_and_remediation():
    """Tests an entirely unrelated retail e-commerce schema with 7 diverse defects."""
    retail_data = pd.DataFrame({
        "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104", "ORD-105", "ORD-105", "ORD-106"],
        "product_sku": ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E", "SKU-E", "SKU-F"],
        "customer_email": ["john@example.com", "sarah@store.org", "invalid-email-address", "mike@co.uk", "lisa@corp.com", "lisa@corp.com", "alex@shop.net"],
        "quantity": [2, -5, 1, 3, 10, 10, 4],  # -5 is a bounds violation
        "unit_price": [15.0, 25.0, "corrupted_price", 10.0, 50.0, 50.0, 30.0],  # Schema mismatch
        "order_date": ["2024-01-10", "2024-01-12", "01/15/2024", "2024-01-20", "2024-01-25", "2024-01-25", "2024-01-28"],  # Mixed formats
        "delivery_date": ["2024-01-15", "2024-01-18", "2024-01-20", "2024-01-10", "2024-01-30", "2024-01-30", "2024-02-02"], # Inversion on row 3 (Jan 20 vs Jan 10)
        "return_status": ["Completed", "Pending", "COMPLETED", "completed ", "Pending", "Pending", None]  # Category variants + null
    })

    orch = DataGuardOrchestrator(dataset_name="retail_ecommerce")
    ctx = orch.run_lifecycle_to_review(retail_data)

    assert ctx.current_step == AgentStep.REVIEW
    assert ctx.status == WorkflowStatus.WAITING_APPROVAL
    assert ctx.initial_score.score < 90.0
    
    # Verify specific findings detected on retail schema
    finding_rules = {f.rule_name for f in ctx.findings}
    assert "Exact Duplicate Record Check" in finding_rules or "Primary Key Collision Check" in finding_rules
    assert "Domain Range Violation" in finding_rules  # negative quantity
    assert "Schema Type Contamination" in finding_rules  # corrupted price
    assert "Date Format Inconsistency" in finding_rules
    assert "Categorical Variant Normalization" in finding_rules
    assert "Temporal Chronology Inversion" in finding_rules

    # Approve all actions
    all_action_ids = [a.action_id for a in orch.proposal.actions]
    working_df, audit = orch.apply_approved_and_audit(all_action_ids)

    assert audit.verification_passed is True
    assert audit.after_score > audit.before_score
    # Verify original df was not mutated
    assert retail_data["quantity"].iloc[1] == -5
    assert len(retail_data) == 7


# ==============================================================================
# 3. UNIVERSAL DATASET 2: HEALTHCARE PATIENT VITALS & CUSTOM RULES
# ==============================================================================

def test_healthcare_vitals_with_custom_constraints():
    """Tests clinical vitals schema with custom bounds and relational comparison rules."""
    vitals_data = pd.DataFrame({
        "patient_id": [f"PT-{i:03d}" for i in range(1, 21)],
        "systolic_bp": [120, 115, 130, 290, 110, 125, 140, 135, 118, 122, 124, 119, 131, 128, 116, 121, 123, 117, 126, 120],  # 290 is out of bounds
        "diastolic_bp": [80, 75, 85, 90, 70, 80, 88, 82, 78, 80, 82, 76, 84, 80, 75, 79, 81, 77, 83, 150],  # Row 19: diastolic 150 > systolic 120 (contradiction)
        "heart_rate": [72, 68, 75, 80, 65, 70, 78, 74, 71, 69, 73, 70, 76, 72, 67, 71, 72, 68, 74, -10],  # -10 is invalid
        "blood_type": ["A+", "O+", "B+", "A+", "O-", "AB+", "A+", "O+", "B-", "A-", "A+", "O+", "B+", "A+", "O+", "AB-", "A+", "O+", "B+", "O+"]
    })

    orch = DataGuardOrchestrator(dataset_name="patient_vitals")
    ctx = orch.run_lifecycle_to_review(
        vitals_data,
        custom_bounds={"systolic_bp": (50, 240), "heart_rate": (30, 220)},
        custom_rules=[{"type": "comparison", "left": "systolic_bp", "op": ">", "right": "diastolic_bp", "title": "Systolic > Diastolic Relation"}]
    )

    assert ctx.current_step == AgentStep.REVIEW
    assert any("systolic_bp" in f.affected_columns for f in ctx.findings)
    assert any("Custom Relational Constraint" in f.rule_name or "Systolic" in f.rule_name for f in ctx.findings)


# ==============================================================================
# 4. UNIVERSAL DATASET 3: IOT SENSOR TELEMETRY & STATISTICAL OUTLIERS
# ==============================================================================

def test_iot_telemetry_outliers_and_selective_approval():
    """Verifies that statistical outliers are flagged as warnings and selective approval is strictly respected."""
    np.random.seed(42)
    normal_temp = np.random.normal(loc=22.0, scale=1.5, size=50).tolist()
    normal_temp.extend([98.5, -45.0])  # Extreme sensor spikes
    
    telemetry_data = pd.DataFrame({
        "device_id": [f"DEV-{i:03d}" for i in range(52)],
        "temperature_c": normal_temp,
        "humidity_pct": [45.0] * 52,
        "pressure_kpa": [101.3] * 52
    })

    orch = DataGuardOrchestrator(dataset_name="iot_telemetry")
    ctx = orch.run_lifecycle_to_review(telemetry_data)

    outlier_findings = [f for f in ctx.findings if f.category == QualityCategory.STATISTICAL_OUTLIER]
    assert len(outlier_findings) > 0
    assert outlier_findings[0].severity == Severity.LOW

    # Selective Approval: User chooses NOT to approve outlier capping (preserving raw readings)
    outlier_action_ids = [a.action_id for a in orch.proposal.actions if "OUTLIER" in a.action_id]
    non_outlier_action_ids = [a.action_id for a in orch.proposal.actions if "OUTLIER" not in a.action_id]

    working_df, audit = orch.apply_approved_and_audit(non_outlier_action_ids)

    # Verify that unapproved outlier action was NOT applied (raw 98.5 temperature remains intact)
    assert 98.5 in working_df["temperature_c"].values
    assert audit.verification_passed is True


# ==============================================================================
# 5. DATA EXPORT VERIFICATION (CSV & EXCEL)
# ==============================================================================

def test_remediated_dataset_export_formats():
    df_raw = pd.DataFrame({"item": ["A", "B", "B"], "qty": [10, 20, 20]})
    orch = DataGuardOrchestrator(dataset_name="export_test")
    orch.run_lifecycle_to_review(df_raw)
    orch.apply_approved_and_audit([orch.proposal.actions[0].action_id])

    # 1. Export as CSV
    csv_bytes = orch.export_dataset("csv")
    assert isinstance(csv_bytes, bytes)
    df_reloaded_csv = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df_reloaded_csv) == 2  # Deduplicated

    # 2. Export as Excel (.xlsx)
    xlsx_bytes = orch.export_dataset("excel")
    assert isinstance(xlsx_bytes, bytes)
    df_reloaded_xlsx = pd.read_excel(io.BytesIO(xlsx_bytes), engine="openpyxl")
    assert len(df_reloaded_xlsx) == 2
