import pytest
import pandas as pd
import numpy as np

from agent.session_manager import SessionManager
from models.remediation import FixType, RemediationAction
from models.findings import QualityCategory, Severity


@pytest.fixture
def sample_dataset():
    """Generates a multi-issue dataset with missing values, duplicates, out-of-bounds, and unnormalized categories."""
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C3", "C5", "C6", "C7", "C8"],
        "age": [25.0, np.nan, -5.0, -5.0, 42.0, 38.0, 50.0, 60.0],
        "tier": ["Gold", "Silver", "gold", "gold", "Platinum", "Silver", "Gold", "Silver"],
        "signup_date": ["2023-01-15", "02/20/2023", "2023-03-10", "2023-03-10", "2023-04-05", "2023-05-12", "2023-06-01", "2023-07-04"],
        "revenue": [100.0, 250.0, 300.0, 300.0, 500.0, 150.0, 200.0, 450.0]
    })


def test_session_initialization(sample_dataset):
    session = SessionManager()
    init_res = session.initialize_session(
        sample_dataset,
        dataset_name="TestCustomers",
        user_goal="machine_learning"
    )

    assert init_res["total_issues"] > 0
    assert session.current_version_idx == 0
    assert len(session.versions) == 1
    assert session.versions[0]["version_id"] == "v0"
    assert session.user_goal == "machine_learning"

    # Invariant: Original dataframe must be preserved identically
    assert len(session.original_df) == len(sample_dataset)
    assert (session.original_df["age"].isna().sum()) == 1


def test_sequential_workflow_approve_and_immutability(sample_dataset):
    session = SessionManager()
    session.initialize_session(sample_dataset, dataset_name="TestCustomers", user_goal="data_analysis")

    original_age_nulls = session.original_df["age"].isna().sum()
    curr_issue = session.get_current_issue()
    assert curr_issue is not None
    assert curr_issue["status"] == "PENDING"
    assert "why_it_matters" in curr_issue["action"].dict()

    # Generate preview
    preview = session.preview_issue_change()
    assert "affected_rows_count" in preview
    assert preview["method"] != ""

    # Approve fix
    appr_res = session.approve_and_apply()
    assert appr_res["success"] is True
    assert appr_res["new_version"] == "v1"
    assert len(session.versions) == 2

    # Invariant: Original DF must never change!
    assert session.original_df["age"].isna().sum() == original_age_nulls
    # Active working copy is modified
    assert session.current_version_idx == 1


def test_reject_and_skip_behavior(sample_dataset):
    session = SessionManager()
    session.initialize_session(sample_dataset, dataset_name="TestCustomers")

    first_issue = session.get_current_issue()
    first_fnd_id = first_issue["finding"].finding_id

    # Reject first issue
    rej_res = session.reject_current_issue(reason="Deliberately keeping original values")
    assert rej_res["success"] is True
    assert first_issue["status"] == "REJECTED"

    # Check next issue is now current
    second_issue = session.get_current_issue()
    assert second_issue is not None
    assert second_issue["finding"].finding_id != first_fnd_id

    # Skip second issue
    skip_res = session.skip_current_issue()
    assert skip_res["success"] is True
    assert second_issue["status"] == "SKIPPED"


def test_natural_language_revisions(sample_dataset):
    session = SessionManager()
    session.initialize_session(sample_dataset, dataset_name="TestCustomers")

    # Ask Why
    why_res = session.parse_natural_language_revision("Why did you recommend this?")
    assert why_res["action_taken"] == "EXPLAIN"
    assert "Why DataGuard AI recommended this" in why_res["response"]

    # Natural Language Preference: "use mean instead of median"
    pref_res = session.parse_natural_language_revision("Use mean instead of median")
    assert pref_res["action_taken"] == "PREFERENCE_UPDATED"
    curr = session.get_current_issue()
    assert curr["action"].fix_type == FixType.IMPUTE_MEAN

    # Natural Language: "flag duplicates"
    flag_res = session.parse_natural_language_revision("keep duplicates but flag them")
    assert flag_res["action_taken"] == "PREFERENCE_UPDATED"
    assert curr["action"].fix_type == FixType.FLAG_DUPLICATES


def test_version_history_and_rollback(sample_dataset):
    session = SessionManager()
    session.initialize_session(sample_dataset, dataset_name="TestCustomers")

    # Apply fix 1 -> v1
    res1 = session.approve_and_apply()
    assert res1["new_version"] == "v1"

    # Apply fix 2 -> v2
    res2 = session.approve_and_apply()
    assert res2["new_version"] == "v2"

    assert len(session.versions) == 3
    assert session.current_version_idx == 2

    # Rollback to v1
    rb_res = session.rollback_to_version(1)
    assert rb_res["success"] is True
    assert session.current_version_idx == 1
    assert rb_res["version_id"] == "v1"

    # Rollback via natural language "undo"
    undo_res = session.parse_natural_language_revision("undo last change")
    assert undo_res["action_taken"] == "ROLLBACK"
    assert session.current_version_idx == 0


def test_final_report_and_audit(sample_dataset):
    session = SessionManager()
    session.initialize_session(sample_dataset, dataset_name="TestCustomers", user_goal="business_reporting")

    # Apply one fix
    session.approve_and_apply()

    report = session.generate_final_report_and_audit()
    assert "audit_record" in report
    assert "summary_markdown" in report
    assert report["audit_record"].content_hash_before != ""
    assert report["audit_record"].content_hash_after != ""
    assert "Reliability Score Transformation" in report["summary_markdown"]
    assert report["progress"]["fixed"] == 1
