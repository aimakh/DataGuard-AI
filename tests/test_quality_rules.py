import pytest
import pandas as pd
import numpy as np
from agent.quality_engine import QualityEngine
from models.findings import QualityCategory, Severity


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "email": ["a@example.com", "b@example.com", "c@example.com", "d@example.com", "e@example.com"],
        "status": ["Active", "Active", "Active", "Active", "Active"],
        "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "spend": [100.0, 150.0, 200.0, 250.0, 300.0]
    })


def test_clean_data_passes(clean_df):
    score, findings = QualityEngine.evaluate(clean_df)
    assert score.score >= 95.0
    assert score.critical_count == 0
    assert len(findings) == 0


def test_null_detection():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "val": [10.0, np.nan, -999.0, 40.0],
        "txt": ["hello", "N/A", "world", "none"]
    })
    res, findings = QualityEngine.check_null_values(df)
    assert not res.passed
    assert res.category == QualityCategory.NULL_VALUES
    assert len(findings) == 2


def test_duplicate_detection():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"]
    })
    res, findings = QualityEngine.check_duplicate_records(df)
    assert not res.passed
    assert any(f.category == QualityCategory.DUPLICATES for f in findings)


def test_schema_type_mismatch():
    df = pd.DataFrame({
        "price": ["10.5", "20.0", "30.5", "ERR_INVALID", "50.0", "60.0", "70.0", "80.0", "90.0", "100.0"]
    })
    res, findings = QualityEngine.check_schema_and_types(df)
    assert not res.passed
    assert len(findings) == 1
    assert findings[0].affected_columns == ["price"]


def test_bounds_violations():
    df = pd.DataFrame({
        "age": [25, -5, 30, 150],
        "churn_rate_pct": [10.0, 20.0, 120.0, 50.0]
    })
    res, findings = QualityEngine.check_bounds_violations(df)
    assert not res.passed
    assert len(findings) == 2


def test_format_inconsistencies():
    df = pd.DataFrame({
        "signup_date": ["2023-01-01", "2023-02-01", "03/15/2023", "2023-04-01"],
        "email": ["user@test.com", "bad_email_at_domain", "good@test.org", "another@test.co"]
    })
    res, findings = QualityEngine.check_format_inconsistencies(df)
    assert not res.passed
    assert len(findings) == 2


def test_category_normalization():
    df = pd.DataFrame({
        "status": ["Active", "active", "Active", "ACTIVE ", "Churned", "Active"]
    })
    res, findings = QualityEngine.check_category_normalization(df)
    assert not res.passed
    assert len(findings) == 1
    assert findings[0].category == QualityCategory.CATEGORY_NORMALIZATION


def test_statistical_outlier():
    # 50 continuous items and 1 extreme outlier
    values = [float(i) for i in range(50)] + [1500.0]
    df = pd.DataFrame({"metric": values})
    res, findings = QualityEngine.check_statistical_outliers(df)
    assert not res.passed
    assert len(findings) == 1
    assert findings[0].affected_rows_count == 1


def test_semantic_contradictions():
    df = pd.DataFrame({
        "signup_date": ["2024-05-01", "2024-01-01"],
        "churn_date": ["2023-01-01", "2024-06-01"],
        "status": ["Active", "Active"],
        "cancellation_date": ["2024-03-01", None]
    })
    res, findings = QualityEngine.check_semantic_contradictions(df)
    assert not res.passed
    assert any(f.severity == Severity.CRITICAL for f in findings)


def test_sample_customers_benchmark():
    df = pd.read_csv("/home/aima/open-agent-hack/data/sample_customers.csv")
    score, findings = QualityEngine.evaluate(df)
    assert len(findings) >= 8
    assert 0.0 < score.score < 90.0
    assert score.critical_count > 0
