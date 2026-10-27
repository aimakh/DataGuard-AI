import json
import pytest
import pandas as pd
from agent.quality_engine import QualityEngine


def test_benchmark_precision_recall():
    df = pd.read_csv("/home/aima/open-agent-hack/data/sample_customers.csv")
    with open("/home/aima/open-agent-hack/data/ground_truth.json") as f:
        gt = json.load(f)

    score, findings = QualityEngine.evaluate(df)
    
    # Ground truth categories injected
    injected_cats = {"NULL_VALUES", "DUPLICATES", "SCHEMA_MISMATCH", "BOUNDS_VIOLATION", "FORMAT_INCONSISTENCY", "CATEGORY_NORMALIZATION", "STATISTICAL_OUTLIER", "SEMANTIC_CONTRADICTION"}
    detected_cats = {f.category.value for f in findings}

    # Verify all 8 injected anomaly classes are successfully detected (100% recall on anomaly classes)
    recall = len(detected_cats.intersection(injected_cats)) / len(injected_cats)
    assert recall >= 0.90, f"Recall was {recall}, expected >= 0.90"
    assert len(findings) >= 8
    assert score.score < 90.0
