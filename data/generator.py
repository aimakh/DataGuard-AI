import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class SyntheticDataGenerator:
    """Generates realistic customer telemetry data with controlled, ground-truth-labeled anomalies."""

    @classmethod
    def generate(cls, n_rows: int = 500, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)

        ground_truth: Dict[str, Any] = {
            "total_rows": n_rows,
            "injected_anomalies": {}
        }

        # 1. Base clean attributes
        customer_ids = [f"CUST-{1000 + i}" for i in range(n_rows)]
        first_names = ["Emma", "Liam", "Olivia", "Noah", "Sophia", "Jackson", "Ava", "Aiden", "Isabella", "Lucas"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        
        names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_rows)]
        emails = [f"{n.lower().replace(' ', '.')}{random.randint(10, 99)}@example.com" for n in names]
        ages = [int(np.random.normal(38, 12)) for _ in range(n_rows)]
        ages = [max(18, min(80, a)) for a in ages]
        
        incomes = [round(float(np.random.exponential(55000) + 25000), 2) for _ in range(n_rows)]
        monthly_spends = [round(float(np.random.gamma(4, 50)), 2) for _ in range(n_rows)]
        membership_tiers = [random.choice(["Bronze", "Silver", "Gold", "Platinum"]) for _ in range(n_rows)]
        statuses = [random.choice(["Active", "Active", "Active", "Churned"]) for _ in range(n_rows)]
        churn_risks = [round(random.uniform(5.0, 95.0), 1) for _ in range(n_rows)]
        annual_orders = [int(np.random.poisson(12) + 1) for _ in range(n_rows)]

        # Base dates
        base_date = pd.Timestamp("2023-01-01")
        signup_dates = [(base_date + pd.Timedelta(days=random.randint(0, 600))).strftime("%Y-%m-%d") for _ in range(n_rows)]
        churn_dates = []
        for i in range(n_rows):
            if statuses[i] == "Churned":
                s_dt = pd.to_datetime(signup_dates[i])
                churn_dt = s_dt + pd.Timedelta(days=random.randint(30, 300))
                churn_dates.append(churn_dt.strftime("%Y-%m-%d"))
            else:
                churn_dates.append(None)

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "full_name": names,
            "email": emails,
            "age": ages,
            "annual_income": incomes,
            "monthly_spend": [str(x) for x in monthly_spends],
            "membership_tier": membership_tiers,
            "status": statuses,
            "churn_risk_score": churn_risks,
            "annual_orders": annual_orders,
            "signup_date": signup_dates,
            "churn_date": churn_dates
        })

        # --- Injected Anomalies ---

        # 1. NULLs & Sentinels (annual_income)
        null_income_idx = [12, 45, 78, 110, 215]
        for idx in null_income_idx:
            df.at[idx, "annual_income"] = np.nan
        sentinel_income_idx = [150, 290, 340]
        for idx in sentinel_income_idx:
            df.at[idx, "annual_income"] = -999.0
        ground_truth["injected_anomalies"]["NULL_VALUES_INCOME"] = {
            "column": "annual_income",
            "indices": null_income_idx + sentinel_income_idx,
            "count": len(null_income_idx) + len(sentinel_income_idx)
        }

        # 2. Type contamination (monthly_spend)
        type_corrupt_idx = [22, 134, 255, 388]
        for idx in type_corrupt_idx:
            df.at[idx, "monthly_spend"] = random.choice(["UNKNOWN", "ERROR_VAL", "PENDING_REV"])
        ground_truth["injected_anomalies"]["SCHEMA_MISMATCH_SPEND"] = {
            "column": "monthly_spend",
            "indices": type_corrupt_idx,
            "count": len(type_corrupt_idx)
        }

        # 3. Bounds violations (age & churn_risk_score)
        bnd_age_idx = [50, 160, 420]
        df.at[50, "age"] = -4
        df.at[160, "age"] = 185
        df.at[420, "age"] = -1
        bnd_risk_idx = [77, 305]
        df.at[77, "churn_risk_score"] = 145.0
        df.at[305, "churn_risk_score"] = -15.5
        ground_truth["injected_anomalies"]["BOUNDS_VIOLATION"] = {
            "columns": ["age", "churn_risk_score"],
            "indices": bnd_age_idx + bnd_risk_idx,
            "count": len(bnd_age_idx) + len(bnd_risk_idx)
        }

        # 4. Format Inconsistency (signup_date & email)
        fmt_date_idx = [35, 120, 280, 410]
        for idx in fmt_date_idx:
            dt = pd.to_datetime(df.at[idx, "signup_date"])
            df.at[idx, "signup_date"] = dt.strftime("%m/%d/%Y")
        fmt_email_idx = [60, 199, 350]
        df.at[60, "email"] = "invalid_user_without_at_symbol.com"
        df.at[199, "email"] = "missing_domain@"
        df.at[350, "email"] = "user@@doubleat..com"
        ground_truth["injected_anomalies"]["FORMAT_INCONSISTENCY"] = {
            "columns": ["signup_date", "email"],
            "indices": fmt_date_idx + fmt_email_idx,
            "count": len(fmt_date_idx) + len(fmt_email_idx)
        }

        # 5. Category Normalization (membership_tier)
        cat_idx = [15, 88, 140, 275, 430]
        df.at[15, "membership_tier"] = "gold"
        df.at[88, "membership_tier"] = "GOLD "
        df.at[140, "membership_tier"] = "silver"
        df.at[275, "membership_tier"] = "Platinum "
        df.at[430, "membership_tier"] = "bronze"
        ground_truth["injected_anomalies"]["CATEGORY_NORMALIZATION"] = {
            "column": "membership_tier",
            "indices": cat_idx,
            "count": len(cat_idx)
        }

        # 6. Statistical Outlier (annual_orders)
        outlier_idx = [95, 320]
        df.at[95, "annual_orders"] = 1250
        df.at[320, "annual_orders"] = 890
        ground_truth["injected_anomalies"]["STATISTICAL_OUTLIER"] = {
            "column": "annual_orders",
            "indices": outlier_idx,
            "count": len(outlier_idx)
        }

        # 7. Semantic Contradiction (temporal inversion & active with churn date)
        sem_temp_idx = [40, 175, 390]
        for idx in sem_temp_idx:
            df.at[idx, "status"] = "Churned"
            df.at[idx, "signup_date"] = "2024-06-01"
            df.at[idx, "churn_date"] = "2023-01-15"
        sem_status_idx = [105, 240, 365, 480]
        for idx in sem_status_idx:
            df.at[idx, "status"] = "Active"
            df.at[idx, "churn_date"] = "2024-04-10"
        ground_truth["injected_anomalies"]["SEMANTIC_CONTRADICTION"] = {
            "columns": ["signup_date", "churn_date", "status"],
            "indices": sem_temp_idx + sem_status_idx,
            "count": len(sem_temp_idx) + len(sem_status_idx)
        }

        # 8. Duplicate rows
        dup_rows = df.iloc[[10, 25, 65]].copy()
        df = pd.concat([df, dup_rows], ignore_index=True)
        ground_truth["injected_anomalies"]["DUPLICATES"] = {
            "count": 3,
            "indices": list(range(n_rows, n_rows + 3))
        }

        return df, ground_truth


if __name__ == "__main__":
    out_dir = Path("/home/aima/open-agent-hack/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df, gt = SyntheticDataGenerator.generate(n_rows=500)
    
    csv_path = out_dir / "sample_customers.csv"
    gt_path = out_dir / "ground_truth.json"
    
    df.to_csv(csv_path, index=False)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
        
    print(f"Generated {len(df)} rows into {csv_path}")
    print(f"Saved ground truth metadata to {gt_path}")
