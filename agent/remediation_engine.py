import re
import numpy as np
import pandas as pd
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from models.findings import Finding, QualityScore, Severity, QualityCategory
from models.remediation import RemediationAction, RemediationProposal, FixType
from agent.quality_engine import QualityEngine


class RemediationEngine:
    """
    Intelligent Data Quality Remediation Engine.
    Generates parameterized, reversible, and validated preprocessing proposals
    with contextualized justifications and alternative techniques.
    """

    @classmethod
    def generate_proposals(
        cls,
        findings: List[Finding],
        df: pd.DataFrame,
        user_goal: str = "data_analysis",
        target_col: Optional[str] = None
    ) -> RemediationProposal:
        actions: List[RemediationAction] = []

        for fnd in findings:
            col = fnd.affected_columns[0] if fnd.affected_columns else None
            
            # 1. NULL VALUES
            if fnd.category == QualityCategory.NULL_VALUES and col and col in df.columns:
                s = df[col]
                is_num = pd.api.types.is_numeric_dtype(s)
                if is_num:
                    s_clean = s[s.notna() & (s != -999.0)]
                    median_val = float(s_clean.median()) if not s_clean.empty else 0.0
                    mean_val = float(s_clean.mean()) if not s_clean.empty else 0.0

                    alts = [
                        {"fix_type": FixType.IMPUTE_MEAN, "label": "Mean Imputation", "description": f"Fill with column mean ({round(mean_val, 2)})", "parameters": {"replacement_value": mean_val}},
                        {"fix_type": FixType.IMPUTE_CONSTANT, "label": "Constant (0.0)", "description": "Fill missing entries with 0.0", "parameters": {"replacement_value": 0.0}},
                        {"fix_type": FixType.DROP_NULL_ROWS, "label": "Drop Null Rows", "description": f"Drop {fnd.affected_rows_count} rows with missing {col}", "parameters": {}},
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Leave As-Is", "description": "Retain missing values unchanged", "parameters": {}}
                    ]

                    actions.append(RemediationAction(
                        action_id=f"ACT-IMP-MED-{col}",
                        finding_id=fnd.finding_id,
                        title=f"Impute Missing Numeric Values in '{col}'",
                        description=f"Replace NaN and sentinel nulls in '{col}' with column median ({round(median_val, 2)}).",
                        fix_type=FixType.IMPUTE_MEDIAN,
                        target_columns=[col],
                        parameters={"replacement_value": median_val, "sentinels": list(QualityEngine.SENTINEL_NULLS)},
                        estimated_impact_rows=fnd.affected_rows_count,
                        safety_level="SAFE",
                        requires_approval=True,
                        why_it_matters=f"Missing values in '{col}' disrupt statistical calculations and cause ML pipelines to fail.",
                        risks_assumptions="Assumes Missing At Random (MAR). Median is robust against skewed distributions.",
                        alternatives=alts
                    ))
                else:
                    mode_val = s.mode().iloc[0] if not s.mode().empty else "Unknown"
                    alts = [
                        {"fix_type": FixType.IMPUTE_CONSTANT, "label": "'Unknown' Category", "description": "Fill with 'Unknown' placeholder", "parameters": {"replacement_value": "Unknown"}},
                        {"fix_type": FixType.DROP_NULL_ROWS, "label": "Drop Null Rows", "description": f"Drop {fnd.affected_rows_count} rows with missing {col}", "parameters": {}},
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Leave As-Is", "description": "Retain missing values unchanged", "parameters": {}}
                    ]
                    actions.append(RemediationAction(
                        action_id=f"ACT-IMP-MODE-{col}",
                        finding_id=fnd.finding_id,
                        title=f"Impute Missing Categories in '{col}'",
                        description=f"Replace missing entries in '{col}' with mode category ('{mode_val}').",
                        fix_type=FixType.IMPUTE_MODE,
                        target_columns=[col],
                        parameters={"replacement_value": str(mode_val), "sentinels": list(QualityEngine.SENTINEL_NULLS)},
                        estimated_impact_rows=fnd.affected_rows_count,
                        safety_level="SAFE",
                        requires_approval=True,
                        why_it_matters=f"Missing categorical values fragment groupings and lead to unassigned segments.",
                        risks_assumptions="Assumes most frequent category is the most probable replacement.",
                        alternatives=alts
                    ))

            # 2. DUPLICATES
            elif fnd.category == QualityCategory.DUPLICATES:
                alts = [
                    {"fix_type": FixType.FLAG_DUPLICATES, "label": "Flag Duplicates", "description": "Keep rows but append 'is_duplicate' boolean column", "parameters": {"subset": fnd.affected_columns if len(fnd.affected_columns) == 1 else None}},
                    {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep All", "description": "Treat duplicates as valid repeated business events", "parameters": {}}
                ]
                actions.append(RemediationAction(
                    action_id="ACT-DEDUP",
                    finding_id=fnd.finding_id,
                    title="Deduplicate Redundant Records",
                    description="Drop redundant duplicate rows, preserving first occurrence.",
                    fix_type=FixType.DROP_DUPLICATES,
                    target_columns=fnd.affected_columns,
                    parameters={"subset": fnd.affected_columns if len(fnd.affected_columns) == 1 else None},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="SAFE",
                    requires_approval=True,
                    why_it_matters="Duplicate rows artificially inflate sample sizes, double-count metrics, and cause data leakage.",
                    risks_assumptions="Preserves first occurrence; verify if duplicates represent distinct chronological transactions.",
                    alternatives=alts
                ))

            # 3. SCHEMA MISMATCH
            elif fnd.category == QualityCategory.SCHEMA_MISMATCH and col:
                s_numeric = pd.to_numeric(df[col], errors="coerce")
                median_val = float(s_numeric.dropna().median()) if not s_numeric.dropna().empty else 0.0
                actions.append(RemediationAction(
                    action_id=f"ACT-SCHEMA-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Coerce Schema & Repair Corrupted Text in '{col}'",
                    description=f"Cast '{col}' to float and replace non-numeric strings with median ({round(median_val, 2)}).",
                    fix_type=FixType.IMPUTE_MEDIAN,
                    target_columns=[col],
                    parameters={"replacement_value": median_val, "coerce_numeric": True},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="SAFE",
                    requires_approval=True,
                    why_it_matters=f"Non-numeric values in numerical column '{col}' break mathematical calculations.",
                    risks_assumptions="Replaces corrupted text strings with median value.",
                    alternatives=[
                        {"fix_type": FixType.DROP_NULL_ROWS, "label": "Drop Corrupted Rows", "description": "Remove rows with non-numeric strings", "parameters": {}},
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Leave As-Is", "description": "Keep non-numeric strings", "parameters": {}}
                    ]
                ))

            # 4. BOUNDS VIOLATION
            elif fnd.category == QualityCategory.BOUNDS_VIOLATION and col:
                min_b = fnd.evidence.get("min_allowed")
                max_b = fnd.evidence.get("max_allowed")
                actions.append(RemediationAction(
                    action_id=f"ACT-CLAMP-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Clamp Out-of-Bounds Values in '{col}'",
                    description=f"Clamp values in '{col}' to allowed domain interval [{min_b}, {max_b}].",
                    fix_type=FixType.CLAMP_BOUNDS,
                    target_columns=[col],
                    parameters={"min": min_b, "max": max_b},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="SAFE",
                    requires_approval=True,
                    why_it_matters=f"Negative or out-of-interval values in '{col}' violate physical/business constraints.",
                    risks_assumptions=f"Clamps out-of-bound values to boundary limits [{min_b}, {max_b}].",
                    alternatives=[
                        {"fix_type": FixType.REMOVE_OUTLIERS, "label": "Drop Invalid Rows", "description": "Delete records with invalid values", "parameters": {"min": min_b, "max": max_b}},
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Leave As-Is", "description": "Preserve extreme values", "parameters": {}}
                    ]
                ))

            # 5. FORMAT INCONSISTENCY
            elif fnd.category == QualityCategory.FORMAT_INCONSISTENCY and col:
                if any(k in col.lower() for k in ["date", "time", "signup"]):
                    actions.append(RemediationAction(
                        action_id=f"ACT-FMT-DATE-{col}",
                        finding_id=fnd.finding_id,
                        title=f"Standardize Date Format in '{col}'",
                        description=f"Parse mixed date representations in '{col}' into ISO 8601 (YYYY-MM-DD).",
                        fix_type=FixType.STANDARDIZE_DATE,
                        target_columns=[col],
                        parameters={"target_format": "%Y-%m-%d"},
                        estimated_impact_rows=fnd.affected_rows_count,
                        safety_level="SAFE",
                        requires_approval=True,
                        why_it_matters=f"Mixed date strings in '{col}' prevent chronological ordering and time-series aggregation.",
                        risks_assumptions="Parses ambiguous dates using day-first heuristic fallback.",
                        alternatives=[
                            {"fix_type": FixType.STANDARDIZE_DATE, "label": "DD-MM-YYYY Format", "description": "Format as Day-Month-Year", "parameters": {"target_format": "%d-%m-%Y"}},
                            {"fix_type": FixType.STANDARDIZE_DATE, "label": "MM/DD/YYYY Format", "description": "Format as Month/Day/Year", "parameters": {"target_format": "%m/%d/%Y"}}
                        ]
                    ))
                elif "email" in col.lower():
                    actions.append(RemediationAction(
                        action_id=f"ACT-FMT-EMAIL-{col}",
                        finding_id=fnd.finding_id,
                        title=f"Sanitize Malformed Emails in '{col}'",
                        description=f"Replace unparseable email strings in '{col}' with standardized placeholder.",
                        fix_type=FixType.IMPUTE_CONSTANT,
                        target_columns=[col],
                        parameters={"placeholder": "unverified@example.com"},
                        estimated_impact_rows=fnd.affected_rows_count,
                        safety_level="CAUTION",
                        requires_approval=True,
                        why_it_matters="Malformed email syntax leads to bounce-backs and invalid communication records.",
                        risks_assumptions="Flags invalid emails with placeholder.",
                        alternatives=[
                            {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep As-Is", "description": "Do not modify email column", "parameters": {}}
                        ]
                    ))

            # 6. CATEGORY NORMALIZATION
            elif fnd.category == QualityCategory.CATEGORY_NORMALIZATION and col:
                split_variants = fnd.evidence.get("split_variants", {})
                canonical_map = {}
                for norm, variants in split_variants.items():
                    canonical = sorted(variants, key=lambda x: (len(x.strip()), x.strip()[0].isupper()), reverse=True)[0].strip()
                    for v in variants:
                        canonical_map[v] = canonical
                        
                actions.append(RemediationAction(
                    action_id=f"ACT-CAT-NORM-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Normalize Categorical Variants in '{col}'",
                    description=f"Map whitespace/casing variations in '{col}' to canonical values ({canonical_map}).",
                    fix_type=FixType.NORMALIZE_CATEGORY,
                    target_columns=[col],
                    parameters={"mapping": canonical_map},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="SAFE",
                    requires_approval=True,
                    why_it_matters=f"Inconsistent capitalization ('{list(canonical_map.keys())[:3]}') fragments category counts.",
                    risks_assumptions="Consolidates variations into standard Title Case representation.",
                    alternatives=[
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep As-Is", "description": "Retain original spelling and casing", "parameters": {}}
                    ]
                ))

            # 7. STATISTICAL OUTLIER
            elif fnd.category == QualityCategory.STATISTICAL_OUTLIER and col:
                lower_b = fnd.evidence.get("lower_threshold")
                upper_b = fnd.evidence.get("upper_threshold")
                actions.append(RemediationAction(
                    action_id=f"ACT-OUTLIER-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Cap Extreme Statistical Outliers in '{col}'",
                    description=f"Cap values in '{col}' to 3.0*IQR statistical thresholds [{lower_b}, {upper_b}].",
                    fix_type=FixType.CLAMP_BOUNDS,
                    target_columns=[col],
                    parameters={"min": lower_b, "max": upper_b},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="CAUTION",
                    requires_approval=True,
                    why_it_matters=f"Extreme values in '{col}' heavily bias linear models and skew standard deviation.",
                    risks_assumptions="Statistical outliers may represent genuine high-value customers or events.",
                    alternatives=[
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep As-Is (Recommended)", "description": "Preserve extreme values as legitimate phenomena", "parameters": {}},
                        {"fix_type": FixType.REMOVE_OUTLIERS, "label": "Drop Outlier Rows", "description": f"Remove {fnd.affected_rows_count} outlier rows entirely", "parameters": {"min": lower_b, "max": upper_b}}
                    ]
                ))

            # 8. SEMANTIC CONTRADICTION
            elif fnd.category == QualityCategory.SEMANTIC_CONTRADICTION:
                if fnd.finding_id.startswith("FND-SEM-TEMPORAL"):
                    start_c = fnd.evidence.get("start_column") or fnd.affected_columns[0]
                    end_c = fnd.evidence.get("end_column") or fnd.affected_columns[1]
                    action_id = "ACT-REPAIR-TEMPORAL" if fnd.finding_id == "FND-SEM-TEMPORAL" else f"ACT-REPAIR-TEMPORAL-{start_c}-{end_c}"
                    actions.append(RemediationAction(
                        action_id=action_id,
                        finding_id=fnd.finding_id,
                        title=f"Align Inverted Dates between '{start_c}' and '{end_c}'",
                        description=f"Swap chronological dates where {end_c} precedes {start_c}.",
                        fix_type=FixType.REPAIR_CONTRADICTION,
                        target_columns=[start_c, end_c],
                        parameters={"strategy": "swap_inverted_dates"},
                        estimated_impact_rows=fnd.affected_rows_count,
                        safety_level="SAFE",
                        requires_approval=True,
                        why_it_matters=f"Chronological sequence violation: '{end_c}' occurred before '{start_c}'.",
                        risks_assumptions="Assumes timestamps were transposed at ingest time.",
                        alternatives=[
                            {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep As-Is", "description": "Leave inverted timestamps untouched", "parameters": {}}
                        ]
                    ))
                elif fnd.finding_id == "FND-SEM-STATUS":
                    status_c = fnd.evidence.get("status_column")
                    churn_c = fnd.evidence.get("churn_column")
                    actions.append(RemediationAction(
                        action_id="ACT-REPAIR-STATUS-CHURN",
                        finding_id=fnd.finding_id,
                        title=f"Resolve Active Status with Churn Date Conflict",
                        description="Clear invalid churn date for customers actively marked as 'Active'.",
                        fix_type=FixType.REPAIR_CONTRADICTION,
                        target_columns=[status_c, churn_c],
                        parameters={"strategy": "clear_churn_for_active"},
                        estimated_impact_rows=fnd.affected_rows_count,
                        safety_level="SAFE",
                        requires_approval=True,
                        why_it_matters="Active customers with populated churn dates corrupt cohort retention analysis.",
                        risks_assumptions="Clears churn date for confirmed Active users.",
                        alternatives=[
                            {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep As-Is", "description": "Preserve conflicting records", "parameters": {}}
                        ]
                    ))

            # 9. CONSTANT COLUMN
            elif fnd.category == QualityCategory.CONSTANT_COLUMN and col:
                actions.append(RemediationAction(
                    action_id=f"ACT-DROP-CONST-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Drop Constant Column '{col}'",
                    description=f"Remove column '{col}' because it has zero/near-zero variance.",
                    fix_type=FixType.DROP_COLUMN,
                    target_columns=[col],
                    parameters={},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="SAFE",
                    requires_approval=True,
                    why_it_matters=f"Column '{col}' is near-constant and provides zero informational value.",
                    risks_assumptions="Removing uninformative column simplifies dataset without losing signal.",
                    alternatives=[
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep Column", "description": "Retain constant column in dataset", "parameters": {}}
                    ]
                ))

            # 10. HIGHLY INCOMPLETE COLUMN
            elif fnd.category == QualityCategory.HIGHLY_INCOMPLETE and col:
                actions.append(RemediationAction(
                    action_id=f"ACT-DROP-INCOMP-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Drop Highly Incomplete Column '{col}'",
                    description=f"Remove '{col}' because >60% of values are missing.",
                    fix_type=FixType.DROP_COLUMN,
                    target_columns=[col],
                    parameters={},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="CAUTION",
                    requires_approval=True,
                    why_it_matters=f"Column '{col}' has severe missingness. Imputing it introduces synthetic bias.",
                    risks_assumptions="Dropping column avoids fabricating >60% of entries.",
                    alternatives=[
                        {"fix_type": FixType.IMPUTE_CONSTANT, "label": "Fill Unknown", "description": "Keep column and fill missing with 'Unknown'", "parameters": {"replacement_value": "Unknown"}},
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep Column As-Is", "description": "Retain sparse column", "parameters": {}}
                    ]
                ))

            # 11. ML PREPARATION RISK / TARGET LEAKAGE
            elif fnd.category == QualityCategory.ML_PREPARATION_RISK and col:
                actions.append(RemediationAction(
                    action_id=f"ACT-LEAK-DROP-{col}",
                    finding_id=fnd.finding_id,
                    title=f"Drop Leaked Feature '{col}'",
                    description=f"Remove '{col}' to prevent artificial target leakage in machine learning.",
                    fix_type=FixType.DROP_COLUMN,
                    target_columns=[col],
                    parameters={},
                    estimated_impact_rows=fnd.affected_rows_count,
                    safety_level="SAFE",
                    requires_approval=True,
                    why_it_matters=f"Feature '{col}' duplicates or leaks target information, creating false model accuracy.",
                    risks_assumptions="Removing leakage ensures real-world model generalizability.",
                    alternatives=[
                        {"fix_type": FixType.KEEP_DUPLICATES, "label": "Keep Feature", "description": "Retain feature without modification", "parameters": {}}
                    ]
                ))

        return RemediationProposal(
            proposal_id=f"PROP-{hashlib.md5(str([a.action_id for a in actions]).encode()).hexdigest()[:6]}",
            root_cause_summary="Deterministic root-cause synthesis completed across missingness, bounds, types, and relational constraints.",
            actions=actions,
            confidence=0.96,
            model_used="DataGuard-Deterministic-Synthesizer"
        )

    @classmethod
    def apply_action(cls, df: pd.DataFrame, action: RemediationAction) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applies a single remediation action on a dataframe copy, returning transformed df and diff info."""
        work_df = df.copy()
        diff_info: Dict[str, Any] = {"action_id": action.action_id, "modified_rows": 0, "sample_changes": []}

        # 1. IMPUTATION (Median, Mean, Mode, Constant)
        if action.fix_type in (FixType.IMPUTE_MEDIAN, FixType.IMPUTE_MEAN, FixType.IMPUTE_MODE, FixType.IMPUTE_CONSTANT):
            col = action.target_columns[0]
            if col in work_df.columns:
                rep_val = action.parameters.get("replacement_value")
                if rep_val is None:
                    rep_val = action.parameters.get("placeholder", 0.0)

                sentinels = set(action.parameters.get("sentinels", QualityEngine.SENTINEL_NULLS))

                if action.parameters.get("coerce_numeric", False):
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

                null_mask = work_df[col].isna()
                if work_df[col].dtype == object:
                    sent_mask = work_df[col].astype(str).str.strip().str.lower().isin(sentinels)
                    null_mask = null_mask | sent_mask
                elif pd.api.types.is_numeric_dtype(work_df[col]):
                    sent_mask = (work_df[col] == -999.0)
                    null_mask = null_mask | sent_mask

                modified_count = int(null_mask.sum())
                if modified_count > 0:
                    indices = work_df.index[null_mask].tolist()
                    samples = work_df.loc[indices[:3], [col]].to_dict(orient="records")
                    work_df.loc[null_mask, col] = rep_val
                    diff_info["modified_rows"] = modified_count
                    diff_info["sample_changes"] = samples

        # 2. DROP NULL ROWS
        elif action.fix_type == FixType.DROP_NULL_ROWS:
            col = action.target_columns[0]
            if col in work_df.columns:
                before_len = len(work_df)
                work_df = work_df.dropna(subset=[col]).reset_index(drop=True)
                diff_info["modified_rows"] = before_len - len(work_df)

        # 3. DROP COLUMN
        elif action.fix_type == FixType.DROP_COLUMN:
            cols = [c for c in action.target_columns if c in work_df.columns]
            if cols:
                work_df = work_df.drop(columns=cols)
                diff_info["modified_rows"] = len(work_df)
                diff_info["dropped_columns"] = cols

        # 4. DROP DUPLICATES
        elif action.fix_type == FixType.DROP_DUPLICATES:
            before_len = len(work_df)
            subset = action.parameters.get("subset")
            work_df = work_df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
            diff_info["modified_rows"] = before_len - len(work_df)

        # 5. FLAG DUPLICATES
        elif action.fix_type == FixType.FLAG_DUPLICATES:
            subset = action.parameters.get("subset")
            work_df["is_duplicate"] = work_df.duplicated(subset=subset, keep=False)
            diff_info["modified_rows"] = int(work_df["is_duplicate"].sum())

        # 6. KEEP DUPLICATES (NO-OP)
        elif action.fix_type == FixType.KEEP_DUPLICATES:
            diff_info["modified_rows"] = 0
            diff_info["note"] = "Acknowledged and kept unchanged."

        # 7. CLAMP BOUNDS
        elif action.fix_type == FixType.CLAMP_BOUNDS:
            col = action.target_columns[0]
            if col in work_df.columns:
                min_b = action.parameters.get("min")
                max_b = action.parameters.get("max")
                work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
                mask = pd.Series(False, index=work_df.index)
                if min_b is not None:
                    mask |= (work_df[col] < min_b)
                if max_b is not None:
                    mask |= (work_df[col] > max_b)
                diff_info["modified_rows"] = int(mask.sum())
                if min_b is not None:
                    work_df[col] = work_df[col].clip(lower=min_b)
                if max_b is not None:
                    work_df[col] = work_df[col].clip(upper=max_b)

        # 8. REMOVE OUTLIERS
        elif action.fix_type == FixType.REMOVE_OUTLIERS:
            col = action.target_columns[0]
            if col in work_df.columns:
                min_b = action.parameters.get("min")
                max_b = action.parameters.get("max")
                s = pd.to_numeric(work_df[col], errors="coerce")
                valid = pd.Series(True, index=work_df.index)
                if min_b is not None:
                    valid &= (s >= min_b)
                if max_b is not None:
                    valid &= (s <= max_b)
                before_len = len(work_df)
                work_df = work_df[valid].reset_index(drop=True)
                diff_info["modified_rows"] = before_len - len(work_df)

        # 9. STANDARDIZE DATE
        elif action.fix_type == FixType.STANDARDIZE_DATE:
            col = action.target_columns[0]
            if col in work_df.columns:
                target_fmt = action.parameters.get("target_format", "%Y-%m-%d")
                parsed = pd.to_datetime(work_df[col], errors="coerce")
                valid_mask = parsed.notna()
                diff_info["modified_rows"] = int(valid_mask.sum())
                work_df.loc[valid_mask, col] = parsed[valid_mask].dt.strftime(target_fmt)

        # 10. NORMALIZE CATEGORY
        elif action.fix_type == FixType.NORMALIZE_CATEGORY:
            col = action.target_columns[0]
            if col in work_df.columns:
                mapping = action.parameters.get("mapping", {})
                mask = work_df[col].isin(mapping.keys())
                diff_info["modified_rows"] = int(mask.sum())
                work_df[col] = work_df[col].replace(mapping)

        # 11. REPAIR CONTRADICTION
        elif action.fix_type == FixType.REPAIR_CONTRADICTION:
            strategy = action.parameters.get("strategy")
            if strategy == "swap_inverted_dates" and len(action.target_columns) >= 2:
                c1, c2 = action.target_columns[0], action.target_columns[1]
                if c1 in work_df.columns and c2 in work_df.columns:
                    d1 = pd.to_datetime(work_df[c1], errors="coerce")
                    d2 = pd.to_datetime(work_df[c2], errors="coerce")
                    inversion = d1.notna() & d2.notna() & (d2 < d1)
                    diff_info["modified_rows"] = int(inversion.sum())
                    for idx in work_df.index[inversion]:
                        val1 = work_df.at[idx, c1]
                        val2 = work_df.at[idx, c2]
                        work_df.at[idx, c1] = val2
                        work_df.at[idx, c2] = val1
            elif strategy == "clear_churn_for_active" and len(action.target_columns) >= 2:
                status_c, churn_c = action.target_columns[0], action.target_columns[1]
                if status_c in work_df.columns and churn_c in work_df.columns:
                    active_mask = work_df[status_c].astype(str).str.strip().str.lower().isin({"active", "active ", "current"})
                    has_churn = work_df[churn_c].notna() & ~work_df[churn_c].astype(str).str.strip().str.lower().isin(QualityEngine.SENTINEL_NULLS)
                    conflict = active_mask & has_churn
                    diff_info["modified_rows"] = int(conflict.sum())
                    work_df.loc[conflict, churn_c] = None

        # 12. ML FEATURE SCALING (Standardization / Min-Max)
        elif action.fix_type == FixType.STANDARDIZE_FEATURES:
            col = action.target_columns[0]
            if col in work_df.columns and pd.api.types.is_numeric_dtype(work_df[col]):
                s = pd.to_numeric(work_df[col], errors="coerce")
                std = s.std()
                if std > 0:
                    work_df[col] = (s - s.mean()) / std
                    diff_info["modified_rows"] = int(s.notna().sum())

        elif action.fix_type == FixType.MIN_MAX_SCALE:
            col = action.target_columns[0]
            if col in work_df.columns and pd.api.types.is_numeric_dtype(work_df[col]):
                s = pd.to_numeric(work_df[col], errors="coerce")
                rng = s.max() - s.min()
                if rng > 0:
                    work_df[col] = (s - s.min()) / rng
                    diff_info["modified_rows"] = int(s.notna().sum())

        # 13. ML ENCODING (One-Hot / Ordinal)
        elif action.fix_type == FixType.ONE_HOT_ENCODE:
            col = action.target_columns[0]
            if col in work_df.columns:
                work_df = pd.get_dummies(work_df, columns=[col], drop_first=True)
                diff_info["modified_rows"] = len(work_df)

        return work_df, diff_info

    @classmethod
    def preview_remediation(
        cls,
        df: pd.DataFrame,
        actions: List[RemediationAction]
    ) -> Tuple[pd.DataFrame, QualityScore, List[Dict[str, Any]]]:
        """Simulates applying a sequence of remediation actions on a copy and returns recalculated score."""
        current_df = df.copy()
        diffs = []

        for action in actions:
            if action.status == "REJECTED_BY_USER":
                continue
            current_df, diff_info = cls.apply_action(current_df, action)
            diffs.append(diff_info)

        recalc_score, _ = QualityEngine.evaluate(current_df)
        return current_df, recalc_score, diffs
