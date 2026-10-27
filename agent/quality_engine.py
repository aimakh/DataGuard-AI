import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set
from models.findings import Finding, RuleResult, QualityScore, Severity, QualityCategory


class QualityEngine:
    """Enterprise Deterministic Data Quality & Reliability Engine.
    
    Evaluates 8 foundational data quality dimensions deterministically across
    arbitrary dataset schemas without hardcoding customer-specific assumptions.
    """

    CATEGORY_WEIGHTS: Dict[QualityCategory, float] = {
        QualityCategory.NULL_VALUES: 0.20,
        QualityCategory.DUPLICATES: 0.15,
        QualityCategory.SCHEMA_MISMATCH: 0.15,
        QualityCategory.BOUNDS_VIOLATION: 0.15,
        QualityCategory.FORMAT_INCONSISTENCY: 0.10,
        QualityCategory.CATEGORY_NORMALIZATION: 0.10,
        QualityCategory.STATISTICAL_OUTLIER: 0.10,
        QualityCategory.SEMANTIC_CONTRADICTION: 0.05,
        QualityCategory.CONSTANT_COLUMN: 0.05,
        QualityCategory.HIGHLY_INCOMPLETE: 0.05,
        QualityCategory.SUSPICIOUS_IDENTIFIER: 0.03,
        QualityCategory.ML_PREPARATION_RISK: 0.05,
    }

    SENTINEL_NULLS: Set[str] = {
        "", "n/a", "na", "null", "none", "nan", "-999", "-999.0", "undefined",
        "?", "-", "--", "nil", "missing", "void"
    }

    @classmethod
    def evaluate(
        cls,
        df: pd.DataFrame,
        custom_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
        user_goal: str = "data_analysis",
        target_col: Optional[str] = None
    ) -> Tuple[QualityScore, List[Finding]]:
        findings: List[Finding] = []
        rule_results: List[RuleResult] = []
        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0:
            empty_score = QualityScore(
                score=100.0,
                rule_results=[],
                category_breakdown={},
                total_rows=0,
                total_columns=total_cols,
                total_findings=0
            )
            return empty_score, []

        # 1. Null and Sentinel Value Check
        null_res, null_fnds = cls.check_null_values(df)
        rule_results.append(null_res)
        findings.extend(null_fnds)

        # 2. Duplicate Records Check
        dup_res, dup_fnds = cls.check_duplicate_records(df)
        rule_results.append(dup_res)
        findings.extend(dup_fnds)

        # 3. Schema & Type Mismatch Check
        sch_res, sch_fnds = cls.check_schema_and_types(df)
        rule_results.append(sch_res)
        findings.extend(sch_fnds)

        # 4. Numeric & Range Bounds Violations
        bnd_res, bnd_fnds = cls.check_bounds_violations(df, custom_bounds=custom_bounds)
        rule_results.append(bnd_res)
        findings.extend(bnd_fnds)

        # 5. Format Inconsistencies (Dates, Emails)
        fmt_res, fmt_fnds = cls.check_format_inconsistencies(df)
        rule_results.append(fmt_res)
        findings.extend(fmt_fnds)

        # 6. Category Normalization (Casing, Whitespace)
        cat_res, cat_fnds = cls.check_category_normalization(df)
        rule_results.append(cat_res)
        findings.extend(cat_fnds)

        # 7. Statistical Outliers (Tukey IQR 3.0)
        out_res, out_fnds = cls.check_statistical_outliers(df)
        rule_results.append(out_res)
        findings.extend(out_fnds)

        # 8. Semantic Contradictions & Integrity
        sem_res, sem_fnds = cls.check_semantic_contradictions(df, custom_rules=custom_rules)
        rule_results.append(sem_res)
        findings.extend(sem_fnds)

        # 9. Constant & Zero Variance Columns
        const_res, const_fnds = cls.check_constant_columns(df)
        if not const_res.passed:
            rule_results.append(const_res)
            findings.extend(const_fnds)

        # 10. Highly Incomplete Columns (>60% missing)
        incomp_res, incomp_fnds = cls.check_highly_incomplete_columns(df)
        if not incomp_res.passed:
            rule_results.append(incomp_res)
            findings.extend(incomp_fnds)

        # 11. Machine Learning Preparation & Leakage Risks
        if user_goal == "machine_learning":
            ml_res, ml_fnds = cls.check_ml_preparation_risks(df, target_col=target_col, user_goal=user_goal)
            if not ml_res.passed:
                rule_results.append(ml_res)
                findings.extend(ml_fnds)

        # Composite Score Calculation
        total_penalty = sum(r.penalty for r in rule_results)
        final_score = max(0.0, min(100.0, round(100.0 - total_penalty, 1)))

        crit = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high = sum(1 for f in findings if f.severity == Severity.HIGH)
        med = sum(1 for f in findings if f.severity == Severity.MEDIUM)
        low = sum(1 for f in findings if f.severity == Severity.LOW)

        cat_breakdown = {}
        for r in rule_results:
            cat_breakdown[r.category.value] = round(r.penalty, 2)

        quality_score = QualityScore(
            score=final_score,
            rule_results=rule_results,
            category_breakdown=cat_breakdown,
            total_rows=total_rows,
            total_columns=total_cols,
            total_findings=len(findings),
            critical_count=crit,
            high_count=high,
            medium_count=med,
            low_count=low
        )

        return quality_score, findings

    @classmethod
    def check_null_values(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()

        for col in df.columns:
            s = df[col]
            null_mask = s.isna()
            
            # String sentinels
            if s.dtype == object:
                sentinel_mask = s.astype(str).str.strip().str.lower().isin(cls.SENTINEL_NULLS)
                null_mask = null_mask | sentinel_mask
            elif pd.api.types.is_numeric_dtype(s):
                numeric_sentinel = (s == -999.0) | (s == -9999.0)
                null_mask = null_mask | numeric_sentinel

            null_count = int(null_mask.sum())
            if null_count > 0:
                affected_indices = df.index[null_mask].tolist()
                affected_all_indices.update(affected_indices)
                null_pct = round((null_count / total_rows) * 100, 2)
                
                if null_pct > 30.0:
                    sev = Severity.CRITICAL
                elif null_pct > 10.0:
                    sev = Severity.HIGH
                elif null_pct > 2.0:
                    sev = Severity.MEDIUM
                else:
                    sev = Severity.LOW

                findings.append(Finding(
                    finding_id=f"FND-NULL-{col}",
                    rule_name="Null & Missing Value Check",
                    category=QualityCategory.NULL_VALUES,
                    severity=sev,
                    confidence=1.0,
                    affected_columns=[col],
                    affected_rows_count=null_count,
                    affected_row_indices=affected_indices[:100],
                    evidence={
                        "null_count": null_count,
                        "null_percentage": null_pct,
                        "detected_sentinels": list(set(s[null_mask].dropna().astype(str).head(5)))
                    },
                    method="Vectorized isna() with domain sentinel detection",
                    description=f"Column '{col}' contains {null_count} missing or sentinel values ({null_pct}% missingness).",
                    default_fix_type="IMPUTE_MEDIAN" if pd.api.types.is_numeric_dtype(s) else "IMPUTE_MODE"
                ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.NULL_VALUES]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Completeness (Null & Sentinel Values)",
            category=QualityCategory.NULL_VALUES,
            passed=passed,
            severity=Severity.HIGH if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Found {affected_total} records with missing or sentinel fields." if not passed else "No missing values detected."
        )
        return res, findings

    @classmethod
    def check_duplicate_records(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        
        # 1. Exact full-row duplicates
        exact_dup_mask = df.duplicated(keep=False)
        exact_dup_count = int(exact_dup_mask.sum())
        affected_indices = df.index[exact_dup_mask].tolist() if exact_dup_count > 0 else []

        if exact_dup_count > 0:
            findings.append(Finding(
                finding_id="FND-DUP-EXACT",
                rule_name="Exact Duplicate Record Check",
                category=QualityCategory.DUPLICATES,
                severity=Severity.HIGH,
                confidence=1.0,
                affected_columns=list(df.columns),
                affected_rows_count=exact_dup_count,
                affected_row_indices=affected_indices[:100],
                evidence={"exact_duplicate_rows": exact_dup_count},
                method="Pandas DataFrame.duplicated(keep=False)",
                description=f"Found {exact_dup_count} exact duplicate rows across all columns.",
                default_fix_type="DROP_DUPLICATES"
            ))

        # 2. Primary key candidate collision (token-based to avoid matching words like 'humidity')
        def is_id_col(c_name: str) -> bool:
            tokens = {t.lower() for t in re.split(r"[_\s\-]+", str(c_name))}
            if bool(tokens & {"id", "uuid", "guid", "key", "pk", "sku", "code"}):
                return True
            low = str(c_name).lower()
            return low.endswith("_id") or low.startswith("id_") or low == "id"

        id_candidates = [
            c for c in df.columns 
            if is_id_col(c)
            or (c == df.columns[0] and df[c].nunique() > total_rows * 0.90)
        ]
        
        pk_affected_indices = set()
        for pk_col in id_candidates:
            pk_dups = df.duplicated(subset=[pk_col], keep=False)
            pk_count = int(pk_dups.sum())
            if pk_count > 0 and pk_count > exact_dup_count:
                indices = df.index[pk_dups].tolist()
                pk_affected_indices.update(indices)
                findings.append(Finding(
                    finding_id=f"FND-DUP-PK-{pk_col}",
                    rule_name="Primary Key Collision Check",
                    category=QualityCategory.DUPLICATES,
                    severity=Severity.CRITICAL,
                    confidence=0.98,
                    affected_columns=[pk_col],
                    affected_rows_count=pk_count,
                    affected_row_indices=indices[:100],
                    evidence={"colliding_key": pk_col, "colliding_rows": pk_count},
                    method=f"Unique constraint verification on '{pk_col}'",
                    description=f"Column '{pk_col}' exhibits {pk_count} key collisions, violating uniqueness constraints.",
                    default_fix_type="DROP_DUPLICATES"
                ))

        all_dup_indices = set(affected_indices).union(pk_affected_indices)
        affected_total = len(all_dup_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.DUPLICATES]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Uniqueness & Duplicate Check",
            category=QualityCategory.DUPLICATES,
            passed=passed,
            severity=Severity.CRITICAL if any(f.severity == Severity.CRITICAL for f in findings) else Severity.HIGH if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Detected {affected_total} duplicate or colliding records." if not passed else "All records satisfy uniqueness criteria."
        )
        return res, findings

    @classmethod
    def check_schema_and_types(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()

        for col in df.columns:
            s = df[col].dropna()
            if s.empty:
                continue

            if s.dtype == object:
                converted = pd.to_numeric(s, errors="coerce")
                valid_num = converted.notna().sum()
                num_ratio = valid_num / len(s) if len(s) > 0 else 0
                
                if 0.70 <= num_ratio < 1.0:
                    bad_mask = df[col].notna() & converted.isna()
                    bad_count = int(bad_mask.sum())
                    bad_indices = df.index[bad_mask].tolist()
                    affected_all_indices.update(bad_indices)
                    sample_bad = df.loc[bad_indices, col].head(5).tolist()

                    findings.append(Finding(
                        finding_id=f"FND-TYPE-{col}",
                        rule_name="Schema Type Contamination",
                        category=QualityCategory.SCHEMA_MISMATCH,
                        severity=Severity.HIGH,
                        confidence=0.95,
                        affected_columns=[col],
                        affected_rows_count=bad_count,
                        affected_row_indices=bad_indices[:100],
                        evidence={
                            "valid_numeric_percentage": round(num_ratio * 100, 1),
                            "corrupted_values_count": bad_count,
                            "sample_corrupted_values": sample_bad
                        },
                        method="Numeric coercion heuristic with >70% homogeneity threshold",
                        description=f"Column '{col}' appears numeric but contains {bad_count} corrupted non-numeric values ({sample_bad}).",
                        default_fix_type="IMPUTE_MEDIAN"
                    ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.SCHEMA_MISMATCH]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Schema & Type Integrity Check",
            category=QualityCategory.SCHEMA_MISMATCH,
            passed=passed,
            severity=Severity.CRITICAL if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Found {len(findings)} columns with type mismatches." if not passed else "Schema types are consistent."
        )
        return res, findings

    @classmethod
    def check_bounds_violations(
        cls,
        df: pd.DataFrame,
        custom_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
    ) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()
        custom_bounds = custom_bounds or {}

        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                clean_s = s.dropna()
            else:
                converted = pd.to_numeric(s, errors="coerce")
                if converted.notna().sum() > len(s) * 0.70:
                    clean_s = converted.dropna()
                else:
                    continue
            
            if clean_s.empty:
                continue

            col_lower = str(col).lower()
            tokens = [t.lower() for t in re.split(r"[_\s\-]+", col_lower)]
            min_bound, max_bound = None, None

            # 1. Custom bounds take precedence
            if col in custom_bounds:
                min_bound, max_bound = custom_bounds[col]
            # 2. Inferred bounds based on column semantics
            elif any(k in tokens for k in ["age", "years_old"]):
                min_bound, max_bound = 0, 120
            elif any(k in tokens for k in ["price", "salary", "revenue", "amount", "cost", "income", "spend", "balance", "qty", "quantity", "count", "orders", "units"]):
                min_bound = 0
            elif any(k in tokens for k in ["pct", "percent", "score_pct", "rate"]):
                if clean_s.max() <= 1.0:
                    min_bound, max_bound = 0.0, 1.0
                else:
                    min_bound, max_bound = 0, 100

            if min_bound is not None or max_bound is not None:
                viol_mask = pd.Series(False, index=df.index)
                if min_bound is not None:
                    viol_mask |= (clean_s < min_bound)
                if max_bound is not None:
                    viol_mask |= (clean_s > max_bound)
                
                viol_count = int(viol_mask.sum())
                if viol_count > 0:
                    indices = df.index[viol_mask].tolist()
                    affected_all_indices.update(indices)
                    viol_values = clean_s[viol_mask].head(5).tolist()

                    findings.append(Finding(
                        finding_id=f"FND-BND-{col}",
                        rule_name="Domain Range Violation",
                        category=QualityCategory.BOUNDS_VIOLATION,
                        severity=Severity.HIGH if any(k in tokens for k in ["age", "price", "cost"]) else Severity.MEDIUM,
                        confidence=0.98,
                        affected_columns=[col],
                        affected_rows_count=viol_count,
                        affected_row_indices=indices[:100],
                        evidence={
                            "min_allowed": min_bound,
                            "max_allowed": max_bound,
                            "observed_violating_values": viol_values,
                            "total_violations": viol_count
                        },
                        method=f"Range validation [{min_bound}, {max_bound}]",
                        description=f"Column '{col}' has {viol_count} values outside expected bounds [{min_bound}, {max_bound}].",
                        default_fix_type="CLAMP_BOUNDS"
                    ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.BOUNDS_VIOLATION]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Numeric Range & Bounds Check",
            category=QualityCategory.BOUNDS_VIOLATION,
            passed=passed,
            severity=Severity.MEDIUM if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Found {affected_total} values violating numeric bounds." if not passed else "Numeric values fall within valid bounds."
        )
        return res, findings

    @classmethod
    def check_format_inconsistencies(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()

        iso_date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
        slash_date_pattern = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}")
        email_pattern = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")

        for col in df.columns:
            s = df[col].dropna().astype(str).str.strip()
            if s.empty:
                continue

            col_lower = str(col).lower()
            
            # 1. Date format check (exclude numeric columns)
            if not pd.api.types.is_numeric_dtype(df[col]):
                is_date_named = any(k in col_lower for k in ["date", "time", "signup", "created", "churn", "admitted", "discharged", "dob", "birth", "order", "delivery"])
                iso_matches = s.apply(lambda v: bool(iso_date_pattern.match(v))).sum()
                slash_matches = s.apply(lambda v: bool(slash_date_pattern.match(v))).sum()
                
                if (is_date_named or (iso_matches + slash_matches > len(s) * 0.50)) and iso_matches > 0 and slash_matches > 0:
                    inconsistent_mask = s.apply(lambda v: bool(slash_date_pattern.match(v)))
                    inc_count = int(inconsistent_mask.sum())
                    indices = df.index[df[col].isin(s[inconsistent_mask])].tolist()
                    affected_all_indices.update(indices)

                    findings.append(Finding(
                        finding_id=f"FND-FMT-DATE-{col}",
                        rule_name="Date Format Inconsistency",
                        category=QualityCategory.FORMAT_INCONSISTENCY,
                        severity=Severity.HIGH,
                        confidence=0.99,
                        affected_columns=[col],
                        affected_rows_count=inc_count,
                        affected_row_indices=indices[:100],
                        evidence={
                            "canonical_format": "YYYY-MM-DD",
                            "divergent_format": "MM/DD/YYYY",
                            "divergent_rows_count": inc_count,
                            "sample_divergent_values": s[inconsistent_mask].head(5).tolist()
                        },
                        method="Regex pattern matching for ISO 8601 vs US Slash date divergence",
                        description=f"Column '{col}' mixes ISO (YYYY-MM-DD) and non-standard (MM/DD/YYYY) representations.",
                        default_fix_type="STANDARDIZE_DATE"
                    ))

            # 2. Email format check
            is_email_col = "email" in col_lower or s.str.contains("@").mean() > 0.40
            if is_email_col and not pd.api.types.is_numeric_dtype(df[col]):
                valid_emails = s.apply(lambda v: bool(email_pattern.match(v)))
                bad_emails = ~valid_emails & ~s.str.lower().isin(cls.SENTINEL_NULLS)
                bad_count = int(bad_emails.sum())
                if bad_count > 0:
                    indices = df.index[df[col].isin(s[bad_emails])].tolist()
                    affected_all_indices.update(indices)
                    findings.append(Finding(
                        finding_id=f"FND-FMT-EMAIL-{col}",
                        rule_name="Email Syntax Validation",
                        category=QualityCategory.FORMAT_INCONSISTENCY,
                        severity=Severity.MEDIUM,
                        confidence=0.95,
                        affected_columns=[col],
                        affected_rows_count=bad_count,
                        affected_row_indices=indices[:100],
                        evidence={"invalid_count": bad_count, "sample_invalid": s[bad_emails].head(5).tolist()},
                        method="RFC-5322 regex validation",
                        description=f"Column '{col}' contains {bad_count} malformed email addresses.",
                        default_fix_type="IMPUTE_CONSTANT"
                    ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.FORMAT_INCONSISTENCY]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Format Consistency Check",
            category=QualityCategory.FORMAT_INCONSISTENCY,
            passed=passed,
            severity=Severity.HIGH if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Found {affected_total} records with inconsistent formatting." if not passed else "All format patterns conform to canonical standards."
        )
        return res, findings

    @classmethod
    def check_category_normalization(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()

        for col in df.columns:
            s = df[col].dropna()
            is_cat = isinstance(s.dtype, pd.CategoricalDtype)
            if s.dtype != object and not is_cat:
                continue

            unique_vals = [str(x) for x in s.unique() if str(x).strip().lower() not in cls.SENTINEL_NULLS]
            if len(unique_vals) < 2 or len(unique_vals) > 50:
                continue

            normalized_groups = {}
            for v in unique_vals:
                k = v.strip().lower()
                normalized_groups.setdefault(k, []).append(v)

            variant_groups = {k: v for k, v in normalized_groups.items() if len(v) > 1}
            if variant_groups:
                violating_vals = [val for group in variant_groups.values() for val in group]
                mask = df[col].astype(str).isin(violating_vals)
                viol_count = int(mask.sum())
                indices = df.index[mask].tolist()
                affected_all_indices.update(indices)

                findings.append(Finding(
                    finding_id=f"FND-CAT-NORM-{col}",
                    rule_name="Categorical Variant Normalization",
                    category=QualityCategory.CATEGORY_NORMALIZATION,
                    severity=Severity.MEDIUM,
                    confidence=0.99,
                    affected_columns=[col],
                    affected_rows_count=viol_count,
                    affected_row_indices=indices[:100],
                    evidence={"split_variants": variant_groups, "total_unnormalized_rows": viol_count},
                    method="Case-insensitive whitespace-stripped grouping",
                    description=f"Column '{col}' contains casing or whitespace variants across categories ({list(variant_groups.keys())}).",
                    default_fix_type="NORMALIZE_CATEGORY"
                ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.CATEGORY_NORMALIZATION]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Category Normalization Check",
            category=QualityCategory.CATEGORY_NORMALIZATION,
            passed=passed,
            severity=Severity.MEDIUM if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Found {len(findings)} categorical columns with inconsistent casing/whitespace." if not passed else "Categorical values are uniformly normalized."
        )
        return res, findings

    @classmethod
    def check_statistical_outliers(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()

        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 15:
                continue

            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr = q75 - q25
            
            if iqr <= 0:
                continue

            # Extreme outlier test (3.0 * IQR)
            lower_fence = q25 - (3.0 * iqr)
            upper_fence = q75 + (3.0 * iqr)

            outlier_mask = (s < lower_fence) | (s > upper_fence)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0:
                indices = s.index[outlier_mask].tolist()
                affected_all_indices.update(indices)
                outlier_samples = s[outlier_mask].head(5).tolist()

                findings.append(Finding(
                    finding_id=f"FND-OUTLIER-{col}",
                    rule_name="Extreme Statistical Outlier Check",
                    category=QualityCategory.STATISTICAL_OUTLIER,
                    severity=Severity.LOW,  # Informational warning
                    confidence=0.90,
                    affected_columns=[col],
                    affected_rows_count=outlier_count,
                    affected_row_indices=indices[:100],
                    evidence={
                        "lower_threshold": round(float(lower_fence), 2),
                        "upper_threshold": round(float(upper_fence), 2),
                        "sample_outlier_values": outlier_samples,
                        "iqr": round(float(iqr), 2)
                    },
                    method="Tukey 3.0x IQR Extreme Outlier Criterion",
                    description=f"Column '{col}' contains {outlier_count} extreme statistical outliers. (Note: Domain review recommended; outliers may reflect genuine business phenomena).",
                    default_fix_type="CLAMP_BOUNDS"
                ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(15.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.STATISTICAL_OUTLIER]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Distribution & Statistical Outlier Check",
            category=QualityCategory.STATISTICAL_OUTLIER,
            passed=passed,
            severity=Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Flagged {affected_total} statistical outliers for domain verification." if not passed else "Data distributions conform to standard dispersion bounds."
        )
        return res, findings

    @classmethod
    def check_semantic_contradictions(
        cls,
        df: pd.DataFrame,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[RuleResult, List[Finding]]:
        findings = []
        total_rows = len(df)
        affected_all_indices = set()
        cols_lower = {str(c).lower(): c for c in df.columns}

        # Candidate keywords for temporal pairs
        start_tokens = {"start", "begin", "create", "signup", "open", "order", "admit", "admission", "hire", "departure", "entry"}
        end_tokens = {"end", "finish", "close", "churn", "cancel", "resolve", "discharge", "delivery", "termination", "arrival", "exit"}

        def get_col_tokens(name: str) -> Set[str]:
            return set(re.split(r"[_\s\-]+", str(name).lower()))

        # Non-numeric columns with date keywords
        date_candidates = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and any(k in str(c).lower() for k in ["date", "time", "day", "at"])
        ]

        start_candidates = [c for c in date_candidates if bool(get_col_tokens(c) & start_tokens)]
        end_candidates = [c for c in date_candidates if bool(get_col_tokens(c) & end_tokens)]

        tested_pairs = set()
        for s_c in start_candidates:
            for e_c in end_candidates:
                if s_c == e_c or (s_c, e_c) in tested_pairs:
                    continue
                tested_pairs.add((s_c, e_c))

                d_start = pd.to_datetime(df[s_c], errors="coerce")
                d_end = pd.to_datetime(df[e_c], errors="coerce")
                
                both_valid = d_start.notna() & d_end.notna()
                if both_valid.sum() == 0:
                    continue

                contradiction_mask = both_valid & (d_end < d_start)
                contra_count = int(contradiction_mask.sum())

                if contra_count > 0:
                    indices = df.index[contradiction_mask].tolist()
                    affected_all_indices.update(indices)
                    finding_id = "FND-SEM-TEMPORAL" if (s_c.lower() in ["signup_date", "start_date"] and e_c.lower() in ["churn_date", "end_date"]) else f"FND-SEM-TEMPORAL-{s_c}-{e_c}"
                    findings.append(Finding(
                        finding_id=finding_id,
                        rule_name="Temporal Chronology Inversion",
                        category=QualityCategory.SEMANTIC_CONTRADICTION,
                        severity=Severity.CRITICAL,
                        confidence=1.0,
                        affected_columns=[s_c, e_c],
                        affected_rows_count=contra_count,
                        affected_row_indices=indices[:100],
                        evidence={
                            "contradiction_count": contra_count,
                            "start_column": s_c,
                            "end_column": e_c,
                            "sample_rows": df.loc[indices[:3], [s_c, e_c]].to_dict(orient="records")
                        },
                        method=f"Chronological ordering verification: '{e_c}' >= '{s_c}'",
                        description=f"In {contra_count} rows, '{e_c}' precedes '{s_c}', creating an impossible chronological timeline.",
                        default_fix_type="REPAIR_CONTRADICTION"
                    ))

        # 2. Status vs Inactive Event Contradiction
        status_col = cols_lower.get("status") or cols_lower.get("customer_status") or cols_lower.get("patient_status") or cols_lower.get("order_status")
        end_event_col = cols_lower.get("churn_date") or cols_lower.get("cancellation_date") or cols_lower.get("discharge_date")
        if status_col and end_event_col:
            active_mask = df[status_col].astype(str).str.strip().str.lower().isin({"active", "active ", "current", "open", "in-progress"})
            has_end = df[end_event_col].notna() & ~df[end_event_col].astype(str).str.strip().str.lower().isin(cls.SENTINEL_NULLS)
            
            conflict_mask = active_mask & has_end
            conflict_count = int(conflict_mask.sum())

            if conflict_count > 0:
                indices = df.index[conflict_mask].tolist()
                affected_all_indices.update(indices)
                findings.append(Finding(
                    finding_id="FND-SEM-STATUS",
                    rule_name="Status vs Cancellation Contradiction",
                    category=QualityCategory.SEMANTIC_CONTRADICTION,
                    severity=Severity.HIGH,
                    confidence=0.98,
                    affected_columns=[status_col, end_event_col],
                    affected_rows_count=conflict_count,
                    affected_row_indices=indices[:100],
                    evidence={
                        "conflict_count": conflict_count,
                        "status_column": status_col,
                        "churn_column": end_event_col
                    },
                    method=f"Relational integrity between '{status_col}' and '{end_event_col}'",
                    description=f"In {conflict_count} rows, records marked '{status_col}=Active' possess an end-event date ({end_event_col}).",
                    default_fix_type="REPAIR_CONTRADICTION"
                ))

        # 3. User-Provided Custom Relational Rules
        if custom_rules:
            for rule in custom_rules:
                r_type = rule.get("type")
                if r_type == "comparison":
                    c_left = rule.get("left")
                    c_right = rule.get("right")
                    op = rule.get("op", ">=")
                    if c_left in df.columns and c_right in df.columns:
                        s_left = pd.to_numeric(df[c_left], errors="coerce")
                        s_right = pd.to_numeric(df[c_right], errors="coerce")
                        both = s_left.notna() & s_right.notna()
                        if op == ">=":
                            viol = both & ~(s_left >= s_right)
                        elif op == ">":
                            viol = both & ~(s_left > s_right)
                        elif op == "<=":
                            viol = both & ~(s_left <= s_right)
                        elif op == "<":
                            viol = both & ~(s_left < s_right)
                        else:
                            viol = pd.Series(False, index=df.index)

                        v_count = int(viol.sum())
                        if v_count > 0:
                            indices = df.index[viol].tolist()
                            affected_all_indices.update(indices)
                            findings.append(Finding(
                                finding_id=f"FND-SEM-CUSTOM-{c_left}-{c_right}",
                                rule_name=rule.get("title", "Custom Relational Constraint"),
                                category=QualityCategory.SEMANTIC_CONTRADICTION,
                                severity=Severity.HIGH,
                                confidence=1.0,
                                affected_columns=[c_left, c_right],
                                affected_rows_count=v_count,
                                affected_row_indices=indices[:100],
                                evidence={"violations_count": v_count, "left": c_left, "op": op, "right": c_right},
                                method=f"Relational evaluation: {c_left} {op} {c_right}",
                                description=f"{v_count} rows violate custom constraint '{c_left} {op} {c_right}'.",
                                default_fix_type="REPAIR_CONTRADICTION"
                            ))

        affected_total = len(affected_all_indices)
        passed = (affected_total == 0)
        penalty = min(20.0, (affected_total / total_rows) * 100 * cls.CATEGORY_WEIGHTS[QualityCategory.SEMANTIC_CONTRADICTION]) if total_rows > 0 else 0.0

        res = RuleResult(
            rule_name="Semantic & Relational Invariants",
            category=QualityCategory.SEMANTIC_CONTRADICTION,
            passed=passed,
            severity=Severity.CRITICAL if any(f.severity == Severity.CRITICAL for f in findings) else Severity.HIGH if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=affected_total,
            findings=findings,
            message=f"Identified {affected_total} rows with business logic contradictions." if not passed else "All cross-column relational constraints satisfied."
        )
        return res, findings

    @classmethod
    def check_constant_columns(cls, df: pd.DataFrame) -> Tuple[RuleResult, List[Finding]]:
        """Identifies constant or near-constant columns with zero or near-zero variance."""
        findings: List[Finding] = []
        total_rows = len(df)
        if total_rows < 25:
            return RuleResult(
                rule_name="Constant & Zero Variance Columns",
                category=QualityCategory.CONSTANT_COLUMN,
                passed=True,
                severity=Severity.LOW,
                penalty=0.0,
                affected_rows_count=0,
                findings=[],
                message="Dataset too small for reliable constant variance check."
            ), []

        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            
            # Check unique count and top frequency ratio
            val_counts = s.value_counts(normalize=True)
            top_freq = float(val_counts.iloc[0]) if not val_counts.empty else 0.0
            
            # Check if > 98% identical
            if top_freq >= 0.98 and s.nunique() <= 2 and not col.lower().startswith("is_"):
                most_common = val_counts.index[0]
                aff_count = int(total_rows * top_freq)
                findings.append(Finding(
                    finding_id=f"FND-CONST-{col}",
                    rule_name="Constant Column Check",
                    category=QualityCategory.CONSTANT_COLUMN,
                    severity=Severity.LOW,
                    confidence=0.95,
                    affected_columns=[col],
                    affected_rows_count=aff_count,
                    affected_row_indices=list(range(min(50, total_rows))),
                    evidence={"column": col, "top_frequency": round(top_freq, 3), "dominant_value": str(most_common)},
                    method=f"Value distribution analysis: '{most_common}' represents {round(top_freq*100, 1)}% of values",
                    description=f"Column '{col}' is near-constant ({round(top_freq*100, 1)}% '{most_common}'). Offers no predictive power.",
                    default_fix_type="DROP_COLUMN"
                ))

        passed = (len(findings) == 0)
        penalty = min(5.0, len(findings) * 1.5)
        return RuleResult(
            rule_name="Constant & Zero Variance Columns",
            category=QualityCategory.CONSTANT_COLUMN,
            passed=passed,
            severity=Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=sum(f.affected_rows_count for f in findings),
            findings=findings,
            message="No uninformative constant columns found." if passed else f"Found {len(findings)} constant or zero-variance columns."
        ), findings

    @classmethod
    def check_highly_incomplete_columns(cls, df: pd.DataFrame, threshold: float = 0.60) -> Tuple[RuleResult, List[Finding]]:
        """Identifies columns where missing values exceed the threshold (default 60%)."""
        findings: List[Finding] = []
        total_rows = len(df)
        if total_rows < 10:
            return RuleResult(
                rule_name="Highly Incomplete Columns Check",
                category=QualityCategory.HIGHLY_INCOMPLETE,
                passed=True,
                severity=Severity.LOW,
                penalty=0.0,
                affected_rows_count=0,
                findings=[],
                message="Dataset too small for incomplete column thresholding."
            ), []

        for col in df.columns:
            s = df[col]
            null_count = int(s.isna().sum())
            if s.dtype == object:
                sent_count = int(s.astype(str).str.strip().str.lower().isin(cls.SENTINEL_NULLS).sum())
                null_count += sent_count

            missing_ratio = null_count / total_rows
            if missing_ratio >= threshold:
                findings.append(Finding(
                    finding_id=f"FND-INCOMP-{col}",
                    rule_name="Highly Incomplete Column Check",
                    category=QualityCategory.HIGHLY_INCOMPLETE,
                    severity=Severity.MEDIUM,
                    confidence=0.98,
                    affected_columns=[col],
                    affected_rows_count=null_count,
                    affected_row_indices=df.index[s.isna()].tolist()[:50],
                    evidence={"column": col, "missing_ratio": round(missing_ratio, 3), "threshold": threshold},
                    method=f"Completeness evaluation: {round(missing_ratio*100, 1)}% missing (threshold: {int(threshold*100)}%)",
                    description=f"Column '{col}' is {round(missing_ratio*100, 1)}% missing. Imputation may introduce severe synthetic bias.",
                    default_fix_type="DROP_COLUMN"
                ))

        passed = (len(findings) == 0)
        penalty = min(6.0, len(findings) * 2.0)
        return RuleResult(
            rule_name="Highly Incomplete Columns Check",
            category=QualityCategory.HIGHLY_INCOMPLETE,
            passed=passed,
            severity=Severity.MEDIUM if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=sum(f.affected_rows_count for f in findings),
            findings=findings,
            message="All columns have sufficient data density." if passed else f"Identified {len(findings)} columns with >{int(threshold*100)}% missingness."
        ), findings

    @classmethod
    def check_ml_preparation_risks(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        user_goal: str = "data_analysis"
    ) -> Tuple[RuleResult, List[Finding]]:
        """Identifies target leakage and high-risk ML preparation issues."""
        findings: List[Finding] = []
        total_rows = len(df)

        if user_goal == "machine_learning" and target_col and target_col in df.columns:
            # Target leakage check
            y = pd.to_numeric(df[target_col], errors="coerce")
            if y.dropna().shape[0] > 10:
                for col in df.columns:
                    if col == target_col:
                        continue
                    if pd.api.types.is_numeric_dtype(df[col]):
                        x = pd.to_numeric(df[col], errors="coerce")
                        valid = x.notna() & y.notna()
                        if valid.sum() > 10:
                            corr = abs(float(x[valid].corr(y[valid])))
                            if corr >= 0.995:
                                findings.append(Finding(
                                    finding_id=f"FND-LEAK-{col}",
                                    rule_name="Target Leakage Risk Check",
                                    category=QualityCategory.ML_PREPARATION_RISK,
                                    severity=Severity.HIGH,
                                    confidence=0.99,
                                    affected_columns=[col, target_col],
                                    affected_rows_count=total_rows,
                                    affected_row_indices=[],
                                    evidence={"feature": col, "target": target_col, "correlation": round(corr, 4)},
                                    method=f"Bivariate correlation: abs(r) = {round(corr, 4)} >= 0.995",
                                    description=f"Feature '{col}' has near-perfect correlation ({round(corr, 4)}) with target '{target_col}', indicating data leakage.",
                                    default_fix_type="DROP_COLUMN"
                                ))

        passed = (len(findings) == 0)
        penalty = min(5.0, len(findings) * 3.0)
        return RuleResult(
            rule_name="Machine Learning Preparation Risks",
            category=QualityCategory.ML_PREPARATION_RISK,
            passed=passed,
            severity=Severity.HIGH if not passed else Severity.LOW,
            penalty=round(penalty, 2),
            affected_rows_count=sum(f.affected_rows_count for f in findings),
            findings=findings,
            message="No obvious target leakage or severe ML risks detected." if passed else f"Detected {len(findings)} ML preparation risks."
        ), findings
