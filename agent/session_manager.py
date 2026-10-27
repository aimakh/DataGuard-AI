import re
import time
import copy
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from models.findings import Finding, QualityScore, Severity, QualityCategory
from models.remediation import RemediationAction, RemediationProposal, FixType, AuditRecord
from agent.quality_engine import QualityEngine
from agent.remediation_engine import RemediationEngine
from agent.auditor import Auditor
from integrations.nemotron import NemotronReasoner


class SessionManager:
    """
    Session and Version Manager for DataGuard AI.
    Drives the Sequential AI Agent workflow:
    - Maintains immutable original upload.
    - Manages versioned working copies (v0, v1, v2...) with rollback.
    - Presents issues one at a time with contextualized 'Why It Matters', suggested fix, and alternatives.
    - Handles user actions: Approve & Apply, Reject, Change Preference, Ask Why, Skip.
    - Detects auto-resolved redundant issues when earlier fixes fix other problems.
    - Interprets natural-language revisions.
    """

    def __init__(self):
        self.dataset_name: str = "dataset"
        self.original_df: Optional[pd.DataFrame] = None
        self.user_goal: str = "data_analysis"  # machine_learning, business_reporting, data_analysis, other
        self.target_column: Optional[str] = None
        self.custom_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.custom_rules: List[Dict[str, Any]] = []

        # Version history
        self.versions: List[Dict[str, Any]] = []
        self.current_version_idx: int = 0

        # Sequential issue queue
        self.issues: List[Dict[str, Any]] = []
        self.current_issue_idx: int = 0

        # Activity & Audit
        self.activity_log: List[Dict[str, Any]] = []
        self.nemotron = NemotronReasoner()

    @property
    def current_df(self) -> Optional[pd.DataFrame]:
        if 0 <= self.current_version_idx < len(self.versions):
            return self.versions[self.current_version_idx]["df"]
        return None

    @property
    def current_score(self) -> Optional[QualityScore]:
        if 0 <= self.current_version_idx < len(self.versions):
            return self.versions[self.current_version_idx]["score"]
        return None

    @property
    def initial_score(self) -> Optional[QualityScore]:
        if self.versions:
            return self.versions[0]["score"]
        return None

    def initialize_session(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        user_goal: str = "data_analysis",
        target_column: Optional[str] = None,
        custom_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Initializes a new dataset reliability session."""
        self.dataset_name = dataset_name
        self.original_df = df.copy()
        self.user_goal = user_goal
        self.target_column = target_column
        self.custom_bounds = custom_bounds or {}
        self.custom_rules = custom_rules or []
        self.versions.clear()
        self.issues.clear()
        self.activity_log.clear()
        self.current_issue_idx = 0

        # Initial deterministic quality evaluation
        initial_score, findings = QualityEngine.evaluate(
            self.original_df,
            custom_bounds=self.custom_bounds,
            custom_rules=self.custom_rules,
            user_goal=self.user_goal,
            target_col=self.target_column
        )

        # Generate proposal actions
        proposal = RemediationEngine.generate_proposals(
            findings,
            self.original_df,
            user_goal=self.user_goal,
            target_col=self.target_column
        )

        # Build Version 0 (Original)
        v0 = {
            "version_id": "v0",
            "tag": "Original Ingest",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Initial uploaded dataset without modifications.",
            "df": self.original_df.copy(),
            "score": initial_score,
            "findings_count": len(findings),
            "action_applied": None,
            "diff_info": None
        }
        self.versions.append(v0)
        self.current_version_idx = 0

        # Build sequential issue queue
        # Sort findings by severity (CRITICAL -> HIGH -> MEDIUM -> LOW)
        sev_rank = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        sorted_findings = sorted(findings, key=lambda f: sev_rank.get(f.severity, 4))

        action_map = {a.finding_id: a for a in proposal.actions}

        for idx, fnd in enumerate(sorted_findings):
            action = action_map.get(fnd.finding_id)
            if not action:
                continue

            # Populate contextual 'why_it_matters' based on user goal
            why_text = self._build_why_it_matters(fnd, action, self.user_goal)
            action.why_it_matters = why_text

            self.issues.append({
                "issue_num": idx + 1,
                "finding": fnd,
                "action": action,
                "status": "PENDING",  # PENDING, APPROVED, REJECTED, SKIPPED, AUTO_RESOLVED
                "resolution_notes": "",
                "resolved_version": None
            })

        self.activity_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event": "SESSION_INITIALIZED",
            "details": f"Ingested '{dataset_name}' ({len(df)} rows, {len(df.columns)} cols). Discovered {len(self.issues)} actionable issues."
        })

        return {
            "initial_score": initial_score.score,
            "total_issues": len(self.issues),
            "findings_count": len(findings),
            "dataset_goal": self.user_goal
        }

    def _build_why_it_matters(self, finding: Finding, action: RemediationAction, goal: str) -> str:
        """Explains why the issue matters in 1-2 simple English sentences."""
        col = finding.affected_columns[0] if finding.affected_columns else "this column"
        cat = finding.category

        if cat == QualityCategory.NULL_VALUES:
            return f"Missing values in '{col}' can cause errors in your calculations, break charts, or leave out important information."
        elif cat == QualityCategory.DUPLICATES:
            return f"Duplicate records can make your counts or totals look bigger than they really are, leading to inaccurate summaries."
        elif cat == QualityCategory.SCHEMA_MISMATCH:
            return f"Text found inside the number column '{col}' will prevent math formulas, averages, and charts from working properly."
        elif cat == QualityCategory.BOUNDS_VIOLATION:
            return f"Values outside the expected range (like negative prices or ages) distort your averages and create misleading results."
        elif cat == QualityCategory.FORMAT_INCONSISTENCY:
            return f"Mixed date formats in '{col}' make it hard to sort by date or group data by month and year."
        elif cat == QualityCategory.CATEGORY_NORMALIZATION:
            return f"Inconsistent capitalization or extra spaces in '{col}' (like 'Gold' vs 'gold') will treat the same category as two different groups."
        elif cat == QualityCategory.STATISTICAL_OUTLIER:
            return f"Unusually large or small numbers in '{col}' might be typing errors and can pull your overall averages way off."
        elif cat == QualityCategory.SEMANTIC_CONTRADICTION:
            return f"Values in '{col}' conflict with related columns (for example, an end date happening before a start date)."
        elif cat == QualityCategory.CONSTANT_COLUMN:
            return f"Every single row in '{col}' has the exact same value, so it adds no useful information to your dataset."
        elif cat == QualityCategory.HIGHLY_INCOMPLETE:
            return f"Most of the values in '{col}' are completely empty (over 60% missing), making it unreliable for analysis."
        elif cat == QualityCategory.SUSPICIOUS_IDENTIFIER:
            return f"Column '{col}' looks like an internal ID number or code, which does not provide predictive patterns."
        elif cat == QualityCategory.ML_PREPARATION_RISK:
            return f"Column '{col}' may cause problems for AI models, such as accidentally leaking the answer or containing too many unique categories."
        else:
            return f"Inconsistent or unexpected values in '{col}' can reduce the accuracy of your analysis."

    def get_current_issue(self) -> Optional[Dict[str, Any]]:
        """Returns the current issue under sequential review, or None if all are reviewed."""
        # Find next pending issue starting from current_issue_idx
        for idx in range(self.current_issue_idx, len(self.issues)):
            issue = self.issues[idx]
            if issue["status"] == "PENDING":
                self.current_issue_idx = idx
                return issue

        # If none from current index forward, check from start for any remaining pending
        for idx in range(len(self.issues)):
            issue = self.issues[idx]
            if issue["status"] == "PENDING":
                self.current_issue_idx = idx
                return issue

        return None

    def get_progress_status(self) -> Dict[str, Any]:
        """Returns sequential review progress metrics in simple English."""
        total = len(self.issues)
        fixed = sum(1 for i in self.issues if i["status"] in ("APPROVED", "AUTO_RESOLVED"))
        rejected = sum(1 for i in self.issues if i["status"] == "REJECTED")
        skipped = sum(1 for i in self.issues if i["status"] == "SKIPPED")
        auto_resolved = sum(1 for i in self.issues if i["status"] == "AUTO_RESOLVED")
        reviewed = fixed + rejected + skipped
        remaining = max(0, total - reviewed)

        return {
            "total": total,
            "fixed": fixed,
            "rejected": rejected,
            "skipped": skipped,
            "auto_resolved": auto_resolved,
            "reviewed": reviewed,
            "remaining": remaining,
            "current_num": min(total, reviewed + (1 if remaining > 0 else 0)),
            "progress_text": f"You have reviewed {reviewed} of {total} problems. {fixed} fixed, {skipped} skipped."
        }

    def preview_issue_change(self, custom_action: Optional[RemediationAction] = None) -> Dict[str, Any]:
        """Generates a detailed preview of what applying the current issue's action will do."""
        curr = self.get_current_issue()
        if not curr or self.current_df is None:
            return {"error": "No active issue or dataset"}

        action = custom_action or curr["action"]
        finding = curr["finding"]

        # Run preview transformation on working copy
        sim_df, diff_info = RemediationEngine.apply_action(self.current_df, action)

        # Extract sample before vs after records
        cols = action.target_columns
        target_col = cols[0] if cols else None

        before_samples = []
        after_samples = []

        if target_col and target_col in self.current_df.columns:
            # Pick affected indices
            aff_indices = finding.affected_row_indices[:5]
            if not aff_indices and diff_info.get("modified_rows", 0) > 0:
                # Find changed rows
                changed_mask = (self.current_df[target_col] != sim_df[target_col]) | (self.current_df[target_col].isna() != sim_df[target_col].isna())
                aff_indices = self.current_df.index[changed_mask].tolist()[:5]

            for idx in aff_indices:
                if idx in self.current_df.index and idx in sim_df.index:
                    before_samples.append({
                        "row_idx": int(idx),
                        "original_value": str(self.current_df.at[idx, target_col])
                    })
                    after_samples.append({
                        "row_idx": int(idx),
                        "proposed_value": str(sim_df.at[idx, target_col])
                    })

        return {
            "action_id": action.action_id,
            "title": action.title,
            "method": action.fix_type.value,
            "target_columns": action.target_columns,
            "affected_rows_count": diff_info.get("modified_rows", finding.affected_rows_count),
            "before_samples": before_samples,
            "after_samples": after_samples,
            "why_it_matters": action.why_it_matters,
            "risks_assumptions": action.risks_assumptions or "Reversible transformation applied exclusively to versioned working copy.",
            "alternatives": action.alternatives
        }

    def approve_and_apply(self, chosen_action: Optional[RemediationAction] = None) -> Dict[str, Any]:
        """
        Applies the approved transformation to a new working version.
        Re-runs quality rules, records the delta, and detects auto-resolved issues.
        """
        curr = self.get_current_issue()
        if not curr or self.current_df is None:
            return {"success": False, "message": "No active issue to approve"}

        action = chosen_action or curr["action"]
        finding = curr["finding"]

        # 1. Apply transformation strictly to a copy
        df_new, diff_info = RemediationEngine.apply_action(self.current_df, action)

        # 2. Re-evaluate quality rules deterministically on transformed dataframe
        new_score, remaining_findings = QualityEngine.evaluate(
            df_new,
            custom_bounds=self.custom_bounds,
            custom_rules=self.custom_rules,
            user_goal=self.user_goal,
            target_col=self.target_column
        )

        # 3. Create new Version
        v_idx = len(self.versions)
        v_new = {
            "version_id": f"v{v_idx}",
            "tag": f"v{v_idx}: {action.title}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"Applied {action.fix_type.value} on {action.target_columns}: {diff_info.get('modified_rows', 0)} rows modified.",
            "df": df_new,
            "score": new_score,
            "findings_count": len(remaining_findings),
            "action_applied": action,
            "diff_info": diff_info
        }
        self.versions.append(v_new)
        self.current_version_idx = v_idx

        # 4. Mark current issue as APPROVED
        curr["status"] = "APPROVED"
        curr["resolved_version"] = f"v{v_idx}"
        curr["resolution_notes"] = f"User approved fix. Score improved to {new_score.score}/100."

        # 5. Redundancy Detection: Check if any remaining pending issues are now auto-resolved!
        remaining_fnd_ids = {f.finding_id for f in remaining_findings}
        auto_resolved_count = 0
        auto_resolved_titles = []

        for issue in self.issues:
            if issue["status"] == "PENDING" and issue["finding"].finding_id not in remaining_fnd_ids:
                issue["status"] = "AUTO_RESOLVED"
                issue["resolved_version"] = f"v{v_idx}"
                issue["resolution_notes"] = f"Automatically resolved by prior change '{action.title}'."
                auto_resolved_count += 1
                auto_resolved_titles.append(issue["finding"].rule_name)

        # Log event
        self.activity_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event": "ACTION_APPROVED",
            "details": f"Applied '{action.title}' -> Score: {v_new['score'].score}/100. Auto-resolved {auto_resolved_count} secondary issues."
        })

        # Advance pointer to next issue
        self.current_issue_idx += 1

        return {
            "success": True,
            "new_version": f"v{v_idx}",
            "previous_score": self.versions[v_idx - 1]["score"].score,
            "new_score": new_score.score,
            "score_delta": round(new_score.score - self.versions[v_idx - 1]["score"].score, 1),
            "rows_affected": diff_info.get("modified_rows", 0),
            "auto_resolved_count": auto_resolved_count,
            "auto_resolved_titles": auto_resolved_titles,
            "remaining_issues": sum(1 for i in self.issues if i["status"] == "PENDING")
        }

    def reject_current_issue(self, reason: str = "User declined recommended fix") -> Dict[str, Any]:
        """Marks current issue as REJECTED without modifying data."""
        curr = self.get_current_issue()
        if not curr:
            return {"success": False, "message": "No active issue to reject"}

        curr["status"] = "REJECTED"
        curr["resolution_notes"] = reason

        self.activity_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event": "ACTION_REJECTED",
            "details": f"Rejected fix for '{curr['finding'].rule_name}'. Dataset unchanged."
        })

        self.current_issue_idx += 1
        return {"success": True, "message": f"Rejected fix for '{curr['finding'].rule_name}'. Moving to next issue."}

    def skip_current_issue(self) -> Dict[str, Any]:
        """Marks current issue as SKIPPED to be revisited later."""
        curr = self.get_current_issue()
        if not curr:
            return {"success": False, "message": "No active issue to skip"}

        curr["status"] = "SKIPPED"
        curr["resolution_notes"] = "Skipped by user to review later."

        self.activity_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event": "ACTION_SKIPPED",
            "details": f"Skipped issue '{curr['finding'].rule_name}'."
        })

        self.current_issue_idx += 1
        return {"success": True, "message": f"Skipped issue '{curr['finding'].rule_name}'."}

    def rollback_to_version(self, target_version_idx: int) -> Dict[str, Any]:
        """Rolls back the active working copy to any previous version in history."""
        if not (0 <= target_version_idx < len(self.versions)):
            return {"success": False, "message": f"Invalid version index {target_version_idx}"}

        self.current_version_idx = target_version_idx
        v = self.versions[target_version_idx]

        # Re-synchronize issue statuses against this version's findings
        score, findings = QualityEngine.evaluate(
            v["df"],
            custom_bounds=self.custom_bounds,
            custom_rules=self.custom_rules,
            user_goal=self.user_goal,
            target_col=self.target_column
        )
        active_fnd_ids = {f.finding_id for f in findings}

        for issue in self.issues:
            if issue["finding"].finding_id in active_fnd_ids:
                if issue["status"] in ("APPROVED", "AUTO_RESOLVED"):
                    issue["status"] = "PENDING"
                    issue["resolved_version"] = None

        self.activity_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event": "ROLLBACK_EXECUTED",
            "details": f"Rolled back to {v['version_id']} ({v['tag']}). Active score: {score.score}/100."
        })

        # Reset pointer
        self.current_issue_idx = 0

        return {
            "success": True,
            "version_id": v["version_id"],
            "score": score.score,
            "remaining_issues": sum(1 for i in self.issues if i["status"] == "PENDING")
        }

    def parse_natural_language_revision(self, user_text: str) -> Dict[str, Any]:
        """
        Interprets natural-language commands and revisions from the user in plain English:
        - 'Use the median instead.'
        - 'Keep these duplicate rows.'
        - 'Do not change this column.'
        - 'Why is this value incorrect?'
        - 'Undo my last change.'
        - 'Explain it more simply.'
        If unclear, asks a short clarifying question instead of guessing.
        """
        text = user_text.strip().lower()
        curr = self.get_current_issue()

        # 1. Undo / Rollback
        if any(w in text for w in ["undo", "revert", "rollback", "previous version", "undo my last change", "undo last change"]):
            if self.current_version_idx > 0:
                target = self.current_version_idx - 1
                res = self.rollback_to_version(target)
                return {
                    "action_taken": "ROLLBACK",
                    "response": f"Undone! Rolled back to version v{target}. Your dataset is restored to score {res['score']}/100.",
                    "requires_approval": False
                }
            else:
                return {
                    "action_taken": "ERROR",
                    "response": "You are already at the original version (v0). There are no previous changes to undo.",
                    "requires_approval": False
                }

        # 2. Skip
        if any(w in text for w in ["skip", "pass", "skip this problem", "skip this issue", "next problem", "leave for now"]):
            self.skip_current_issue()
            return {
                "action_taken": "SKIP",
                "response": "Skipped this problem for now. Moving to the next problem.",
                "requires_approval": False
            }

        # 3. Keep duplicates or Do not change column / Leave data unchanged
        if "flag" in text and "duplicate" in text:
            if curr:
                action = curr["action"]
                action.fix_type = FixType.FLAG_DUPLICATES
                action.title = "Flag Duplicate Records with a New Column"
                action.description = "Keep all records and add a column called 'is_duplicate'."
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": "Updated preference: all duplicate rows will be kept, and marked with an 'is_duplicate' column.",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

        if "keep" in text and "duplicate" in text and "flag" not in text:
            self.reject_current_issue(reason="User chose to keep duplicate rows unchanged.")
            return {
                "action_taken": "REJECT",
                "response": "Understood. I will keep these duplicate rows exactly as they are without modifying them.",
                "requires_approval": False
            }

        if any(phrase in text for phrase in ["do not change this column", "don't change this column", "leave this column", "keep this column", "do not change"]):
            col_name = curr["action"].target_columns[0] if (curr and curr.get("action") and curr["action"].target_columns) else "this column"
            self.reject_current_issue(reason=f"User chose to leave column '{col_name}' unchanged.")
            return {
                "action_taken": "REJECT",
                "response": f"Understood. I will not change '{col_name}'. This data will remain unchanged.",
                "requires_approval": False
            }

        if any(w in text for w in ["leave unchanged", "leave this data unchanged", "reject", "do not apply", "don't fix"]):
            self.reject_current_issue(reason=f"User requested: '{user_text}'")
            return {
                "action_taken": "REJECT",
                "response": "Left this data unchanged. No changes were applied.",
                "requires_approval": False
            }

        # 4. Explain it more simply
        if any(phrase in text for phrase in ["explain it more simply", "explain simply", "simpler", "simple english", "plain english", "easy words"]):
            if curr:
                finding = curr["finding"]
                action = curr["action"]
                col = action.target_columns[0] if action.target_columns else "this column"
                simple_explanation = (
                    f"**Here is what is happening in simple words:**\n\n"
                    f"1. **What is wrong:** We looked at `{col}` and found {finding.affected_rows_count:,} values that don't look right ({finding.description}).\n"
                    f"2. **Why it matters:** {action.why_it_matters}\n"
                    f"3. **What I recommend doing:** {action.title}. This will make the data clean and consistent without touching other parts of your file.\n"
                    f"4. **You are in charge:** You can click **Apply This Fix**, choose a different way to fix it, or click **Leave This Data Unchanged**."
                )
                return {
                    "action_taken": "EXPLAIN",
                    "response": simple_explanation,
                    "requires_approval": False
                }

        # 5. Why is this value incorrect?
        if any(phrase in text for phrase in ["why is this value incorrect", "why is this incorrect", "why is this wrong", "why incorrect", "why did you flag this", "why", "explain"]):
            if curr:
                action = curr["action"]
                finding = curr["finding"]
                col = action.target_columns[0] if action.target_columns else "this column"
                explanation = (
                    f"**Why DataGuard AI recommended this:**\n\n"
                    f"- **What I found:** In column `{col}`, {finding.affected_rows_count:,} entries violate data quality rules ({finding.description}).\n"
                    f"- **Why you should care:** {action.why_it_matters}\n"
                    f"- **Suggested fix:** {action.title}.\n"
                    f"- **Preview safety:** Only the affected values are touched on a working copy; your original file is never modified."
                )
                return {
                    "action_taken": "EXPLAIN",
                    "response": explanation,
                    "requires_approval": False
                }

        # 6. Specific Method Revisions on Current Issue
        if curr:
            action = curr["action"]
            col = action.target_columns[0] if action.target_columns else ""

            # Check for 'use X instead' constructs
            desired_method = None
            instead_match = re.search(r"use\s+(?:the\s+)?([a-zA-Z_-]+)\s+instead", text)
            if instead_match:
                desired_method = instead_match.group(1).lower()

            # Mean / Average preference
            if desired_method in ("mean", "average") or (desired_method is None and ("mean" in text or "average" in text) and "median" not in text):
                s = self.current_df[col]
                s_num = pd.to_numeric(s, errors="coerce")
                mean_val = float(s_num.dropna().mean()) if s_num.dropna().shape[0] > 0 else 0.0
                action.fix_type = FixType.IMPUTE_MEAN
                action.title = f"Fill in Missing Values in '{col}' with Average ({round(mean_val, 2)})"
                action.parameters = {"replacement_value": mean_val}
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": f"Updated your preference: I will fill in missing values in '{col}' using the average ({round(mean_val, 2)}).",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

            # Median preference
            if desired_method == "median" or (desired_method is None and "median" in text):
                s = self.current_df[col]
                s_num = pd.to_numeric(s, errors="coerce")
                med = float(s_num.dropna().median()) if s_num.dropna().shape[0] > 0 else 0.0
                action.fix_type = FixType.IMPUTE_MEDIAN
                action.title = f"Fill in Missing Values in '{col}' with Median ({round(med, 2)})"
                action.parameters = {"replacement_value": med}
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": f"Updated your preference: I will fill in missing values in '{col}' using the middle value / median ({round(med, 2)}).",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

            # Mode / Most common preference
            if desired_method == "mode" or "most frequent" in text or "most common" in text:
                s = self.current_df[col].dropna()
                mode_val = s.mode().iloc[0] if len(s.mode()) > 0 else ""
                action.fix_type = FixType.IMPUTE_MODE
                action.title = f"Fill in Missing Values in '{col}' with Most Common Value ({mode_val})"
                action.parameters = {"replacement_value": mode_val}
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": f"Updated your preference: I will fill in missing values in '{col}' using the most common value ('{mode_val}').",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

            # Flag duplicates
            if "flag" in text and "duplicate" in text:
                action.fix_type = FixType.FLAG_DUPLICATES
                action.title = "Flag Duplicate Records with a New Column"
                action.description = "Keep all records and add a column called 'is_duplicate'."
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": "Updated preference: all duplicate rows will be kept, and marked with an 'is_duplicate' column.",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

            # Date format change
            if "dd-mm-yyyy" in text or "day" in text:
                action.parameters["target_format"] = "%d-%m-%Y"
                action.title = f"Format Dates in '{col}' as DD-MM-YYYY"
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": f"Updated preference: dates in '{col}' will be formatted as DD-MM-YYYY.",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

            if "yyyy-mm-dd" in text or "iso" in text:
                action.parameters["target_format"] = "%Y-%m-%d"
                action.title = f"Format Dates in '{col}' as YYYY-MM-DD"
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": f"Updated preference: dates in '{col}' will be formatted as YYYY-MM-DD.",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

            # Drop column
            if "drop column" in text or "remove column" in text or "delete column" in text:
                action.fix_type = FixType.DROP_COLUMN
                action.title = f"Remove Column '{col}'"
                action.description = f"Remove column '{col}' entirely from your improved data."
                return {
                    "action_taken": "PREFERENCE_UPDATED",
                    "response": f"Updated preference: column '{col}' will be removed from your improved data.",
                    "requires_approval": True,
                    "preview": self.preview_issue_change(action)
                }

        # Clarifying Question for ambiguous inputs (Prompt requirement: If the request is unclear, ask a short question instead of guessing)
        return {
            "action_taken": "CLARIFICATION_NEEDED",
            "response": (
                f"I'm not completely sure what you would like to do with '{user_text}'.\n\n"
                f"Could you please clarify? For example, you can tell me:\n"
                f"- **'Use the median instead'** or **'Use the mean instead'**\n"
                f"- **'Keep these duplicate rows'** or **'Do not change this column'**\n"
                f"- **'Why is this value incorrect?'** or **'Explain it more simply'**\n"
                f"- **'Undo my last change'** to step back."
            ),
            "requires_approval": False
        }

    def generate_final_report_and_audit(self) -> Dict[str, Any]:
        """Generates comprehensive final dataset export summary and cryptographically sealed audit trail."""
        if not self.versions or self.current_df is None:
            raise ValueError("No session active for audit generation.")

        v_initial = self.versions[0]
        v_current = self.versions[self.current_version_idx]

        applied_actions = [
            v["action_applied"] for v in self.versions if v.get("action_applied") is not None
        ]

        audit = Auditor.create_audit(
            run_id=f"DG-SESSION-{int(time.time())}",
            dataset_name=self.dataset_name,
            df_before=v_initial["df"],
            df_after=v_current["df"],
            score_before=v_initial["score"],
            score_after=v_current["score"],
            applied_actions=applied_actions
        )

        # Summary for users in simple English
        progress = self.get_progress_status()
        summary_md = f"""# 🛡️ DataGuard AI — Dataset Reliability Certification Report
**Dataset:** `{self.dataset_name}`  
**User Goal:** `{self.user_goal.replace('_', ' ').title()}`  
**Generated At:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC')}`  
**Final Version:** `{v_current['version_id']} ({v_current['tag']})`  

---

## 📊 Reliability Score Transformation
- **Initial Quality Score:** `{v_initial['score'].score} / 100` (Grade: `{'A' if v_initial['score'].score>=90 else 'B' if v_initial['score'].score>=80 else 'C' if v_initial['score'].score>=70 else 'D' if v_initial['score'].score>=60 else 'F'}`)
- **Final Quality Score:** `{v_current['score'].score} / 100` (Grade: `{'A' if v_current['score'].score>=90 else 'B' if v_current['score'].score>=80 else 'C' if v_current['score'].score>=70 else 'D' if v_current['score'].score>=60 else 'F'}`)
- **Score Delta:** `+{round(v_current['score'].score - v_initial['score'].score, 1)} points`

---

## 📈 Dataset Dimensions
- **Original Dimensions:** `{len(v_initial['df']):,} rows × {len(v_initial['df'].columns)} columns`
- **Final Dimensions:** `{len(v_current['df']):,} rows × {len(v_current['df'].columns)} columns`
- **Total Issues Detected:** `{progress['total']}`
- **Approved & Applied Fixes:** `{progress['fixed']}`
- **Automatically Resolved Fixes:** `{progress['auto_resolved']}`
- **User-Rejected Fixes:** `{progress['rejected']}`
- **Skipped / Unresolved:** `{progress['skipped'] + progress['remaining']}`

---

## 🛠️ Executed Transformations Log
"""
        for idx, v in enumerate(self.versions[1:], 1):
            act = v.get("action_applied")
            title = act.title if act else v.get("tag", "Step")
            summary_md += f"{idx}. **{v['version_id']} ({v['timestamp']}):** {title}\n"

        summary_md += f"""
---

## 🔒 Cryptographic Governance Verification
- **Source SHA-256 Hash:** `{audit.content_hash_before}`
- **Remediated SHA-256 Hash:** `{audit.content_hash_after}`
- **Safety Guarantee:** Original upload was preserved 100% read-only. All transformations were authorized by human operator and executed exclusively on versioned working copies.
"""

        audit.summary_markdown = summary_md
        return {
            "audit_record": audit,
            "summary_markdown": summary_md,
            "progress": progress,
            "final_score": v_current["score"].score,
            "initial_score": v_initial["score"].score,
            "version_count": len(self.versions)
        }

    def export_dataset(self, format: str = "csv") -> bytes:
        """Exports the active working dataset as CSV or Excel bytes."""
        if self.current_df is None:
            raise ValueError("No working dataset available for export.")

        fmt = format.lower().strip()
        if fmt in ["xlsx", "excel"]:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                self.current_df.to_excel(writer, index=False, sheet_name="CleanedData")
            buffer.seek(0)
            return buffer.getvalue()
        else:
            return self.current_df.to_csv(index=False).encode("utf-8")
