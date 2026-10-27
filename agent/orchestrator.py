import uuid
import io
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd

from models.findings import Finding, QualityScore
from models.state import AgentStep, WorkflowStatus, AgentRunContext
from models.remediation import RemediationAction, RemediationProposal, AuditRecord
from agent.profiler import DatasetProfiler
from agent.quality_engine import QualityEngine
from agent.remediation_engine import RemediationEngine
from agent.investigator import Investigator
from agent.auditor import Auditor
from integrations.zetaris_client import ZetarisClient


class DataGuardOrchestrator:
    """Universal 10-state Master Orchestrator for DataGuard AI autonomous reliability lifecycle."""

    def __init__(self, dataset_name: str = "dataset"):
        self.dataset_name = dataset_name
        self.context = AgentRunContext(
            run_id=f"DG-{uuid.uuid4().hex[:8]}",
            dataset_name=dataset_name,
            current_step=AgentStep.INGEST,
            status=WorkflowStatus.IDLE
        )
        self.original_df: Optional[pd.DataFrame] = None
        self.working_df: Optional[pd.DataFrame] = None
        self.profile_data: Dict[str, Any] = {}
        self.proposal: Optional[RemediationProposal] = None
        self.preview_score: Optional[QualityScore] = None
        self.preview_diffs: List[Dict[str, Any]] = []
        self.audit_record: Optional[AuditRecord] = None
        self.schema_overrides: Dict[str, str] = {}
        self.custom_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.custom_rules: List[Dict[str, Any]] = []
        self.investigator = Investigator()

    def log_step(self, step: AgentStep, details: str):
        self.context.current_step = step
        self.context.step_history.append({"step": step.value, "details": details})

    def run_lifecycle_to_review(
        self,
        df: pd.DataFrame,
        schema_overrides: Optional[Dict[str, str]] = None,
        custom_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> AgentRunContext:
        """Executes INGEST -> PROFILE -> DETECT -> TRIAGE -> INVESTIGATE -> RECOMMEND -> VALIDATE."""
        # 1. INGEST (Preserve original data 100% read-only)
        self.original_df = df.copy()
        
        # Apply user-specified schema overrides if provided
        if schema_overrides:
            self.schema_overrides = schema_overrides
            active_df = DatasetProfiler.apply_schema_overrides(self.original_df, schema_overrides)
        else:
            active_df = self.original_df.copy()

        self.working_df = active_df.copy()
        self.custom_bounds = custom_bounds or {}
        self.custom_rules = custom_rules or []
        self.log_step(AgentStep.INGEST, f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

        # 2. PROFILE
        self.profile_data = DatasetProfiler.profile(self.working_df)
        self.log_step(AgentStep.PROFILE, "Calculated dataset distributions and summary statistics.")

        # 3. DETECT
        initial_score, findings = QualityEngine.evaluate(
            self.working_df,
            custom_bounds=self.custom_bounds,
            custom_rules=self.custom_rules
        )
        self.context.initial_score = initial_score
        self.context.current_score = initial_score
        self.context.findings = findings
        self.log_step(AgentStep.DETECT, f"Evaluated 8 quality rules. Identified {len(findings)} findings.")

        # 4. TRIAGE
        self.log_step(AgentStep.TRIAGE, f"Computed quality score {initial_score.score}/100. Ranked findings by severity.")

        # 5. INVESTIGATE
        inv_results = self.investigator.investigate(findings, self.working_df, self.dataset_name)
        self.context.investigation_notes = inv_results.get("hypotheses", [])
        self.log_step(AgentStep.INVESTIGATE, f"Meterless Markovian investigation completed. Tokens saved: {inv_results.get('tokens_saved', 0)}.")

        # 6. RECOMMEND
        self.proposal = RemediationEngine.generate_proposals(findings, self.working_df)
        self.log_step(AgentStep.RECOMMEND, f"Generated proposal with {len(self.proposal.actions)} safe remediation actions.")

        # 7. VALIDATE
        sim_df, preview_score, diffs = RemediationEngine.preview_remediation(self.working_df, self.proposal.actions)
        self.preview_score = preview_score
        self.preview_diffs = diffs
        self.log_step(AgentStep.VALIDATE, f"Simulated transformations. Predicted score delta: {initial_score.score} -> {preview_score.score}.")

        # 8. REVIEW (Pause for human-in-the-loop approval)
        self.context.current_step = AgentStep.REVIEW
        self.context.status = WorkflowStatus.WAITING_APPROVAL
        self.log_step(AgentStep.REVIEW, "Awaiting explicit human approval before modifying working copy.")

        return self.context

    def apply_approved_and_audit(self, approved_action_ids: List[str]) -> Tuple[pd.DataFrame, AuditRecord]:
        """Executes APPLY -> AUDIT for approved remediation actions strictly on working copy."""
        if self.original_df is None or self.proposal is None:
            raise ValueError("Pipeline has not been executed through REVIEW stage.")

        # 9. APPLY
        actions_to_apply = [a for a in self.proposal.actions if a.action_id in approved_action_ids]
        for a in self.proposal.actions:
            if a.action_id in approved_action_ids:
                a.approved = True
                a.status = "APPLIED"
            else:
                a.approved = False
                a.status = "REJECTED_BY_USER"

        work_df, final_score, diffs = RemediationEngine.preview_remediation(self.original_df, actions_to_apply)
        self.working_df = work_df
        self.context.current_score = final_score
        self.log_step(AgentStep.APPLY, f"Applied {len(actions_to_apply)} approved transformations on working copy ({len(self.proposal.actions) - len(actions_to_apply)} rejected).")

        # 10. AUDIT
        audit = Auditor.create_audit(
            run_id=self.context.run_id,
            dataset_name=self.dataset_name,
            df_before=self.original_df,
            df_after=self.working_df,
            score_before=self.context.initial_score,
            score_after=final_score,
            applied_actions=actions_to_apply
        )
        self.audit_record = audit
        self.context.current_step = AgentStep.AUDIT
        self.context.status = WorkflowStatus.COMPLETED
        self.log_step(AgentStep.AUDIT, "Generated immutable audit trail and verification certificate.")

        return self.working_df, audit

    def export_dataset(self, format: str = "csv") -> bytes:
        """Exports the cleaned working dataset as CSV or Excel bytes."""
        if self.working_df is None:
            raise ValueError("No working dataset available for export.")
        
        fmt = format.lower().strip()
        if fmt in ["xlsx", "excel"]:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                self.working_df.to_excel(writer, index=False, sheet_name="CleanedData")
            buffer.seek(0)
            return buffer.getvalue()
        else:
            return self.working_df.to_csv(index=False).encode("utf-8")
