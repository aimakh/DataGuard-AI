import pytest
import pandas as pd
import numpy as np

from agent.orchestrator import DataGuardOrchestrator
from agent.remediation_engine import RemediationEngine
from agent.auditor import Auditor
from models.state import AgentStep, WorkflowStatus
from integrations.zetaris_client import ZetarisClient
from integrations.meterless_nightshift import MeterlessNightshiftClient
from integrations.meterless_engine import MarkovianEngine
from integrations.nemotron import NemotronReasoner


def test_remediation_and_audit_flow():
    df_raw = pd.read_csv("/home/aima/open-agent-hack/data/sample_customers.csv")
    orch = DataGuardOrchestrator(dataset_name="sample_customers")
    
    # Run lifecycle up to REVIEW
    ctx = orch.run_lifecycle_to_review(df_raw)
    assert ctx.current_step == AgentStep.REVIEW
    assert ctx.status == WorkflowStatus.WAITING_APPROVAL
    assert len(orch.proposal.actions) > 0
    assert ctx.initial_score is not None
    initial_score_val = ctx.initial_score.score
    
    # Approve all actions
    all_action_ids = [a.action_id for a in orch.proposal.actions]
    working_df, audit_record = orch.apply_approved_and_audit(all_action_ids)
    
    assert ctx.current_step == AgentStep.AUDIT
    assert ctx.status == WorkflowStatus.COMPLETED
    assert audit_record is not None
    assert audit_record.verification_passed is True
    assert audit_record.after_score >= initial_score_val
    assert audit_record.content_hash_before != audit_record.content_hash_after
    assert len(audit_record.applied_actions) == len(all_action_ids)
    assert "Cryptographically Sealed" in audit_record.summary_markdown
    
    # Verify original df was not mutated
    assert len(df_raw) == 503
    assert df_raw.isna().sum().sum() > 0


def test_remediation_selective_approval():
    df_raw = pd.read_csv("/home/aima/open-agent-hack/data/sample_customers.csv")
    orch = DataGuardOrchestrator(dataset_name="sample_customers")
    ctx = orch.run_lifecycle_to_review(df_raw)
    
    # Approve only the first action
    first_action_id = orch.proposal.actions[0].action_id
    working_df, audit_record = orch.apply_approved_and_audit([first_action_id])
    
    assert len(audit_record.applied_actions) == 1
    assert audit_record.applied_actions[0]["action_id"] == first_action_id


def test_meterless_engine_telemetry():
    engine = MarkovianEngine(chunk_size=4000, max_chunks=3, carryover_tokens=128)
    
    def test_generator(prompt, carryover):
        if not carryover:
            return "Step 1 analysis. [STATE_CHECKPOINT] Anomaly detected in date format."
        return "Step 2 resolved. Root cause found. [TASK_COMPLETE]"

    run = engine.run_chain("Investigate date formats", test_generator)
    assert run.completed is True
    assert len(run.chunks) == 2
    assert run.total_tokens_saved > 0


def test_nemotron_investigation():
    nemotron = NemotronReasoner()
    df_raw = pd.read_csv("/home/aima/open-agent-hack/data/sample_customers.csv")
    orch = DataGuardOrchestrator(dataset_name="sample_customers")
    orch.run_lifecycle_to_review(df_raw)
    
    analysis = nemotron.synthesize_investigation(
        findings=orch.context.findings,
        dataset_name="sample_customers"
    )
    assert "root_cause_analysis" in analysis
    assert "hypotheses" in analysis
    assert len(analysis["hypotheses"]) > 0
    assert float(analysis["confidence_score"]) > 0.0


def test_zetaris_live_status():
    client = ZetarisClient()
    status = client.check_health()
    assert "api_online" in status
