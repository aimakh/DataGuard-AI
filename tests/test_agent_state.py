import pytest
import pandas as pd
from agent.orchestrator import DataGuardOrchestrator
from models.state import AgentStep, WorkflowStatus


def test_orchestrator_lifecycle():
    df = pd.read_csv("/home/aima/open-agent-hack/data/sample_customers.csv")
    orch = DataGuardOrchestrator(dataset_name="sample_customers")
    
    # Run through review
    context = orch.run_lifecycle_to_review(df)
    assert context.current_step == AgentStep.REVIEW
    assert context.status == WorkflowStatus.WAITING_APPROVAL
    assert context.initial_score is not None
    assert orch.preview_score is not None
    assert orch.preview_score.score > context.initial_score.score

    # Approve all actions
    all_action_ids = [a.action_id for a in orch.proposal.actions]
    working_df, audit = orch.apply_approved_and_audit(all_action_ids)

    assert orch.context.current_step == AgentStep.AUDIT
    assert orch.context.status == WorkflowStatus.COMPLETED
    assert audit.verification_passed
    assert audit.after_score > audit.before_score
    assert len(working_df) > 0
