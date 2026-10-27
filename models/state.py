from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .findings import Finding, QualityScore


class AgentStep(str, Enum):
    INGEST = "INGEST"
    PROFILE = "PROFILE"
    DETECT = "DETECT"
    TRIAGE = "TRIAGE"
    INVESTIGATE = "INVESTIGATE"
    RECOMMEND = "RECOMMEND"
    VALIDATE = "VALIDATE"
    REVIEW = "REVIEW"
    APPLY = "APPLY"
    AUDIT = "AUDIT"


class WorkflowStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentRunContext(BaseModel):
    run_id: str
    dataset_name: str
    current_step: AgentStep = AgentStep.INGEST
    status: WorkflowStatus = WorkflowStatus.IDLE
    initial_score: Optional[QualityScore] = None
    current_score: Optional[QualityScore] = None
    findings: List[Finding] = Field(default_factory=list)
    investigation_notes: List[str] = Field(default_factory=list)
    step_history: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None
