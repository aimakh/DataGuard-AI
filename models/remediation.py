from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class FixType(str, Enum):
    IMPUTE_MEDIAN = "IMPUTE_MEDIAN"
    IMPUTE_MEAN = "IMPUTE_MEAN"
    IMPUTE_MODE = "IMPUTE_MODE"
    IMPUTE_CONSTANT = "IMPUTE_CONSTANT"
    DROP_NULL_ROWS = "DROP_NULL_ROWS"
    DROP_COLUMN = "DROP_COLUMN"
    DROP_DUPLICATES = "DROP_DUPLICATES"
    FLAG_DUPLICATES = "FLAG_DUPLICATES"
    KEEP_DUPLICATES = "KEEP_DUPLICATES"
    CLAMP_BOUNDS = "CLAMP_BOUNDS"
    STANDARDIZE_DATE = "STANDARDIZE_DATE"
    CLEAN_EMAIL = "CLEAN_EMAIL"
    NORMALIZE_CATEGORY = "NORMALIZE_CATEGORY"
    REMOVE_OUTLIERS = "REMOVE_OUTLIERS"
    REPAIR_CONTRADICTION = "REPAIR_CONTRADICTION"
    ONE_HOT_ENCODE = "ONE_HOT_ENCODE"
    ORDINAL_ENCODE = "ORDINAL_ENCODE"
    STANDARDIZE_FEATURES = "STANDARDIZE_FEATURES"
    MIN_MAX_SCALE = "MIN_MAX_SCALE"
    CUSTOM_TRANSFORM = "CUSTOM_TRANSFORM"


class RemediationAction(BaseModel):
    action_id: str
    finding_id: str
    title: str
    description: str
    fix_type: FixType
    target_columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    estimated_impact_rows: int
    safety_level: str = "SAFE"  # SAFE, CAUTION, DESTRUCTIVE
    requires_approval: bool = True
    approved: bool = False
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED, APPLIED
    why_it_matters: str = ""
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    risks_assumptions: str = ""


class RemediationProposal(BaseModel):
    proposal_id: str
    root_cause_summary: str
    actions: List[RemediationAction] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    model_used: str = "deterministic-rule-engine"


class AuditRecord(BaseModel):
    run_id: str
    timestamp: str
    dataset_name: str
    before_score: float
    after_score: float
    findings_count_before: int
    findings_count_after: int
    applied_actions: List[Dict[str, Any]] = Field(default_factory=list)
    verification_passed: bool
    content_hash_before: str
    content_hash_after: str
    summary_markdown: str = ""
