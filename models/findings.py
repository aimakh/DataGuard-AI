from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QualityCategory(str, Enum):
    NULL_VALUES = "NULL_VALUES"
    DUPLICATES = "DUPLICATES"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    BOUNDS_VIOLATION = "BOUNDS_VIOLATION"
    FORMAT_INCONSISTENCY = "FORMAT_INCONSISTENCY"
    CATEGORY_NORMALIZATION = "CATEGORY_NORMALIZATION"
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"
    SEMANTIC_CONTRADICTION = "SEMANTIC_CONTRADICTION"
    CONSTANT_COLUMN = "CONSTANT_COLUMN"
    HIGHLY_INCOMPLETE = "HIGHLY_INCOMPLETE"
    SUSPICIOUS_IDENTIFIER = "SUSPICIOUS_IDENTIFIER"
    ML_PREPARATION_RISK = "ML_PREPARATION_RISK"


class Finding(BaseModel):
    finding_id: str
    rule_name: str
    category: QualityCategory
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    affected_columns: List[str]
    affected_rows_count: int
    affected_row_indices: List[int]
    evidence: Dict[str, Any] = Field(default_factory=dict)
    method: str
    description: str
    default_fix_type: str


class RuleResult(BaseModel):
    rule_name: str
    category: QualityCategory
    passed: bool
    severity: Severity
    penalty: float
    affected_rows_count: int
    findings: List[Finding] = Field(default_factory=list)
    message: str


class QualityScore(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    rule_results: List[RuleResult] = Field(default_factory=list)
    category_breakdown: Dict[str, float] = Field(default_factory=dict)
    total_rows: int
    total_columns: int
    total_findings: int
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
