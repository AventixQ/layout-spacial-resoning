"""Schemas for generated method outputs and evaluation results."""

from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.schemas.layout import Layout


class GeneratedLayoutRecord(BaseModel):
    """One generated layout produced by one method for one form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    layout: Layout


class EvaluationRecord(BaseModel):
    """Deterministic evaluation metrics for one generated layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form_id: str
    method: str
    grid_utilization: float
    semantic_coherence_row: float
    semantic_coherence_section: float
    has_grid_constraint_violation: bool
    has_row_underutilization: bool
    has_orphan_field: bool
    has_section_boundary_misplacement: bool
    has_reading_order_violation: bool | None
    row_underutilization_count: int
    orphan_field_count: int
    section_boundary_misplacement_score: float
    validation_error_count: int
    missing_control_count: int
    duplicated_control_count: int
    unknown_control_count: int
    grid_constraint_violation_count: int
    reading_order_constraint_count: int
    reading_order_violation_count: int
    reading_order_violation_rate: float | None
