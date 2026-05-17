"""Layout quality metrics."""

from collections import Counter

from layout_spatial_reasoning.evaluation.grid_constraints import grid_constraint_violations
from layout_spatial_reasoning.evaluation.reading_order import (
    reading_order_violation_count,
    reading_order_violation_rate,
)
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.form import FormSpec
from layout_spatial_reasoning.schemas.layout import Layout
from layout_spatial_reasoning.schemas.results import EvaluationRecord, GeneratedLayoutRecord
from layout_spatial_reasoning.schemas.validation import validate_layout


def grid_utilization(layout: Layout, columns: int = 12) -> float:
    row_fill = [
        sum(control.colSpan for control in row.controls) / columns
        for section in layout.sections
        for row in section.rows
    ]
    if not row_fill:
        return 0.0
    return sum(row_fill) / len(row_fill)


def control_issue_counts(controls: list[Control], layout: Layout) -> dict[str, int]:
    """Count missing, duplicated, and unknown controls in a generated layout."""
    expected = {control.id for control in controls}
    actual = [
        layout_control.id
        for section in layout.sections
        for row in section.rows
        for layout_control in row.controls
    ]
    counts = Counter(actual)

    return {
        "missing": len(expected.difference(actual)),
        "duplicated": sum(1 for count in counts.values() if count > 1),
        "unknown": len(set(actual).difference(expected)),
    }


def evaluate_generated_layout(
    form: FormSpec,
    generated: GeneratedLayoutRecord,
    *,
    columns: int = 12,
) -> EvaluationRecord:
    """Compute deterministic metrics for one generated layout."""
    control_counts = control_issue_counts(form.controls, generated.layout)
    grid_errors = grid_constraint_violations(generated.layout, columns=columns)
    reading_order_errors = reading_order_violation_count(
        generated.layout,
        form.order_constraints,
    )

    return EvaluationRecord(
        form_id=generated.form_id,
        method=generated.method,
        grid_utilization=grid_utilization(generated.layout, columns=columns),
        validation_error_count=len(
            validate_layout(form.controls, generated.layout, columns=columns)
        ),
        missing_control_count=control_counts["missing"],
        duplicated_control_count=control_counts["duplicated"],
        unknown_control_count=control_counts["unknown"],
        grid_constraint_violation_count=len(grid_errors),
        reading_order_constraint_count=len(form.order_constraints),
        reading_order_violation_count=reading_order_errors,
        reading_order_violation_rate=reading_order_violation_rate(
            generated.layout,
            form.order_constraints,
        ),
    )
