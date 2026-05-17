"""Layout quality metrics."""

from collections import Counter
from collections.abc import Callable

from layout_spatial_reasoning.embeddings.provider import embed_texts
from layout_spatial_reasoning.evaluation.grid_constraints import grid_constraint_violations
from layout_spatial_reasoning.evaluation.orphan_field import orphan_field_count
from layout_spatial_reasoning.evaluation.reading_order import (
    reading_order_violation_count,
    reading_order_violation_rate,
)
from layout_spatial_reasoning.evaluation.row_underutilization import (
    row_underutilization_count,
)
from layout_spatial_reasoning.evaluation.section_boundary import section_boundary_score
from layout_spatial_reasoning.evaluation.semantic_coherence import (
    within_row_coherence,
    within_section_coherence,
)
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.form import FormSpec
from layout_spatial_reasoning.schemas.layout import Layout
from layout_spatial_reasoning.schemas.results import EvaluationRecord, GeneratedLayoutRecord
from layout_spatial_reasoning.schemas.validation import validate_layout

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


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
    embedding_function: EmbeddingFunction = embed_texts,
) -> EvaluationRecord:
    """Compute deterministic metrics for one generated layout."""
    control_counts = control_issue_counts(form.controls, generated.layout)
    grid_errors = grid_constraint_violations(generated.layout, columns=columns)
    row_underutilization_errors = row_underutilization_count(
        form.controls,
        generated.layout,
        embedding_function=embedding_function,
    )
    orphan_field_errors = orphan_field_count(
        form.controls,
        generated.layout,
        embedding_function=embedding_function,
    )
    section_boundary_errors = section_boundary_score(
        form.controls,
        generated.layout,
        embedding_function=embedding_function,
    )
    reading_order_errors = reading_order_violation_count(
        generated.layout,
        form.order_constraints,
    )
    reading_order_rate = reading_order_violation_rate(
        generated.layout,
        form.order_constraints,
    )

    return EvaluationRecord(
        form_id=generated.form_id,
        method=generated.method,
        grid_utilization=grid_utilization(generated.layout, columns=columns),
        semantic_coherence_row=within_row_coherence(
            form.controls,
            generated.layout,
            embedding_function=embedding_function,
        ),
        semantic_coherence_section=within_section_coherence(
            form.controls,
            generated.layout,
            embedding_function=embedding_function,
        ),
        has_grid_constraint_violation=bool(grid_errors),
        has_row_underutilization=row_underutilization_errors > 0,
        has_orphan_field=orphan_field_errors > 0,
        has_section_boundary_misplacement=section_boundary_errors > 0,
        has_reading_order_violation=(
            None if reading_order_rate is None else reading_order_errors > 0
        ),
        row_underutilization_count=row_underutilization_errors,
        orphan_field_count=orphan_field_errors,
        section_boundary_misplacement_score=section_boundary_errors,
        validation_error_count=len(
            validate_layout(form.controls, generated.layout, columns=columns)
        ),
        missing_control_count=control_counts["missing"],
        duplicated_control_count=control_counts["duplicated"],
        unknown_control_count=control_counts["unknown"],
        grid_constraint_violation_count=len(grid_errors),
        reading_order_constraint_count=len(form.order_constraints),
        reading_order_violation_count=reading_order_errors,
        reading_order_violation_rate=reading_order_rate,
    )
