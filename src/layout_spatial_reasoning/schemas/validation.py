"""Validation helpers for layout schemas."""

from collections import Counter

from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout


def validate_control_completeness(controls: list[Control], layout: Layout) -> list[str]:
    """Return validation errors for missing, duplicated, or unknown controls."""
    expected = {control.id for control in controls}
    actual = [
        layout_control.id
        for section in layout.sections
        for row in section.rows
        for layout_control in row.controls
    ]

    errors: list[str] = []
    missing = expected.difference(actual)
    unknown = set(actual).difference(expected)
    duplicated = {
        control_id for control_id, count in Counter(actual).items() if count > 1
    }

    if missing:
        errors.append(f"Missing controls: {sorted(missing)}")
    if unknown:
        errors.append(f"Unknown controls: {sorted(unknown)}")
    if duplicated:
        errors.append(f"Duplicated controls: {sorted(duplicated)}")
    return errors


def validate_grid_constraints(layout: Layout, columns: int = 12) -> list[str]:
    """Return validation errors for range and overlap violations."""
    errors: list[str] = []

    for section in layout.sections:
        for row in section.rows:
            occupied: set[int] = set()
            for control in row.controls:
                start = control.colStart
                end = control.colStart + control.colSpan - 1
                if end > columns:
                    errors.append(
                        f"{control.id} extends beyond column {columns} in {row.row_id}."
                    )

                current = set(range(start, end + 1))
                if occupied.intersection(current):
                    errors.append(f"{control.id} overlaps another control in {row.row_id}.")
                occupied.update(current)

    return errors


def validate_layout(controls: list[Control], layout: Layout, columns: int = 12) -> list[str]:
    """Return all deterministic structural validation errors."""
    return [
        *validate_control_completeness(controls, layout),
        *validate_grid_constraints(layout, columns=columns),
    ]
