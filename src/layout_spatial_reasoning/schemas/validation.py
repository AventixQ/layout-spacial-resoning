"""Validation helpers for layout schemas."""

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
    duplicated = {control_id for control_id in actual if actual.count(control_id) > 1}

    if missing:
        errors.append(f"Missing controls: {sorted(missing)}")
    if unknown:
        errors.append(f"Unknown controls: {sorted(unknown)}")
    if duplicated:
        errors.append(f"Duplicated controls: {sorted(duplicated)}")
    return errors
