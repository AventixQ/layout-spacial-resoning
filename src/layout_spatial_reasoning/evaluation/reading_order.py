"""Reading order violation detector."""

from layout_spatial_reasoning.schemas.form import OrderConstraint
from layout_spatial_reasoning.schemas.layout import Layout


def layout_reading_positions(layout: Layout) -> dict[str, int]:
    """Return top-to-bottom, left-to-right positions for controls in a layout."""
    positions: dict[str, int] = {}
    index = 0
    for section in layout.sections:
        for row in section.rows:
            for control in sorted(row.controls, key=lambda item: item.colStart):
                positions[control.id] = index
                index += 1
    return positions


def reading_order_violation_count(
    layout: Layout,
    constraints: list[OrderConstraint],
) -> int:
    """Count violated partial-order constraints."""
    positions = layout_reading_positions(layout)
    violations = 0

    for constraint in constraints:
        before_position = positions.get(constraint.before)
        after_position = positions.get(constraint.after)
        if before_position is None or after_position is None:
            continue
        if before_position > after_position:
            violations += 1

    return violations


def reading_order_violation_rate(
    layout: Layout,
    constraints: list[OrderConstraint],
) -> float:
    """Return the proportion of violated reading-order constraints."""
    if not constraints:
        return 0.0
    return reading_order_violation_count(layout, constraints) / len(constraints)


def reading_order_violation(
    layout: Layout,
    constraints: list[OrderConstraint],
) -> float:
    """Backward-compatible alias for the reading-order violation rate."""
    return reading_order_violation_rate(layout, constraints)
