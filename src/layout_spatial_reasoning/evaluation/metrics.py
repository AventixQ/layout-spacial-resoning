"""Layout quality metrics."""

from layout_spatial_reasoning.schemas.layout import Layout


def grid_utilization(layout: Layout, columns: int = 12) -> float:
    row_fill = [
        sum(control.colSpan for control in row.controls) / columns
        for section in layout.sections
        for row in section.rows
    ]
    if not row_fill:
        return 0.0
    return sum(row_fill) / len(row_fill)
