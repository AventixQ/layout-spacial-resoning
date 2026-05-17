"""Hard twelve-column grid constraint checks."""

from layout_spatial_reasoning.schemas.layout import Layout


def grid_constraint_violations(layout: Layout, columns: int = 12) -> list[str]:
    errors: list[str] = []
    for section in layout.sections:
        for row in section.rows:
            occupied: set[int] = set()
            for control in row.controls:
                start = control.colStart
                end = control.colStart + control.colSpan - 1
                if start < 1 or end > columns or control.colSpan < 1:
                    errors.append(f"{control.id} is outside the grid in {row.row_id}.")
                current = set(range(start, end + 1))
                if occupied.intersection(current):
                    errors.append(f"{control.id} overlaps another control in {row.row_id}.")
                occupied.update(current)
    return errors
