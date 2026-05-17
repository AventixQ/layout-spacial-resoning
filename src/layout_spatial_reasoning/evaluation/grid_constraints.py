"""Hard twelve-column grid constraint checks."""

from layout_spatial_reasoning.schemas.layout import Layout
from layout_spatial_reasoning.schemas.validation import validate_grid_constraints


def grid_constraint_violations(layout: Layout, columns: int = 12) -> list[str]:
    return validate_grid_constraints(layout, columns=columns)
