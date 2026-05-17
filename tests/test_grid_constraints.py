from layout_spatial_reasoning.evaluation.grid_constraints import grid_constraint_violations
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control


def test_sequential_baseline_satisfies_grid_constraints():
    layout = generate_layout([Control(id="c01", label="Email", type="text")])

    assert grid_constraint_violations(layout) == []
