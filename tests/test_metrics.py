from layout_spatial_reasoning.evaluation.metrics import grid_utilization
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control


def test_grid_utilization_for_full_width_rows():
    layout = generate_layout([Control(id="c01", label="Email", type="text")])

    assert grid_utilization(layout) == 1.0
