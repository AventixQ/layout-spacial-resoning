from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control, Layout


def test_sequential_baseline_returns_layout():
    layout = generate_layout([Control(id="c01", label="Email", type="text")])

    assert isinstance(layout, Layout)
