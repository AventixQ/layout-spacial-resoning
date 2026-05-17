from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.methods.random_baseline import generate_layout as random_layout
from layout_spatial_reasoning.schemas import Control, Layout
from layout_spatial_reasoning.schemas.validation import validate_layout


def test_sequential_baseline_returns_layout():
    layout = generate_layout([Control(id="c01", label="Email", type="text")])

    assert isinstance(layout, Layout)


def test_random_baseline_returns_valid_layout():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Biography", type="long_text"),
    ]

    layout = random_layout(controls, seed=123)

    assert validate_layout(controls, layout) == []
