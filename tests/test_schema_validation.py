from layout_spatial_reasoning.schemas import Control
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas.validation import validate_control_completeness


def test_sequential_baseline_contains_all_controls_once():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
    ]

    layout = generate_layout(controls)

    assert validate_control_completeness(controls, layout) == []
