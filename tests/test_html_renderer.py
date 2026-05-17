from layout_spatial_reasoning.rendering.html_renderer import render_html
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control


def test_render_html_contains_control_label_and_grid_column():
    controls = [Control(id="c01", label="Email address", type="text")]
    layout = generate_layout(controls)

    rendered = render_html(controls, layout)

    assert "Email address" in rendered
    assert "grid-column: 1 / span 12" in rendered
