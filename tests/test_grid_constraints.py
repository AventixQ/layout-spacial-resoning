from layout_spatial_reasoning.evaluation.grid_constraints import grid_constraint_violations
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control, Layout, LayoutControl, Row, Section


def test_sequential_baseline_satisfies_grid_constraints():
    layout = generate_layout([Control(id="c01", label="Email", type="text")])

    assert grid_constraint_violations(layout) == []


def test_detects_row_overlap():
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Broken",
                rows=[
                    Row(
                        row_id="r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=6, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    assert grid_constraint_violations(layout) == [
        "c02 overlaps another control in r001."
    ]
