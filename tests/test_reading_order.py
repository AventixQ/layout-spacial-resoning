from layout_spatial_reasoning.evaluation.reading_order import (
    layout_reading_positions,
    reading_order_violation_count,
    reading_order_violation_rate,
)
from layout_spatial_reasoning.schemas import Layout, LayoutControl, OrderConstraint, Row, Section


def test_reading_positions_are_left_to_right_within_row():
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Account",
                rows=[
                    Row(
                        row_id="r001",
                        controls=[
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    assert layout_reading_positions(layout) == {"c01": 0, "c02": 1}


def test_reading_order_violation_rate():
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Account",
                rows=[
                    Row(row_id="r001", controls=[LayoutControl(id="c02", colStart=1, colSpan=12)]),
                    Row(row_id="r002", controls=[LayoutControl(id="c01", colStart=1, colSpan=12)]),
                ],
            )
        ]
    )
    constraints = [
        OrderConstraint(before="c01", after="c02"),
        OrderConstraint(before="c02", after="c03"),
    ]

    assert reading_order_violation_count(layout, constraints) == 1
    assert reading_order_violation_rate(layout, constraints) == 0.5


def test_reading_order_violation_rate_is_none_without_constraints():
    assert reading_order_violation_rate(Layout(), []) is None
