from layout_spatial_reasoning.llm.order_extractor import validate_order_constraints
from layout_spatial_reasoning.schemas import Control, OrderConstraint


def test_validate_order_constraints_removes_invalid_pairs():
    controls = [
        Control(id="c01", label="Password", type="text"),
        Control(id="c02", label="Confirm password", type="text"),
    ]
    constraints = [
        OrderConstraint(before="c01", after="c02"),
        OrderConstraint(before="c01", after="c02"),
        OrderConstraint(before="c02", after="c01"),
        OrderConstraint(before="c01", after="c01"),
        OrderConstraint(before="missing", after="c02"),
    ]

    assert validate_order_constraints(constraints, controls) == [
        OrderConstraint(before="c01", after="c02")
    ]
