from layout_spatial_reasoning.evaluation.metrics import grid_utilization
from layout_spatial_reasoning.evaluation.metrics import evaluate_generated_layout
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control, FormSpec, GeneratedLayoutRecord, OrderConstraint


def test_grid_utilization_for_full_width_rows():
    layout = generate_layout([Control(id="c01", label="Email", type="text")])

    assert grid_utilization(layout) == 1.0


def test_evaluate_generated_layout_counts_reading_order_violation():
    form = FormSpec(
        form_id="registration",
        domain="User registration",
        controls=[
            Control(id="c01", label="Password", type="text"),
            Control(id="c02", label="Confirm password", type="text"),
        ],
        order_constraints=[OrderConstraint(before="c01", after="c02")],
    )
    layout = generate_layout(list(reversed(form.controls)))
    generated = GeneratedLayoutRecord(
        form_id=form.form_id,
        method="sequential",
        layout=layout,
    )

    record = evaluate_generated_layout(form, generated)

    assert record.reading_order_violation_count == 1
    assert record.reading_order_violation_rate == 1.0
    assert record.has_reading_order_violation is True


def test_evaluate_generated_layout_marks_reading_order_as_not_applicable():
    form = FormSpec(
        form_id="contact",
        domain="Contact",
        controls=[Control(id="c01", label="Message", type="long_text")],
        order_constraints=[],
    )
    layout = generate_layout(form.controls)
    generated = GeneratedLayoutRecord(
        form_id=form.form_id,
        method="sequential",
        layout=layout,
    )

    record = evaluate_generated_layout(form, generated)

    assert record.reading_order_constraint_count == 0
    assert record.reading_order_violation_rate is None
    assert record.has_reading_order_violation is None
