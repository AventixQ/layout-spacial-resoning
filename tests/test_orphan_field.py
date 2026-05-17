from layout_spatial_reasoning.evaluation.orphan_field import (
    detect_orphan_fields,
    orphan_field_count,
)
from layout_spatial_reasoning.schemas import Control, Layout, LayoutControl, Row, Section


def test_detects_control_that_belongs_to_another_section():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
        Control(id="c04", label="Shipping date", type="date"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Personal",
                rows=[
                    Row(row_id="r001", controls=[LayoutControl(id="c01", colStart=1, colSpan=6)]),
                    Row(row_id="r002", controls=[LayoutControl(id="c02", colStart=1, colSpan=6)]),
                    Row(row_id="r003", controls=[LayoutControl(id="c03", colStart=1, colSpan=6)]),
                ],
            ),
            Section(
                section_id="s002",
                section_name="Delivery",
                rows=[
                    Row(row_id="r004", controls=[LayoutControl(id="c04", colStart=1, colSpan=6)]),
                ],
            ),
        ]
    )

    cases = detect_orphan_fields(
        controls,
        layout,
        embedding_function=_fake_embeddings,
        margin_threshold=0.05,
    )

    assert any(
        case.control_id == "c03" and case.best_section_id == "s002"
        for case in cases
    )


def test_ignores_coherent_sections():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
        Control(id="c04", label="Shipping date", type="date"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Personal",
                rows=[
                    Row(row_id="r001", controls=[LayoutControl(id="c01", colStart=1, colSpan=6)]),
                    Row(row_id="r002", controls=[LayoutControl(id="c02", colStart=1, colSpan=6)]),
                ],
            ),
            Section(
                section_id="s002",
                section_name="Delivery",
                rows=[
                    Row(row_id="r003", controls=[LayoutControl(id="c03", colStart=1, colSpan=6)]),
                    Row(row_id="r004", controls=[LayoutControl(id="c04", colStart=1, colSpan=6)]),
                ],
            ),
        ]
    )

    assert orphan_field_count(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    ) == 0


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "First name": [1.0, 0.0],
        "Last name": [1.0, 0.0],
        "Delivery date": [0.0, 1.0],
        "Shipping date": [0.0, 1.0],
    }
    return [vectors[text] for text in texts]
