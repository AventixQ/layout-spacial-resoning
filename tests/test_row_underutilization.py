from layout_spatial_reasoning.evaluation.row_underutilization import (
    detect_row_underutilization,
    row_underutilization_count,
)
from layout_spatial_reasoning.schemas import Control, Layout, LayoutControl, Row, Section


def test_detects_adjacent_related_full_width_rows():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Personal",
                rows=[
                    Row(row_id="r001", controls=[LayoutControl(id="c01", colStart=1, colSpan=12)]),
                    Row(row_id="r002", controls=[LayoutControl(id="c02", colStart=1, colSpan=12)]),
                    Row(row_id="r003", controls=[LayoutControl(id="c03", colStart=1, colSpan=12)]),
                ],
            )
        ]
    )

    cases = detect_row_underutilization(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    )

    assert len(cases) == 1
    assert cases[0].first_control_id == "c01"
    assert cases[0].second_control_id == "c02"


def test_ignores_related_controls_already_in_same_row():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Personal",
                rows=[
                    Row(
                        row_id="r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    assert row_underutilization_count(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    ) == 0


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "First name": [1.0, 0.0],
        "Last name": [1.0, 0.0],
        "Delivery date": [0.0, 1.0],
    }
    return [vectors[text] for text in texts]
