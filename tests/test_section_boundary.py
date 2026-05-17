from layout_spatial_reasoning.evaluation.section_boundary import section_boundary_score
from layout_spatial_reasoning.schemas import Control, Layout, LayoutControl, Row, Section


def test_section_boundary_score_is_zero_for_coherent_sections():
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

    assert section_boundary_score(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    ) == 0.0


def test_section_boundary_score_penalizes_bad_section_split():
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
                section_name="Mixed A",
                rows=[
                    Row(row_id="r001", controls=[LayoutControl(id="c01", colStart=1, colSpan=6)]),
                    Row(row_id="r002", controls=[LayoutControl(id="c03", colStart=1, colSpan=6)]),
                ],
            ),
            Section(
                section_id="s002",
                section_name="Mixed B",
                rows=[
                    Row(row_id="r003", controls=[LayoutControl(id="c02", colStart=1, colSpan=6)]),
                    Row(row_id="r004", controls=[LayoutControl(id="c04", colStart=1, colSpan=6)]),
                ],
            ),
        ]
    )

    assert section_boundary_score(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    ) == 0.5


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "First name": [1.0, 0.0],
        "Last name": [1.0, 0.0],
        "Delivery date": [0.0, 1.0],
        "Shipping date": [0.0, 1.0],
    }
    return [vectors[text] for text in texts]
