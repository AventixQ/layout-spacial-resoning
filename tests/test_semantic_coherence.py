from layout_spatial_reasoning.evaluation.semantic_coherence import (
    semantic_coherence,
    within_row_coherence,
    within_section_coherence,
)
from layout_spatial_reasoning.schemas import Control, Layout, LayoutControl, Row, Section


def test_within_row_coherence_averages_rows_with_pairs():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Account",
                rows=[
                    Row(
                        row_id="r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                        ],
                    ),
                    Row(
                        row_id="r002",
                        controls=[LayoutControl(id="c03", colStart=1, colSpan=12)],
                    ),
                ],
            )
        ]
    )

    assert within_row_coherence(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    ) == 1.0


def test_within_section_coherence_averages_all_pairs_in_section():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Mixed",
                rows=[
                    Row(
                        row_id="r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                        ],
                    ),
                    Row(
                        row_id="r002",
                        controls=[LayoutControl(id="c03", colStart=1, colSpan=12)],
                    ),
                ],
            )
        ]
    )

    assert within_section_coherence(
        controls,
        layout,
        embedding_function=_fake_embeddings,
    ) == 1 / 3


def test_semantic_coherence_rejects_unknown_level():
    try:
        semantic_coherence([], Layout(), level="field")
    except ValueError as error:
        assert str(error) == "level must be either 'row' or 'section'."
    else:
        raise AssertionError("Expected ValueError.")


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "First name": [1.0, 0.0],
        "Last name": [1.0, 0.0],
        "Delivery date": [0.0, 1.0],
    }
    return [vectors[text] for text in texts]
