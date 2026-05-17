from layout_spatial_reasoning.methods.graph_community import (
    build_similarity_graph,
    detect_communities,
    generate_layout,
)
from layout_spatial_reasoning.schemas import Control
from layout_spatial_reasoning.schemas.validation import validate_layout


def test_graph_method_returns_valid_layout():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Shipping address", type="long_text"),
        Control(id="c04", label="Postal code", type="text"),
    ]

    layout = generate_layout(controls)

    assert validate_layout(controls, layout) == []


def test_similarity_graph_links_related_labels_with_custom_embeddings():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
    ]

    def fake_embeddings(_texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    graph = build_similarity_graph(
        controls,
        embedding_function=fake_embeddings,
        similarity_threshold=0.9,
    )

    assert graph.has_edge("c01", "c02")
    assert not graph.has_edge("c01", "c03")


def test_leiden_groups_connected_related_controls():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Delivery date", type="date"),
    ]

    def fake_embeddings(_texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    graph = build_similarity_graph(
        controls,
        embedding_function=fake_embeddings,
        similarity_threshold=0.9,
    )
    communities = detect_communities(graph, algorithm="leiden")
    community_ids = [{control.id for control in community} for community in communities]

    assert {"c01", "c02"} in community_ids
    assert {"c03"} in community_ids
