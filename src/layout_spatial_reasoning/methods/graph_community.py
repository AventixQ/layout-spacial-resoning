"""Graph-based community detection method."""

from collections import defaultdict
from typing import Callable

import igraph as ig
import leidenalg
import networkx as nx

from layout_spatial_reasoning.config import env_float, env_int, env_str
from layout_spatial_reasoning.embeddings.provider import (
    embed_texts,
    embedding_function_from_env,
)
from layout_spatial_reasoning.embeddings.similarity import cosine_similarity
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, LayoutControl, Row, Section


EmbeddingFunction = Callable[[list[str]], list[list[float]]]

DEFAULT_WIDTHS = {
    "long_text": 12,
    "file": 12,
    "multichoice": 12,
    "date": 6,
    "number": 6,
    "boolean": 6,
    "text": 6,
    "choice": 6,
}


def generate_layout(
    controls: list[Control],
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    similarity_threshold: float = 0.25,
    community_algorithm: str = "leiden",
    seed: int = 42,
    default_widths: dict[str, int] | None = None,
) -> Layout:
    """Generate a layout using semantic graph communities and greedy packing."""
    if not controls:
        return Layout(sections=[])

    graph = build_similarity_graph(
        controls,
        embedding_function=embedding_function,
        similarity_threshold=similarity_threshold,
    )
    communities = detect_communities(
        graph,
        algorithm=community_algorithm,
        seed=seed,
    )
    widths = default_widths or DEFAULT_WIDTHS

    sections = [
        _section_from_controls(index, community_controls, widths)
        for index, community_controls in enumerate(communities, start=1)
    ]
    return Layout(sections=sections)


def generate_layout_from_env(controls: list[Control]) -> Layout:
    """Generate Method 4 layout using environment configuration."""
    return generate_layout(
        controls,
        embedding_function=embedding_function_from_env(),
        similarity_threshold=env_float("GRAPH_SIMILARITY_THRESHOLD", 0.25),
        community_algorithm=env_str("GRAPH_COMMUNITY_ALGORITHM", "leiden") or "leiden",
        seed=env_int("GRAPH_RANDOM_SEED", 42),
    )


def build_similarity_graph(
    controls: list[Control],
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    similarity_threshold: float = 0.25,
) -> nx.Graph:
    """Build a weighted graph where controls are linked by label similarity."""
    labels = [_embedding_text(control) for control in controls]
    embeddings = embedding_function(labels)

    graph = nx.Graph()
    for control in controls:
        graph.add_node(control.id, control=control)

    for left_index, left_control in enumerate(controls):
        for right_index in range(left_index + 1, len(controls)):
            right_control = controls[right_index]
            similarity = cosine_similarity(embeddings[left_index], embeddings[right_index])
            if similarity >= similarity_threshold:
                graph.add_edge(left_control.id, right_control.id, weight=similarity)

    return graph


def detect_communities(
    graph: nx.Graph,
    *,
    algorithm: str = "leiden",
    seed: int = 42,
) -> list[list[Control]]:
    """Detect communities in the similarity graph.

    The thesis method uses Leiden by default because it guarantees internally
    connected communities. Greedy modularity is kept as an explicit fallback
    for debugging or ablation experiments.
    """
    if graph.number_of_nodes() == 0:
        return []

    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    elif algorithm == "leiden":
        communities = _leiden_communities(graph, seed=seed)
    elif algorithm == "greedy_modularity":
        communities = nx.community.greedy_modularity_communities(
            graph,
            weight="weight",
        )
    else:
        raise ValueError("algorithm must be either 'leiden' or 'greedy_modularity'.")

    return [
        sorted(
            (graph.nodes[node]["control"] for node in community),
            key=lambda control: control.id,
        )
        for community in sorted(communities, key=lambda item: min(item))
    ]


def _leiden_communities(graph: nx.Graph, *, seed: int) -> list[set[str]]:
    nodes = sorted(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edges = [
        (node_to_index[left], node_to_index[right])
        for left, right in graph.edges
    ]
    weights = [
        graph.edges[left, right].get("weight", 1.0)
        for left, right in graph.edges
    ]

    igraph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    igraph.vs["name"] = nodes
    igraph.es["weight"] = weights

    partition = leidenalg.find_partition(
        igraph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed,
    )
    return [
        {nodes[index] for index in community}
        for community in partition
    ]


def _section_from_controls(
    section_index: int,
    controls: list[Control],
    widths: dict[str, int],
) -> Section:
    return Section(
        section_id=f"s{section_index:03d}",
        section_name=_section_name(controls),
        rows=_pack_rows(controls, section_index, widths),
    )


def _pack_rows(
    controls: list[Control],
    section_index: int,
    widths: dict[str, int],
) -> list[Row]:
    rows: list[Row] = []
    current_controls: list[LayoutControl] = []
    current_width = 0

    for control in controls:
        width = widths.get(control.type, 6)
        if current_controls and current_width + width > 12:
            rows.append(_row(section_index, len(rows) + 1, current_controls))
            current_controls = []
            current_width = 0

        current_controls.append(
            LayoutControl(id=control.id, colStart=current_width + 1, colSpan=width)
        )
        current_width += width

    if current_controls:
        rows.append(_row(section_index, len(rows) + 1, current_controls))

    return rows


def _row(section_index: int, row_index: int, controls: list[LayoutControl]) -> Row:
    return Row(row_id=f"s{section_index:03d}_r{row_index:03d}", controls=controls)


def _section_name(controls: list[Control]) -> str:
    token_counts: dict[str, int] = defaultdict(int)
    for control in controls:
        for token in control.label.lower().split():
            cleaned = token.strip(".,:;()[]{}")
            if len(cleaned) >= 4:
                token_counts[cleaned] += 1

    if not token_counts:
        return "Form section"

    best_token = max(token_counts, key=lambda token: (token_counts[token], -len(token)))
    return f"{best_token.title()} information"


def _embedding_text(control: Control) -> str:
    return control.label
