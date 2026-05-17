"""Section boundary misplacement detector."""

from collections.abc import Callable
from itertools import combinations

from layout_spatial_reasoning.embeddings.provider import embed_texts
from layout_spatial_reasoning.embeddings.similarity import cosine_similarity
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, Section

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def section_boundary_score(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
) -> float:
    """Measure whether section separation is weaker than internal cohesion.

    For every section with at least two known controls, the score adds
    max(0, sep(section) - coh(section)). Lower is better.
    """
    embeddings = _control_embeddings(controls, embedding_function)
    section_control_ids = [
        _known_control_ids(section, embeddings)
        for section in layout.sections
    ]
    comparable_sections = [
        control_ids for control_ids in section_control_ids if len(control_ids) >= 2
    ]
    if not comparable_sections:
        return 0.0

    penalties = []
    for current_ids in comparable_sections:
        cohesion = _average_pairwise_similarity(current_ids, embeddings)
        separation = _max_section_similarity(
            current_ids,
            section_control_ids,
            embeddings,
        )
        penalties.append(max(0.0, separation - cohesion))

    return sum(penalties) / len(penalties)


def _max_section_similarity(
    current_ids: list[str],
    all_sections: list[list[str]],
    embeddings: dict[str, list[float]],
) -> float:
    alternative_scores = [
        _average_cross_similarity(current_ids, alternative_ids, embeddings)
        for alternative_ids in all_sections
        if alternative_ids != current_ids and alternative_ids
    ]
    if not alternative_scores:
        return 0.0
    return max(alternative_scores)


def _average_pairwise_similarity(
    control_ids: list[str],
    embeddings: dict[str, list[float]],
) -> float:
    pairs = list(combinations(control_ids, 2))
    if not pairs:
        return 0.0
    return sum(
        cosine_similarity(embeddings[left], embeddings[right])
        for left, right in pairs
    ) / len(pairs)


def _average_cross_similarity(
    left_ids: list[str],
    right_ids: list[str],
    embeddings: dict[str, list[float]],
) -> float:
    pairs = [(left_id, right_id) for left_id in left_ids for right_id in right_ids]
    if not pairs:
        return 0.0
    return sum(
        cosine_similarity(embeddings[left], embeddings[right])
        for left, right in pairs
    ) / len(pairs)


def _known_control_ids(
    section: Section,
    embeddings: dict[str, list[float]],
) -> list[str]:
    return [
        control.id
        for row in section.rows
        for control in row.controls
        if control.id in embeddings
    ]


def _control_embeddings(
    controls: list[Control],
    embedding_function: EmbeddingFunction,
) -> dict[str, list[float]]:
    labels = [_embedding_text(control) for control in controls]
    vectors = embedding_function(labels)
    return {
        control.id: vector
        for control, vector in zip(controls, vectors, strict=True)
    }


def _embedding_text(control: Control) -> str:
    return control.label
