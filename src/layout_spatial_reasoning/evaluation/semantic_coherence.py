"""Semantic coherence metrics."""

from collections.abc import Callable
from itertools import combinations

from layout_spatial_reasoning.embeddings.provider import embed_texts
from layout_spatial_reasoning.embeddings.similarity import cosine_similarity
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def within_row_coherence(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
) -> float:
    """Average label similarity for control pairs placed in the same row."""
    embeddings = _control_embeddings(controls, embedding_function)
    row_scores = [
        _average_pairwise_similarity(
            [control.id for control in row.controls],
            embeddings,
        )
        for section in layout.sections
        for row in section.rows
        if len(row.controls) >= 2
    ]
    return _mean([score for score in row_scores if score is not None])


def within_section_coherence(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
) -> float:
    """Average label similarity for control pairs placed in the same section."""
    embeddings = _control_embeddings(controls, embedding_function)
    section_scores = [
        _average_pairwise_similarity(
            [
                control.id
                for row in section.rows
                for control in row.controls
            ],
            embeddings,
        )
        for section in layout.sections
        if sum(len(row.controls) for row in section.rows) >= 2
    ]
    return _mean([score for score in section_scores if score is not None])


def semantic_coherence(
    controls: list[Control],
    layout: Layout,
    *,
    level: str = "section",
    embedding_function: EmbeddingFunction = embed_texts,
) -> float:
    """Compute semantic coherence at row or section level."""
    if level == "row":
        return within_row_coherence(
            controls,
            layout,
            embedding_function=embedding_function,
        )
    if level == "section":
        return within_section_coherence(
            controls,
            layout,
            embedding_function=embedding_function,
        )
    raise ValueError("level must be either 'row' or 'section'.")


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


def _average_pairwise_similarity(
    control_ids: list[str],
    embeddings: dict[str, list[float]],
) -> float | None:
    valid_ids = [control_id for control_id in control_ids if control_id in embeddings]
    pairs = list(combinations(valid_ids, 2))
    if not pairs:
        return None

    return _mean(
        [
            cosine_similarity(embeddings[left], embeddings[right])
            for left, right in pairs
        ]
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _embedding_text(control: Control) -> str:
    return control.label
