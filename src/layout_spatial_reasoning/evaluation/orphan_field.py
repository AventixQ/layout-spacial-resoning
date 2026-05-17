"""Orphan field detector."""

from collections.abc import Callable
from dataclasses import dataclass

from layout_spatial_reasoning.embeddings.provider import embed_texts
from layout_spatial_reasoning.embeddings.similarity import cosine_similarity
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, Section

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class OrphanFieldCase:
    """A control whose closest semantic section is not its current section."""

    control_id: str
    current_section_id: str
    best_section_id: str
    current_affinity: float
    best_affinity: float
    margin: float


def detect_orphan_fields(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    margin_threshold: float = 0.05,
) -> list[OrphanFieldCase]:
    """Detect controls that fit another section better than their current one."""
    embeddings = _control_embeddings(controls, embedding_function)
    section_control_ids = _section_control_ids(layout)
    control_to_section = {
        control_id: section_id
        for section_id, control_ids in section_control_ids.items()
        for control_id in control_ids
    }

    cases: list[OrphanFieldCase] = []
    for control_id, current_section_id in control_to_section.items():
        if control_id not in embeddings:
            continue

        current_affinity = _section_affinity(
            control_id,
            current_section_id,
            section_control_ids,
            embeddings,
        )
        alternatives = [
            (
                section_id,
                _section_affinity(control_id, section_id, section_control_ids, embeddings),
            )
            for section_id in section_control_ids
            if section_id != current_section_id
        ]
        if not alternatives:
            continue

        best_section_id, best_affinity = max(
            alternatives,
            key=lambda item: (item[1], item[0]),
        )
        margin = best_affinity - current_affinity
        if margin > margin_threshold:
            cases.append(
                OrphanFieldCase(
                    control_id=control_id,
                    current_section_id=current_section_id,
                    best_section_id=best_section_id,
                    current_affinity=current_affinity,
                    best_affinity=best_affinity,
                    margin=margin,
                )
            )

    return cases


def orphan_field_count(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    margin_threshold: float = 0.05,
) -> int:
    """Return the number of detected orphan fields."""
    return len(
        detect_orphan_fields(
            controls,
            layout,
            embedding_function=embedding_function,
            margin_threshold=margin_threshold,
        )
    )


def _section_control_ids(layout: Layout) -> dict[str, list[str]]:
    return {
        section.section_id: _control_ids(section)
        for section in layout.sections
    }


def _control_ids(section: Section) -> list[str]:
    return [
        control.id
        for row in section.rows
        for control in row.controls
    ]


def _section_affinity(
    control_id: str,
    section_id: str,
    section_control_ids: dict[str, list[str]],
    embeddings: dict[str, list[float]],
) -> float:
    neighbor_ids = [
        candidate_id
        for candidate_id in section_control_ids[section_id]
        if candidate_id != control_id and candidate_id in embeddings
    ]
    if not neighbor_ids:
        return 0.0

    return sum(
        cosine_similarity(embeddings[control_id], embeddings[neighbor_id])
        for neighbor_id in neighbor_ids
    ) / len(neighbor_ids)


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
