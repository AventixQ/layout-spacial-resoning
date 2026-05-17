"""Row underutilization detector."""

from collections.abc import Callable
from dataclasses import dataclass

from layout_spatial_reasoning.embeddings.provider import embed_texts
from layout_spatial_reasoning.embeddings.similarity import cosine_similarity
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, Row, Section

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class RowUnderutilizationCase:
    """Two related controls placed in adjacent underfilled rows."""

    section_id: str
    first_row_id: str
    second_row_id: str
    first_control_id: str
    second_control_id: str
    similarity: float


def detect_row_underutilization(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    full_width_threshold: int = 10,
) -> list[RowUnderutilizationCase]:
    """Detect adjacent full-width rows that likely should be merged."""
    embeddings = _control_embeddings(controls, embedding_function)
    cases: list[RowUnderutilizationCase] = []

    for section in layout.sections:
        full_rows = _full_single_control_rows(section, full_width_threshold)
        full_row_by_control = {row.controls[0].id: row for row in full_rows}
        full_control_ids = list(full_row_by_control)

        for first_row, second_row in zip(section.rows, section.rows[1:], strict=False):
            if first_row not in full_rows or second_row not in full_rows:
                continue

            first_id = first_row.controls[0].id
            second_id = second_row.controls[0].id
            if first_id not in embeddings or second_id not in embeddings:
                continue
            if not _are_mutual_nearest_neighbors(
                first_id,
                second_id,
                full_control_ids,
                embeddings,
            ):
                continue

            cases.append(
                RowUnderutilizationCase(
                    section_id=section.section_id,
                    first_row_id=first_row.row_id,
                    second_row_id=second_row.row_id,
                    first_control_id=first_id,
                    second_control_id=second_id,
                    similarity=cosine_similarity(embeddings[first_id], embeddings[second_id]),
                )
            )

    return cases


def row_underutilization_count(
    controls: list[Control],
    layout: Layout,
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    full_width_threshold: int = 10,
) -> int:
    """Return the number of detected row underutilization cases."""
    return len(
        detect_row_underutilization(
            controls,
            layout,
            embedding_function=embedding_function,
            full_width_threshold=full_width_threshold,
        )
    )


def _full_single_control_rows(section: Section, full_width_threshold: int) -> list[Row]:
    return [
        row
        for row in section.rows
        if len(row.controls) == 1 and row.controls[0].colSpan >= full_width_threshold
    ]


def _are_mutual_nearest_neighbors(
    first_id: str,
    second_id: str,
    candidate_ids: list[str],
    embeddings: dict[str, list[float]],
) -> bool:
    first_neighbor = _nearest_neighbor(first_id, candidate_ids, embeddings)
    second_neighbor = _nearest_neighbor(second_id, candidate_ids, embeddings)
    return first_neighbor == second_id and second_neighbor == first_id


def _nearest_neighbor(
    control_id: str,
    candidate_ids: list[str],
    embeddings: dict[str, list[float]],
) -> str | None:
    candidates = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id != control_id and candidate_id in embeddings
    ]
    if not candidates or control_id not in embeddings:
        return None

    return max(
        candidates,
        key=lambda candidate_id: (
            cosine_similarity(embeddings[control_id], embeddings[candidate_id]),
            candidate_id,
        ),
    )


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
