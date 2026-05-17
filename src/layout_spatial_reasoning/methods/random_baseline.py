"""Random baseline constrained by the twelve-column grid."""

import random

from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, LayoutControl, Row, Section


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
    seed: int | None = None,
    max_sections: int = 4,
) -> Layout:
    """Generate a random but structurally valid layout."""
    rng = random.Random(seed)
    shuffled = list(controls)
    rng.shuffle(shuffled)

    if not shuffled:
        return Layout(sections=[])

    section_count = rng.randint(1, min(max_sections, len(shuffled)))
    section_controls: list[list[Control]] = [[] for _ in range(section_count)]
    for control in shuffled:
        section_controls[rng.randrange(section_count)].append(control)

    sections: list[Section] = []
    for section_index, controls_in_section in enumerate(section_controls, start=1):
        if not controls_in_section:
            continue
        rows = _pack_rows(controls_in_section, section_index)
        sections.append(
            Section(
                section_id=f"s{section_index:03d}",
                section_name=f"Random section {section_index}",
                rows=rows,
            )
        )

    return Layout(sections=sections)


def _pack_rows(controls: list[Control], section_index: int) -> list[Row]:
    rows: list[Row] = []
    current_controls: list[LayoutControl] = []
    current_width = 0

    for control in controls:
        width = DEFAULT_WIDTHS[control.type]
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
