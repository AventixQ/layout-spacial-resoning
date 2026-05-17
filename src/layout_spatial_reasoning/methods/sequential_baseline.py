"""Sequential baseline: one full-width row per control."""

from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, LayoutControl, Row, Section


def generate_layout(controls: list[Control]) -> Layout:
    rows = [
        Row(
            row_id=f"r{index:03d}",
            controls=[LayoutControl(id=control.id, colStart=1, colSpan=12)],
        )
        for index, control in enumerate(controls, start=1)
    ]
    return Layout(sections=[Section(section_id="s001", section_name="Form", rows=rows)])
