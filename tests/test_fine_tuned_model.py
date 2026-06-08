from pathlib import Path

import pytest

from layout_spatial_reasoning.methods.fine_tuned_model import (
    build_fine_tuning_records,
    filter_layout_controls,
    fine_tuning_record,
    maybe_repair_generated_layout,
    normalize_layout_grid,
    parse_fine_tuned_response,
    repair_generated_layout,
    replace_control_labels,
    write_fine_tuning_jsonl,
)
from layout_spatial_reasoning.schemas import (
    Control,
    FormSpec,
    Layout,
    LayoutControl,
    Row,
    Section,
)


def test_parse_fine_tuned_response_accepts_json_fence():
    layout = parse_fine_tuned_response(
        """```json
        {"sections":[{"section_id":"s001","section_name":"Contact","rows":[]}]}
        ```"""
    )

    assert layout.sections[0].section_name == "Contact"


def test_fine_tuning_record_uses_chat_jsonl_shape():
    controls = [Control(id="c01", label="Email", type="text")]
    layout = _layout()

    record = fine_tuning_record(controls, layout)

    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert '"controls"' in record["messages"][1]["content"]
    assert '"sections"' in record["messages"][2]["content"]


def test_fine_tuning_record_rejects_invalid_target_layout():
    controls = [Control(id="c01", label="Email", type="text")]
    invalid_layout = Layout(sections=[])

    with pytest.raises(ValueError, match="invalid target layout"):
        fine_tuning_record(controls, invalid_layout)


def test_filter_layout_controls_removes_empty_rows_and_sections():
    layout = _layout()

    filtered = filter_layout_controls(layout, {"c01"})

    assert filtered.sections == []


def test_normalize_layout_grid_wraps_overflowing_controls():
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                            LayoutControl(id="c03", colStart=12, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    normalized = normalize_layout_grid(layout)

    assert [row.row_id for row in normalized.sections[0].rows] == [
        "s001_r001",
        "s001_r002",
    ]
    assert normalized.sections[0].rows[1].controls[0].id == "c03"
    assert normalized.sections[0].rows[1].controls[0].colStart == 1


def test_repair_generated_layout_removes_duplicates_and_appends_missing_controls():
    controls = [
        Control(id="c01", label="Email", type="text"),
        Control(id="c02", label="Message", type="long_text"),
        Control(id="c03", label="Attachment", type="file"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c01", colStart=7, colSpan=6),
                            LayoutControl(id="unknown", colStart=1, colSpan=12),
                        ],
                    ),
                    Row(
                        row_id="s001_r002",
                        controls=[LayoutControl(id="c02", colStart=1, colSpan=12)],
                    ),
                ],
            )
        ]
    )

    repaired = repair_generated_layout(controls, layout)
    actual = [
        control.id
        for section in repaired.sections
        for row in section.rows
        for control in row.controls
    ]

    assert actual == ["c01", "c02", "c03"]
    assert repaired.sections[-1].rows[-1].controls[0].colSpan == 12


def test_repair_generated_layout_does_not_reflow_grid_overflow():
    controls = [
        Control(id="c01", label="Email", type="text"),
        Control(id="c02", label="Phone", type="text"),
        Control(id="c03", label="Name", type="text"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                            LayoutControl(id="c03", colStart=12, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    repaired = repair_generated_layout(controls, layout)

    assert len(repaired.sections[0].rows) == 1
    assert repaired.sections[0].rows[0].controls[-1].colStart == 12


def test_maybe_repair_generated_layout_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FINE_TUNED_REPAIR_OUTPUT", raising=False)
    controls = [Control(id="c01", label="Email", type="text")]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c01", colStart=7, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    assert maybe_repair_generated_layout(controls, layout) == layout


def test_maybe_repair_generated_layout_can_be_enabled(monkeypatch):
    monkeypatch.setenv("FINE_TUNED_REPAIR_OUTPUT", "true")
    controls = [Control(id="c01", label="Email", type="text")]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c01", colStart=7, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )

    repaired = maybe_repair_generated_layout(controls, layout)

    assert repaired.sections[0].rows[0].controls == [
        LayoutControl(id="c01", colStart=1, colSpan=6)
    ]


def test_replace_control_labels_uses_configured_equivalents():
    controls = [Control(id="c01", label="Email", type="text")]

    replaced = replace_control_labels(controls, {"Email": "Email address"})

    assert replaced[0].label == "Email address"


def test_build_fine_tuning_records_uses_target_layouts_and_augmentations():
    controls = [
        Control(id="c01", label="Email", type="text"),
        Control(id="c02", label="Phone", type="text"),
    ]
    layout = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                        ],
                    )
                ],
            )
        ]
    )
    form = FormSpec(
        form_id="contact",
        domain="Contact",
        controls=controls,
        target_layouts=[layout],
    )

    records = build_fine_tuning_records(
        [form],
        augment=True,
        label_replacements={"Email": "Email address"},
    )

    assert len(records) == 4


def test_write_fine_tuning_jsonl(tmp_path: Path):
    path = tmp_path / "train.jsonl"

    write_fine_tuning_jsonl([fine_tuning_record(_controls(), _layout())], path)

    assert path.read_text(encoding="utf-8").count("\n") == 1


def _controls() -> list[Control]:
    return [Control(id="c01", label="Email", type="text")]


def _layout() -> Layout:
    return Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[LayoutControl(id="c01", colStart=1, colSpan=12)],
                    )
                ],
            )
        ]
    )
