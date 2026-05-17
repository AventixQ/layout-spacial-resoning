from pathlib import Path

import pytest

from layout_spatial_reasoning.methods.fine_tuned_model import (
    build_fine_tuning_records,
    filter_layout_controls,
    fine_tuning_record,
    parse_fine_tuned_response,
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
