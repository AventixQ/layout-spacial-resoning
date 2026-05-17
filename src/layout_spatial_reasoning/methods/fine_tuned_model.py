"""Fine-tuned small language model layout generation."""

import json
import os
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.methods.llm_single import (
    _extract_json_object,
    controls_to_json,
    parse_layout_response,
)
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.form import FormSpec
from layout_spatial_reasoning.schemas.layout import Layout, Row, Section
from layout_spatial_reasoning.schemas.validation import validate_layout


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
INFERENCE_PROMPT = PROMPT_DIR / "fine_tuned_inference.txt"


def generate_layout(
    controls: list[Control],
    *,
    model: str | None = None,
) -> Layout:
    """Generate a layout with a fine-tuned small model in one invocation."""
    return generate_layout_openai(controls, model=model)


def generate_layout_openai(
    controls: list[Control],
    *,
    model: str | None = None,
) -> Layout:
    """OpenAI implementation of Method 3 inference."""
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Method 3.")

    model_name = model or _fine_tuned_model_from_env()
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": INFERENCE_PROMPT.read_text(encoding="utf-8"),
            },
            {"role": "user", "content": controls_to_json(controls)},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Method 3 returned an empty response.")

    layout = parse_layout_response(content)
    errors = validate_layout(controls, layout)
    if errors:
        raise ValueError(f"Invalid Method 3 layout: {'; '.join(errors)}")
    return layout


def build_fine_tuning_records(
    forms: Iterable[FormSpec],
    *,
    augment: bool = True,
    label_replacements: dict[str, str] | None = None,
) -> list[dict[str, list[dict[str, str]]]]:
    """Build chat fine-tuning JSONL records from forms with target layouts."""
    records = []
    for form in forms:
        for layout in form.target_layouts:
            records.append(fine_tuning_record(form.controls, layout))
            if augment:
                records.extend(
                    augmented_fine_tuning_records(
                        form.controls,
                        layout,
                        label_replacements=label_replacements or {},
                    )
                )
    return records


def write_fine_tuning_jsonl(
    records: Iterable[dict[str, list[dict[str, str]]]],
    path: str | Path,
) -> None:
    """Write chat fine-tuning records in OpenAI JSONL format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True))
            file.write("\n")


def fine_tuning_record(
    controls: list[Control],
    layout: Layout,
) -> dict[str, list[dict[str, str]]]:
    """Create one chat fine-tuning record using the thesis schemas."""
    errors = validate_layout(controls, layout)
    if errors:
        raise ValueError(f"Cannot train on invalid target layout: {'; '.join(errors)}")

    return {
        "messages": [
            {
                "role": "system",
                "content": INFERENCE_PROMPT.read_text(encoding="utf-8"),
            },
            {"role": "user", "content": controls_to_json(controls)},
            {
                "role": "assistant",
                "content": layout.model_dump_json(),
            },
        ]
    }


def augmented_fine_tuning_records(
    controls: list[Control],
    layout: Layout,
    *,
    label_replacements: dict[str, str],
) -> list[dict[str, list[dict[str, str]]]]:
    """Create deterministic synthetic variants described for Method 3."""
    records = []

    if len(controls) > 1:
        records.append(fine_tuning_record(list(reversed(controls)), layout))

    for index, control in enumerate(controls):
        filtered_controls = [
            candidate for candidate in controls if candidate.id != control.id
        ]
        filtered_layout = filter_layout_controls(layout, {control.id})
        if filtered_controls and filtered_layout.sections:
            records.append(fine_tuning_record(filtered_controls, filtered_layout))
        if index == 0:
            break

    replaced_controls = replace_control_labels(controls, label_replacements)
    if replaced_controls != controls:
        records.append(fine_tuning_record(replaced_controls, layout))

    return records


def replace_control_labels(
    controls: list[Control],
    label_replacements: dict[str, str],
) -> list[Control]:
    """Replace labels with configured semantic equivalents."""
    return [
        control.model_copy(
            update={"label": label_replacements.get(control.label, control.label)}
        )
        for control in controls
    ]


def filter_layout_controls(layout: Layout, removed_control_ids: set[str]) -> Layout:
    """Remove selected controls from a target layout for synthetic training data."""
    sections = []
    for section in layout.sections:
        rows = []
        for row in section.rows:
            row_controls = [
                control
                for control in row.controls
                if control.id not in removed_control_ids
            ]
            if row_controls:
                rows.append(Row(row_id=row.row_id, controls=row_controls))
        if rows:
            sections.append(
                Section(
                    section_id=section.section_id,
                    section_name=section.section_name,
                    rows=rows,
                )
            )
    return Layout(sections=sections)


def parse_fine_tuned_response(content: str) -> Layout:
    """Parse a fine-tuned model response into the strict output schema."""
    return Layout.model_validate_json(_extract_json_object(content))


def _fine_tuned_model_from_env() -> str:
    model_name = (
        os.environ.get("FINE_TUNED_LAYOUT_MODEL")
        or os.environ.get("OPENAI_FINE_TUNED_LAYOUT_MODEL")
    )
    if not model_name:
        raise RuntimeError(
            "FINE_TUNED_LAYOUT_MODEL is required for Method 3 inference."
        )
    return model_name
