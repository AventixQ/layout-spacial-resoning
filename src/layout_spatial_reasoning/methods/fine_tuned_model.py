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
from layout_spatial_reasoning.schemas.layout import Layout, LayoutControl, Row, Section
from layout_spatial_reasoning.schemas.validation import validate_layout


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
INFERENCE_PROMPT = PROMPT_DIR / "fine_tuned_inference.txt"
_LOCAL_LORA_CACHE = {}


def generate_layout(
    controls: list[Control],
    *,
    model: str | None = None,
    provider: str = "openai",
) -> Layout:
    """Generate a layout with a fine-tuned small model in one invocation."""
    if provider == "local":
        return generate_layout_local_lora(controls, model_path=model)
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

    layout = maybe_repair_generated_layout(controls, parse_layout_response(content))
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
            col_start = 1
            row_controls = []
            for control in row.controls:
                if control.id in removed_control_ids:
                    continue
                row_controls.append(
                    control.model_copy(update={"colStart": col_start})
                )
                col_start += control.colSpan
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


def normalize_layout_grid(layout: Layout, *, columns: int = 12) -> Layout:
    """Reflow generated rows so controls fit the twelve-column grid."""
    sections = []
    for section in layout.sections:
        rows = []
        row_controls = []
        col_start = 1

        def flush_row() -> None:
            nonlocal row_controls, col_start
            if row_controls:
                rows.append(
                    Row(
                        row_id=f"{section.section_id}_r{len(rows) + 1:03d}",
                        controls=row_controls,
                    )
                )
            row_controls = []
            col_start = 1

        for row in section.rows:
            for control in row.controls:
                col_span = min(control.colSpan, columns)
                if row_controls and col_start + col_span - 1 > columns:
                    flush_row()
                row_controls.append(
                    LayoutControl(
                        id=control.id,
                        colStart=col_start,
                        colSpan=col_span,
                    )
                )
                col_start += col_span
            flush_row()

        if rows:
            sections.append(
                Section(
                    section_id=section.section_id,
                    section_name=section.section_name,
                    rows=rows,
                )
            )
    return Layout(sections=sections)


def repair_generated_layout(
    controls: list[Control],
    layout: Layout,
    *,
    columns: int = 12,
) -> Layout:
    """Apply deterministic structural guardrails to generated fine-tuned layouts."""
    expected = {control.id for control in controls}
    seen: set[str] = set()
    sections = []

    for section in layout.sections:
        rows = []
        for row in section.rows:
            row_controls = []
            for control in row.controls:
                if control.id not in expected or control.id in seen:
                    continue
                seen.add(control.id)
                row_controls.append(control)
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

    missing = [control for control in controls if control.id not in seen]
    if missing:
        if sections:
            target_section = sections[-1]
            sections = sections[:-1]
        else:
            target_section = Section(
                section_id="s001",
                section_name="Additional fields",
                rows=[],
            )
        rows = list(target_section.rows)
        next_row = len(rows) + 1
        for control in missing:
            rows.append(
                Row(
                    row_id=f"{target_section.section_id}_r{next_row:03d}",
                    controls=[
                        LayoutControl(
                            id=control.id,
                            colStart=1,
                            colSpan=_default_col_span(control, columns=columns),
                        )
                    ],
                )
            )
            next_row += 1
        sections.append(
            Section(
                section_id=target_section.section_id,
                section_name=target_section.section_name,
                rows=rows,
            )
        )

    return Layout(sections=sections)


def maybe_repair_generated_layout(controls: list[Control], layout: Layout) -> Layout:
    """Optionally repair generated layouts when explicitly enabled."""
    if os.environ.get("FINE_TUNED_REPAIR_OUTPUT", "false").lower() == "true":
        return repair_generated_layout(controls, layout)
    return layout


def _default_col_span(control: Control, *, columns: int) -> int:
    if control.type in {"long_text", "file"}:
        return columns
    return columns // 2


def parse_fine_tuned_response(content: str) -> Layout:
    """Parse a fine-tuned model response into the strict output schema."""
    return Layout.model_validate_json(_extract_json_object(content))


def generate_layout_local_lora(
    controls: list[Control],
    *,
    model_path: str | None = None,
    base_model: str | None = None,
    max_new_tokens: int | None = None,
) -> Layout:
    """Generate a layout with a local Qwen LoRA adapter."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Local LoRA inference dependencies are missing. Install them with "
            "`uv sync --extra lora`."
        ) from error

    load_env()
    adapter_path = model_path or os.environ.get("LORA_OUTPUT_DIR", "outputs/lora/method3")
    base_model_name = (
        base_model
        or os.environ.get("LORA_BASE_MODEL")
        or "Qwen/Qwen3-4B-Instruct-2507"
    )
    max_new_tokens = max_new_tokens or int(os.environ.get("LORA_MAX_NEW_TOKENS", "4096"))
    load_in_4bit = os.environ.get("LORA_LOAD_IN_4BIT", "true").lower() == "true"

    cache_key = (adapter_path, base_model_name, load_in_4bit)
    if cache_key in _LOCAL_LORA_CACHE:
        tokenizer, model = _LOCAL_LORA_CACHE[cache_key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        _LOCAL_LORA_CACHE[cache_key] = (tokenizer, model)

    messages = [
        {"role": "system", "content": INFERENCE_PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": controls_to_json(controls)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True)
    layout = maybe_repair_generated_layout(controls, parse_fine_tuned_response(content))
    errors = validate_layout(controls, layout)
    if errors:
        raise ValueError(f"Invalid local Method 3 layout: {'; '.join(errors)}")
    return layout


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
