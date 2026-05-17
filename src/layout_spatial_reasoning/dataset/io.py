"""JSONL helpers for form specifications and generated layouts."""

import json
from pathlib import Path
from typing import Iterable

from layout_spatial_reasoning.schemas.form import FormSpec
from layout_spatial_reasoning.schemas.layout import Layout
from layout_spatial_reasoning.schemas.results import GeneratedLayoutRecord


def load_forms_jsonl(path: str | Path) -> list[FormSpec]:
    """Load form specifications from a JSONL file."""
    forms: list[FormSpec] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                forms.append(FormSpec.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid form JSON on line {line_number}: {error}") from error
    return forms


def load_generated_layouts_jsonl(path: str | Path) -> list[GeneratedLayoutRecord]:
    """Load generated layout records from a JSONL file."""
    records: list[GeneratedLayoutRecord] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(GeneratedLayoutRecord.model_validate_json(line))
            except ValueError as error:
                message = f"Invalid generated layout JSON on line {line_number}: {error}"
                raise ValueError(message) from error
    return records


def write_layouts_jsonl(
    records: Iterable[tuple[str, str, Layout]],
    path: str | Path,
) -> None:
    """Write generated layouts as JSONL records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for form_id, method, layout in records:
            file.write(
                json.dumps(
                    {
                        "form_id": form_id,
                        "method": method,
                        "layout": layout.model_dump(mode="json"),
                    },
                    ensure_ascii=True,
                )
            )
            file.write("\n")
