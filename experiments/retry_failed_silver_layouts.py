"""Retry incomplete silver-layout records and merge successful targets."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.dataset.io import load_forms_jsonl
from layout_spatial_reasoning.schemas.form import FormSpec
from layout_spatial_reasoning.training.silver_layouts import (
    build_silver_reference_layouts,
)


TARGET_LAYOUT_COUNT = 2


def main() -> None:
    load_env()
    source_path = Path(os.environ.get("SILVER_LAYOUT_INPUT", "data/splits/train.jsonl"))
    current_path = Path(
        os.environ.get(
            "SILVER_LAYOUT_OUTPUT",
            "outputs/fine_tuning/silver_train_forms.jsonl",
        )
    )
    output_path = Path(
        os.environ.get(
            "SILVER_RETRY_OUTPUT",
            "outputs/fine_tuning/silver_train_forms_recovered.jsonl",
        )
    )
    audit_path = Path(
        os.environ.get(
            "SILVER_RETRY_AUDIT_OUTPUT",
            "outputs/fine_tuning/silver_train_retry_audit.jsonl",
        )
    )
    max_forms = _optional_int(os.environ.get("SILVER_RETRY_LIMIT"))

    source_by_id = {form.form_id: form for form in load_forms_jsonl(source_path)}
    current_forms = load_forms_jsonl(current_path)
    incomplete = [
        form
        for form in current_forms
        if len(form.target_layouts) < TARGET_LAYOUT_COUNT
    ]
    if max_forms is not None:
        incomplete = incomplete[:max_forms]

    print(
        f"Retrying {len(incomplete)} incomplete forms from {current_path} "
        f"into {output_path}.",
        flush=True,
    )
    recovered_by_id: dict[str, FormSpec] = {}
    audit_records = []
    for index, existing in enumerate(incomplete, start=1):
        source = source_by_id[existing.form_id]
        print(
            f"[silver-retry] {index}/{len(incomplete)} {existing.form_id} "
            f"existing={len(existing.target_layouts)}",
            flush=True,
        )
        result = build_silver_reference_layouts(source)
        merged_layouts = [
            *existing.target_layouts,
            *result.target_layouts,
        ][:TARGET_LAYOUT_COUNT]
        recovered_by_id[existing.form_id] = source.model_copy(
            update={"target_layouts": merged_layouts}
        )
        audit_records.append(
            {
                "form_id": existing.form_id,
                "existing_target_layout_count": len(existing.target_layouts),
                "retry_target_layout_count": len(result.target_layouts),
                "merged_target_layout_count": len(merged_layouts),
                "errors": result.errors,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "author_provider": candidate.author_provider,
                        "author_model": candidate.author_model,
                        "final_selected": candidate.final_selected,
                        "consensus_score": candidate.consensus_score,
                    }
                    for candidate in result.candidates
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for form in current_forms:
            file.write(recovered_by_id.get(form.form_id, form).model_dump_json())
            file.write("\n")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as file:
        for record in audit_records:
            file.write(json.dumps(record, ensure_ascii=True))
            file.write("\n")

    final_failed = sum(
        1
        for form in load_forms_jsonl(output_path)
        if len(form.target_layouts) < TARGET_LAYOUT_COUNT
    )
    manifest = {
        "input_path": str(current_path),
        "output_path": str(output_path),
        "forms_retried": len(incomplete),
        "forms_written": len(current_forms),
        "forms_failed": final_failed,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote recovered forms to {output_path}.", flush=True)
    print(f"Wrote retry audit to {audit_path}.", flush=True)
    print(f"Remaining failed forms: {final_failed}", flush=True)
    print(f"Wrote manifest to {manifest_path}.", flush=True)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    main()
