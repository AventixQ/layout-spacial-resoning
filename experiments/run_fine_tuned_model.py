"""Run Method 3 fine-tuned model inference on sample forms."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import load_forms_jsonl, write_layouts_jsonl
from layout_spatial_reasoning.methods.fine_tuned_model import generate_layout


def main() -> None:
    input_path = Path(os.environ.get("FORMS_PATH", "data/processed/sample_forms.jsonl"))
    output_path = Path(
        os.environ.get(
            "FINE_TUNED_OUTPUT_PATH",
            "outputs/generated_layouts/sample_fine_tuned_model.jsonl",
        )
    )
    errors_path = Path(
        os.environ.get(
            "FINE_TUNED_ERRORS_PATH",
            "outputs/generated_layouts/sample_fine_tuned_model_errors.jsonl",
        )
    )
    provider = os.environ.get("FINE_TUNED_PROVIDER", "openai")
    model = os.environ.get("FINE_TUNED_MODEL") or None
    start = int(os.environ.get("FINE_TUNED_START", "0"))
    limit = int(os.environ.get("FINE_TUNED_LIMIT", "0"))

    forms = load_forms_jsonl(input_path)
    forms = forms[start : start + limit] if limit else forms[start:]
    records = []
    errors = []
    method = f"{provider}_fine_tuned_model"
    for index, form in enumerate(forms, start=1):
        print(
            f"[fine-tuned] {index}/{len(forms)} {form.form_id} "
            f"controls={len(form.controls)}",
            flush=True,
        )
        try:
            layout = generate_layout(form.controls, provider=provider, model=model)
        except Exception as error:  # noqa: BLE001 - experiment runner records failures.
            errors.append(
                {
                    "form_id": form.form_id,
                    "method": method,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        records.append((form.form_id, method, layout))

    write_layouts_jsonl(records, output_path)
    _write_errors(errors, errors_path)
    print(f"Wrote {len(records)} Method 3 layouts to {output_path}.")
    print(f"Wrote {len(errors)} Method 3 errors to {errors_path}.")


def _write_errors(errors: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=True))
            file.write("\n")


if __name__ == "__main__":
    main()
