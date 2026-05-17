"""Run Method 2 multi-agent LLM pipeline on sample forms."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import load_forms_jsonl, write_layouts_jsonl
from layout_spatial_reasoning.methods.llm_multi_agent import generate_layout


def main() -> None:
    input_path = Path("data/processed/sample_forms.jsonl")
    output_path = Path("outputs/generated_layouts/sample_llm_multi_agent.jsonl")
    errors_path = Path("outputs/generated_layouts/sample_llm_multi_agent_errors.jsonl")

    forms = load_forms_jsonl(input_path)
    records = []
    errors = []
    for form in forms:
        method = "llm_multi_agent"
        try:
            layout = generate_layout(form.controls)
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
    print(f"Wrote {len(records)} Method 2 layouts to {output_path}.")
    print(f"Wrote {len(errors)} Method 2 errors to {errors_path}.")


def _write_errors(errors: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=True))
            file.write("\n")


if __name__ == "__main__":
    main()
