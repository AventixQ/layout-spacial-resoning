"""Evaluate generated layouts."""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import load_forms_jsonl, load_generated_layouts_jsonl
from layout_spatial_reasoning.evaluation.metrics import evaluate_generated_layout


def main() -> None:
    forms_path = Path("data/processed/sample_forms.jsonl")
    generated_path = Path("outputs/generated_layouts/sample_baselines.jsonl")
    output_path = Path("outputs/metrics/sample_metrics.csv")

    forms = {form.form_id: form for form in load_forms_jsonl(forms_path)}
    generated_layouts = load_generated_layouts_jsonl(generated_path)

    records = []
    for generated in generated_layouts:
        form = forms.get(generated.form_id)
        if form is None:
            raise ValueError(f"Generated layout references unknown form {generated.form_id!r}.")
        records.append(evaluate_generated_layout(form, generated))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].model_dump().keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())

    print(f"Wrote {len(records)} evaluation rows to {output_path}.")


if __name__ == "__main__":
    main()
