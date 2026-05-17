"""Evaluate generated layouts."""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import (
    load_forms_jsonl,
    load_generated_layouts_jsonl,
    load_order_constraints_jsonl,
)
from layout_spatial_reasoning.embeddings.provider import embedding_function_from_env
from layout_spatial_reasoning.evaluation.metrics import evaluate_generated_layout


def main() -> None:
    forms_path = Path("data/processed/sample_forms.jsonl")
    order_constraints_path = Path("data/processed/order_constraints.jsonl")
    generated_path = Path("outputs/generated_layouts/sample_baselines.jsonl")
    output_path = Path("outputs/metrics/sample_metrics.csv")

    forms = {form.form_id: form for form in load_forms_jsonl(forms_path)}
    extracted_constraints = (
        load_order_constraints_jsonl(order_constraints_path)
        if order_constraints_path.exists()
        else {}
    )
    generated_layouts = load_generated_layouts_jsonl(generated_path)
    embedding_function = embedding_function_from_env()

    records = []
    for generated in generated_layouts:
        form = forms.get(generated.form_id)
        if form is None:
            raise ValueError(f"Generated layout references unknown form {generated.form_id!r}.")
        if generated.form_id in extracted_constraints:
            form = form.model_copy(
                update={"order_constraints": extracted_constraints[generated.form_id]}
            )
        records.append(
            evaluate_generated_layout(
                form,
                generated,
                embedding_function=embedding_function,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].model_dump().keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())

    print(f"Wrote {len(records)} evaluation rows to {output_path}.")


if __name__ == "__main__":
    main()
