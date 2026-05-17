"""Extract reading-order constraints with the LLM OrderExtractor."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import (
    load_forms_jsonl,
    write_order_constraints_jsonl,
)
from layout_spatial_reasoning.llm.order_extractor import extract_order_constraints_openai
from layout_spatial_reasoning.schemas import OrderConstraintRecord


def main() -> None:
    forms_path = Path("data/processed/sample_forms.jsonl")
    output_path = Path("data/processed/order_constraints.jsonl")

    forms = load_forms_jsonl(forms_path)
    records = []
    for form in forms:
        constraints = extract_order_constraints_openai(form.controls)
        records.append(
            OrderConstraintRecord(
                form_id=form.form_id,
                constraints=constraints,
            )
        )

    write_order_constraints_jsonl(records, output_path)
    print(f"Wrote order constraints for {len(records)} forms to {output_path}.")


if __name__ == "__main__":
    main()
