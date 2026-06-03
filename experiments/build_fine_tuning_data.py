"""Build Method 3 fine-tuning JSONL from forms with target layouts."""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import load_forms_jsonl
from layout_spatial_reasoning.methods.fine_tuned_model import (
    build_fine_tuning_records,
    write_fine_tuning_jsonl,
)


def main() -> None:
    input_path = Path(os.environ.get("FINE_TUNING_FORMS_PATH", "data/processed/sample_forms.jsonl"))
    output_path = Path(os.environ.get("FINE_TUNING_OUTPUT_PATH", "outputs/fine_tuning/method3_train.jsonl"))
    augment = (os.environ.get("FINE_TUNING_AUGMENT", "true").lower() == "true")

    forms = load_forms_jsonl(input_path)
    records = build_fine_tuning_records(forms, augment=augment)
    write_fine_tuning_jsonl(records, output_path)
    print(
        f"Wrote {len(records)} Method 3 fine-tuning records to {output_path} "
        f"(augment={augment})."
    )


if __name__ == "__main__":
    main()
