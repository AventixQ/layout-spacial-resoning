"""Build form-level train/validation JSONL splits for local LoRA training."""

from collections import defaultdict
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import load_forms_jsonl
from layout_spatial_reasoning.methods.fine_tuned_model import (
    build_fine_tuning_records,
    write_fine_tuning_jsonl,
)


def main() -> None:
    input_path = Path(
        os.environ.get(
            "LORA_FORMS_PATH",
            "outputs/fine_tuning/silver_train_forms_recovered.jsonl",
        )
    )
    train_output = Path(
        os.environ.get(
            "LORA_TRAIN_OUTPUT_PATH",
            "outputs/fine_tuning/method3_train_noaug_split.jsonl",
        )
    )
    validation_output = Path(
        os.environ.get(
            "LORA_VALIDATION_OUTPUT_PATH",
            "outputs/fine_tuning/method3_validation_noaug.jsonl",
        )
    )
    validation_ratio = float(os.environ.get("LORA_VALIDATION_RATIO", "0.1"))
    seed = int(os.environ.get("LORA_SPLIT_SEED", "42"))
    augment = os.environ.get("LORA_SPLIT_AUGMENT", "false").lower() == "true"

    forms = load_forms_jsonl(input_path)
    train_forms, validation_forms = _split_by_domain(forms, validation_ratio, seed)
    train_records = build_fine_tuning_records(train_forms, augment=augment)
    validation_records = build_fine_tuning_records(validation_forms, augment=augment)
    write_fine_tuning_jsonl(train_records, train_output)
    write_fine_tuning_jsonl(validation_records, validation_output)
    print(
        "Wrote LoRA splits: "
        f"train_forms={len(train_forms)} train_records={len(train_records)} -> {train_output}; "
        f"validation_forms={len(validation_forms)} validation_records={len(validation_records)} -> {validation_output}; "
        f"augment={augment}."
    )


def _split_by_domain(forms, validation_ratio: float, seed: int):
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for form in forms:
        by_domain[form.domain].append(form)

    train_forms = []
    validation_forms = []
    for domain in sorted(by_domain):
        domain_forms = list(by_domain[domain])
        rng.shuffle(domain_forms)
        validation_count = max(1, round(len(domain_forms) * validation_ratio))
        validation_forms.extend(domain_forms[:validation_count])
        train_forms.extend(domain_forms[validation_count:])

    train_forms.sort(key=lambda form: form.form_id)
    validation_forms.sort(key=lambda form: form.form_id)
    return train_forms, validation_forms


if __name__ == "__main__":
    main()
