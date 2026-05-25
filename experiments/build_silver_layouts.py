"""Build multi-model reviewed silver reference layouts for fine-tuning."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.training.silver_layouts import (
    DEFAULT_SILVER_MODELS,
    build_silver_dataset_file,
    providers_from_env,
)


def main() -> None:
    load_env()
    input_path = Path(os.environ.get("SILVER_LAYOUT_INPUT", "data/splits/train.jsonl"))
    output_path = Path(
        os.environ.get(
            "SILVER_LAYOUT_OUTPUT",
            "outputs/fine_tuning/silver_train_forms.jsonl",
        )
    )
    audit_path = Path(
        os.environ.get(
            "SILVER_LAYOUT_AUDIT_OUTPUT",
            "outputs/fine_tuning/silver_train_audit.jsonl",
        )
    )
    limit = _optional_int(os.environ.get("SILVER_LAYOUT_LIMIT"))
    start = int(os.environ.get("SILVER_LAYOUT_START", "0"))

    providers = providers_from_env()
    print("Silver layout providers:")
    for provider in providers:
        default_marker = " default" if (
            DEFAULT_SILVER_MODELS.get(provider.provider) == provider.model
        ) else ""
        print(f"  {provider.provider}: {provider.model}{default_marker}")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    if limit is not None:
        print(f"Range : start={start}, limit={limit}")
    print(
        "Workers: "
        f"providers={os.environ.get('SILVER_PROVIDER_WORKERS') or 'auto'}, "
        f"forms={os.environ.get('SILVER_FORM_WORKERS') or '1'}"
    )
    print(
        "Retries: "
        f"provider_retries={os.environ.get('SILVER_PROVIDER_RETRIES') or '2'}, "
        f"delay={os.environ.get('SILVER_RETRY_DELAY_SECONDS') or '2.0'}s"
    )
    if limit == 0:
        print("Preflight only: no forms requested, outputs were not modified.")
        return

    manifest = build_silver_dataset_file(
        input_path,
        output_path,
        limit=limit,
        start=start,
        audit_path=audit_path,
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {manifest['forms_written']} forms to {output_path}.")
    print(f"Wrote audit records to {audit_path}.")
    print(f"Failed forms: {manifest['forms_failed']}")
    print(f"Wrote manifest to {manifest_path}.")


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    main()
