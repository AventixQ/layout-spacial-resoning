"""Train Method 3 with a local LoRA adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.training.lora import (
    build_lora_config_from_env,
    train_lora,
)


def main() -> None:
    train_path = Path("outputs/fine_tuning/method3_train.jsonl")
    validation_path = Path("outputs/fine_tuning/method3_validation.jsonl")
    config = build_lora_config_from_env()

    output_dir = train_lora(
        train_path,
        validation_path=validation_path if validation_path.exists() else None,
        config=config,
    )
    print(f"Wrote Method 3 LoRA adapter to {output_dir}.")


if __name__ == "__main__":
    main()
