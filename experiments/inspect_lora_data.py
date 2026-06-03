"""Inspect LoRA chat JSONL length distribution before training."""

from pathlib import Path
import os
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.training.lora import (
    build_lora_config_from_env,
    format_chat_messages,
    load_chat_jsonl,
)


def main() -> None:
    load_env()
    path = Path(os.environ.get("LORA_INSPECT_PATH", "outputs/fine_tuning/method3_train.jsonl"))
    cfg = build_lora_config_from_env()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install LoRA dependencies with `uv sync --extra lora`.") from error

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    lengths = []
    prompt_lengths = []
    for record in load_chat_jsonl(path):
        full_text = format_chat_messages(record["messages"], tokenizer)
        prompt_text = format_chat_messages(
            record["messages"][:-1],
            tokenizer,
            add_generation_prompt=True,
        )
        lengths.append(
            len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        )
        prompt_lengths.append(
            len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        )

    over_limit = sum(length > cfg.max_length for length in lengths)
    print(f"Records: {len(lengths)}")
    print(f"Base model: {cfg.base_model}")
    print(f"LORA_MAX_LENGTH: {cfg.max_length}")
    print(f"Over limit: {over_limit}")
    if lengths:
        print(f"Min tokens: {min(lengths)}")
        print(f"Median tokens: {int(statistics.median(lengths))}")
        print(f"P95 tokens: {_percentile(lengths, 0.95)}")
        print(f"P99 tokens: {_percentile(lengths, 0.99)}")
        print(f"Max tokens: {max(lengths)}")
        print(f"Max prompt tokens: {max(prompt_lengths)}")


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


if __name__ == "__main__":
    main()
