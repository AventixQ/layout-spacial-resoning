"""Check whether a Hugging Face model fits the local LoRA text pipeline."""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.training.lora import (
    format_chat_messages,
    load_chat_jsonl,
)


def main() -> None:
    load_env()
    model_name = os.environ.get("LORA_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    data_path = Path(
        os.environ.get("LORA_COMPAT_DATA", "outputs/fine_tuning/method3_train.jsonl")
    )
    sample_count = int(os.environ.get("LORA_COMPAT_SAMPLE_COUNT", "8"))

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install LoRA dependencies with `uv sync --extra lora`.") from error

    print(f"Model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    architectures = getattr(config, "architectures", None) or []
    model_type = getattr(config, "model_type", "")
    text_config = getattr(config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", "") if text_config is not None else ""
    context_length = _context_length(config, text_config)

    print(f"model_type: {model_type}")
    print(f"text_model_type: {text_model_type or '-'}")
    print(f"architectures: {architectures}")
    print(f"context_length_hint: {context_length or '-'}")
    print(f"chat_template: {'yes' if getattr(tokenizer, 'chat_template', None) else 'no'}")
    print(f"pad_token: {tokenizer.pad_token or '-'}")
    print(f"eos_token: {tokenizer.eos_token or '-'}")

    if not _looks_causal_lm_compatible(model_type, text_model_type, architectures):
        print("WARNING: model does not look like a plain AutoModelForCausalLM target.")
        print("         It may need a different loader than the current LoRA trainer.")

    if data_path.exists():
        lengths = []
        for record in load_chat_jsonl(data_path)[:sample_count]:
            text = format_chat_messages(record["messages"], tokenizer)
            lengths.append(
                len(tokenizer(text, add_special_tokens=False)["input_ids"])
            )
        if lengths:
            print(f"sample_records: {len(lengths)}")
            print(f"sample_min_tokens: {min(lengths)}")
            print(f"sample_max_tokens: {max(lengths)}")
    else:
        print(f"Data file not found, skipped tokenization sample: {data_path}")


def _context_length(config, text_config) -> int | None:
    candidates = [config]
    if text_config is not None:
        candidates.append(text_config)
    for item in candidates:
        for name in (
            "max_position_embeddings",
            "seq_length",
            "model_max_length",
            "max_seq_len",
        ):
            value = getattr(item, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _looks_causal_lm_compatible(
    model_type: str,
    text_model_type: str,
    architectures: list[str],
) -> bool:
    if any("CausalLM" in architecture for architecture in architectures):
        return True
    if text_model_type and text_model_type.startswith("qwen"):
        return True
    return model_type.startswith("qwen") and "vl" not in model_type.lower()


if __name__ == "__main__":
    main()
