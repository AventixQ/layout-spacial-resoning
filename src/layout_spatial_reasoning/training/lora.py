"""LoRA supervised fine-tuning for Method 3."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

from layout_spatial_reasoning.config import env_int, env_str


IGNORE_INDEX = -100
DEFAULT_LORA_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class LoraTrainingConfig:
    """Configuration for local LoRA training."""

    base_model: str
    output_dir: str
    max_length: int = 2048
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 100
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True


def build_lora_config_from_env() -> LoraTrainingConfig:
    """Build LoRA training config from environment variables."""
    target_modules = env_str(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return LoraTrainingConfig(
        base_model=env_str("LORA_BASE_MODEL", DEFAULT_LORA_BASE_MODEL)
        or DEFAULT_LORA_BASE_MODEL,
        output_dir=env_str("LORA_OUTPUT_DIR", "outputs/lora/method3") or "outputs/lora/method3",
        max_length=env_int("LORA_MAX_LENGTH", 2048),
        epochs=env_int("LORA_EPOCHS", 3),
        batch_size=env_int("LORA_BATCH_SIZE", 1),
        gradient_accumulation_steps=env_int("LORA_GRADIENT_ACCUMULATION_STEPS", 8),
        lora_rank=env_int("LORA_RANK", 16),
        lora_alpha=env_int("LORA_ALPHA", 32),
        target_modules=tuple(
            module.strip()
            for module in (target_modules or "").split(",")
            if module.strip()
        ),
        load_in_4bit=(env_str("LORA_LOAD_IN_4BIT", "false") or "false").lower()
        == "true",
    )


def load_chat_jsonl(path: str | Path) -> list[dict[str, list[dict[str, str]]]]:
    """Load chat fine-tuning JSONL records."""
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            _validate_chat_record(record, line_number)
            records.append(record)
    return records


def format_chat_messages(
    messages: list[dict[str, str]],
    tokenizer: Any | None = None,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Format chat messages for supervised fine-tuning."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    rendered = []
    for message in messages:
        rendered.append(f"<|{message['role']}|>\n{message['content']}\n")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "".join(rendered)


def tokenize_supervised_record(
    record: dict[str, list[dict[str, str]]],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one chat record and mask prompt tokens in labels."""
    messages = record["messages"]
    prompt_text = format_chat_messages(
        messages[:-1],
        tokenizer,
        add_generation_prompt=True,
    )
    full_text = format_chat_messages(messages, tokenizer, add_generation_prompt=False)

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized.get("attention_mask", [1] * len(input_ids))
    prompt_length = min(len(prompt_tokens), len(input_ids))
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]

    if labels and all(label == IGNORE_INDEX for label in labels):
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_lora(
    train_path: str | Path,
    *,
    validation_path: str | Path | None = None,
    config: LoraTrainingConfig | None = None,
) -> str:
    """Train a LoRA adapter and return the output directory."""
    cfg = config or build_lora_config_from_env()
    _ensure_lora_dependencies()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.target_modules),
        ),
    )

    train_dataset = _TokenizedChatDataset(
        load_chat_jsonl(train_path),
        tokenizer,
        max_length=cfg.max_length,
    )
    eval_dataset = None
    if validation_path is not None:
        eval_dataset = _TokenizedChatDataset(
            load_chat_jsonl(validation_path),
            tokenizer,
            max_length=cfg.max_length,
        )

    training_args = TrainingArguments(**_training_arguments_kwargs(cfg, eval_dataset))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_SupervisedDataCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir


class _TokenizedChatDataset:
    def __init__(self, records, tokenizer, *, max_length: int):
        self._items = [
            tokenize_supervised_record(record, tokenizer, max_length=max_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


class _SupervisedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {key: torch.tensor(value) for key, value in batch.items()}


def _validate_chat_record(record: dict[str, Any], line_number: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"Invalid chat record on line {line_number}: messages missing.")
    if messages[-1].get("role") != "assistant":
        raise ValueError(
            f"Invalid chat record on line {line_number}: last message must be assistant."
        )
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"Invalid chat record on line {line_number}: bad message keys."
            )


def _training_arguments_kwargs(cfg: LoraTrainingConfig, eval_dataset) -> dict[str, Any]:
    import torch
    from transformers import TrainingArguments

    eval_key = "eval_strategy"
    if "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
        eval_key = "evaluation_strategy"

    kwargs = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": False,
        eval_key: "steps" if eval_dataset is not None else "no",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    return kwargs


def _ensure_lora_dependencies() -> None:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "LoRA training dependencies are missing. Install them with "
            "`uv sync --extra lora`, or add `--extra lora-quantized` for 4-bit training."
        ) from error
