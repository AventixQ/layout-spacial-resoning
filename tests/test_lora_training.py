import json
from pathlib import Path

import pytest

from layout_spatial_reasoning.training.lora import (
    DEFAULT_LORA_BASE_MODEL,
    IGNORE_INDEX,
    build_lora_config_from_env,
    format_chat_messages,
    load_chat_jsonl,
    tokenize_supervised_record,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        del add_special_tokens
        input_ids = list(range(1, len(text.split()) + 1))
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def test_format_chat_messages_without_chat_template():
    rendered = format_chat_messages(
        [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Input"},
        ],
        add_generation_prompt=True,
    )

    assert "<|system|>" in rendered
    assert "<|assistant|>" in rendered


def test_tokenize_supervised_record_masks_prompt_tokens():
    record = {
        "messages": [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Input controls"},
            {"role": "assistant", "content": '{"sections":[]}'},
        ]
    }

    tokenized = tokenize_supervised_record(record, FakeTokenizer(), max_length=128)

    assert tokenized["labels"].count(IGNORE_INDEX) > 0
    assert tokenized["labels"][-1] != IGNORE_INDEX
    assert len(tokenized["input_ids"]) == len(tokenized["labels"])


def test_tokenize_supervised_record_rejects_truncation_by_default():
    record = {
        "messages": [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Input controls"},
            {"role": "assistant", "content": '{"sections":[]}'},
        ]
    }

    with pytest.raises(ValueError, match="exceeds LORA_MAX_LENGTH"):
        tokenize_supervised_record(record, FakeTokenizer(), max_length=2)


def test_tokenize_supervised_record_can_truncate_when_explicitly_allowed():
    record = {
        "messages": [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Input controls"},
            {"role": "assistant", "content": '{"sections":[]}'},
        ]
    }

    tokenized = tokenize_supervised_record(
        record,
        FakeTokenizer(),
        max_length=2,
        fail_on_truncation=False,
    )

    assert len(tokenized["input_ids"]) == 2


def test_load_chat_jsonl_validates_records(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Rules"},
                    {"role": "user", "content": "Input"},
                    {"role": "assistant", "content": "{}"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_chat_jsonl(path)

    assert len(records) == 1


def test_load_chat_jsonl_rejects_non_assistant_completion(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Rules"},
                    {"role": "user", "content": "Input"},
                    {"role": "user", "content": "{}"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="last message must be assistant"):
        load_chat_jsonl(path)


def test_build_lora_config_from_env(monkeypatch):
    monkeypatch.setenv("LORA_BASE_MODEL", "tiny/model")
    monkeypatch.setenv("LORA_OUTPUT_DIR", "outputs/test-lora")
    monkeypatch.setenv("LORA_TARGET_MODULES", "q_proj,v_proj")
    monkeypatch.setenv("LORA_RANK", "8")

    config = build_lora_config_from_env()

    assert config.base_model == "tiny/model"
    assert config.output_dir == "outputs/test-lora"
    assert config.target_modules == ("q_proj", "v_proj")
    assert config.lora_rank == 8
    assert config.fail_on_truncation


def test_default_lora_base_model_is_qwen3_4b(monkeypatch):
    monkeypatch.delenv("LORA_BASE_MODEL", raising=False)

    config = build_lora_config_from_env()

    assert config.base_model == DEFAULT_LORA_BASE_MODEL
    assert config.base_model == "Qwen/Qwen3-4B-Instruct-2507"
