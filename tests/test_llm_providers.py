from layout_spatial_reasoning.llm.providers import (
    _openai_extra_body_for_response_format,
    _openai_temperature_kwargs,
    _openai_token_limit_kwargs,
)
from layout_spatial_reasoning.methods.llm_single import _layout_json_schema


def test_openai_response_format_can_be_mapped_to_vllm_guided_json():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "form_layout",
            "strict": True,
            "schema": _layout_json_schema(),
        },
    }

    extra_body = _openai_extra_body_for_response_format(
        response_format,
        "vllm_guided_json",
    )

    assert extra_body == {"guided_json": response_format["json_schema"]["schema"]}


def test_openai_response_format_can_be_mapped_to_vllm_structured_outputs():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "form_layout",
            "strict": True,
            "schema": _layout_json_schema(),
        },
    }

    extra_body = _openai_extra_body_for_response_format(
        response_format,
        "vllm_structured_outputs",
    )

    assert extra_body == {
        "structured_outputs": {"json": response_format["json_schema"]["schema"]}
    }


def test_openai_gpt5_models_use_max_completion_tokens(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_TOKEN_PARAMETER", raising=False)

    assert _openai_token_limit_kwargs("gpt-5.4-mini", 123) == {
        "max_completion_tokens": 123
    }


def test_openai_compatible_base_url_keeps_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.delenv("OPENAI_TOKEN_PARAMETER", raising=False)

    assert _openai_token_limit_kwargs("layout-local-model", 123) == {"max_tokens": 123}


def test_openai_default_temperature_only_models_omit_temperature(monkeypatch):
    monkeypatch.delenv("OPENAI_TEMPERATURE_PARAMETER", raising=False)

    assert _openai_temperature_kwargs("gpt-5.5", 0) == {}
    assert _openai_temperature_kwargs("gpt-5.4-mini", 0) == {}


def test_openai_other_models_keep_temperature(monkeypatch):
    monkeypatch.delenv("OPENAI_TEMPERATURE_PARAMETER", raising=False)

    assert _openai_temperature_kwargs("gpt-4.1", 0) == {"temperature": 0}


def test_openai_temperature_parameter_can_be_forced(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE_PARAMETER", "send")

    assert _openai_temperature_kwargs("gpt-5.5", 0) == {"temperature": 0}
