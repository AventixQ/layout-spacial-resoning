import json

import pytest

from layout_spatial_reasoning.llm.providers import (
    _claude_messages,
    _gemini_messages,
    _post_json,
    model_from_env,
    normalize_provider,
)


def test_normalize_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        normalize_provider("other")


def test_model_from_env_uses_provider_specific_defaults(monkeypatch):
    monkeypatch.delenv("GEMINI_LAYOUT_MODEL", raising=False)

    assert model_from_env("gemini") == "gemini-2.5-pro"


def test_gemini_messages_split_system_instruction():
    system, contents = _gemini_messages(
        [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Input"},
            {"role": "assistant", "content": "Output"},
        ]
    )

    assert system == "Rules"
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"


def test_claude_messages_split_system_instruction():
    system, messages = _claude_messages(
        [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Input"},
            {"role": "assistant", "content": "Output"},
        ]
    )

    assert system == "Rules"
    assert messages == [
        {"role": "user", "content": "Input"},
        {"role": "assistant", "content": "Output"},
    ]


def test_post_json_decodes_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert request.method == "POST"
        assert timeout == 120
        return FakeResponse()

    monkeypatch.setattr("layout_spatial_reasoning.llm.providers.request.urlopen", fake_urlopen)

    assert _post_json("https://example.test", {"hello": "world"}, headers={}) == {"ok": True}
