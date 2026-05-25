"""Provider-neutral JSON generation helpers for LLM-backed methods."""

import json
import os
from typing import Literal
from urllib import error, request

from openai import OpenAI

from layout_spatial_reasoning.config import load_env


LLMProvider = Literal["openai", "gemini", "claude"]

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

PROVIDER_MODEL_ENV_KEYS = {
    "openai": "OPENAI_LAYOUT_MODEL",
    "gemini": "GEMINI_LAYOUT_MODEL",
    "claude": "CLAUDE_LAYOUT_MODEL",
}

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "claude": "claude-sonnet-4-5",
}


def generate_json(
    provider: str,
    *,
    model: str | None,
    messages: list[dict[str, str]],
    response_format: dict[str, object] | None = None,
    assistant_prefill: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> str:
    """Generate a JSON object with the selected LLM provider."""
    load_env()
    provider_name = normalize_provider(provider)
    model_name = model or model_from_env(provider_name)
    if provider_name == "openai":
        return _openai_generate_json(
            model=model_name,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
    if provider_name == "gemini":
        return _gemini_generate_json(
            model=model_name,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider_name == "claude":
        return _claude_generate_json(
            model=model_name,
            messages=messages,
            assistant_prefill=assistant_prefill,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def normalize_provider(provider: str | None) -> LLMProvider:
    provider_name = (provider or "openai").lower().strip()
    if provider_name not in PROVIDER_ENV_KEYS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return provider_name  # type: ignore[return-value]


def model_from_env(provider: str) -> str:
    provider_name = normalize_provider(provider)
    key = PROVIDER_MODEL_ENV_KEYS[provider_name]
    return os.environ.get(key) or PROVIDER_DEFAULT_MODELS[provider_name]


def provider_from_env(default: str = "openai") -> LLMProvider:
    load_env()
    return normalize_provider(os.environ.get("DEFAULT_LLM_PROVIDER", default))


def _api_key(provider: LLMProvider) -> str:
    key = os.environ.get(PROVIDER_ENV_KEYS[provider])
    if not key:
        raise RuntimeError(f"{PROVIDER_ENV_KEYS[provider]} is required for {provider}.")
    return key


def _openai_generate_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, object] | None,
    temperature: float,
) -> str:
    client = OpenAI(api_key=_api_key("openai"))
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format=response_format or {"type": "json_object"},
        messages=messages,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned an empty response.")
    return content


def _gemini_generate_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, object] | None,
    temperature: float,
    max_tokens: int,
) -> str:
    system, contents = _gemini_messages(messages)
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    if response_format and response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema", {})
        if isinstance(json_schema, dict) and "schema" in json_schema:
            payload["generationConfig"]["responseJsonSchema"] = json_schema["schema"]
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": _api_key("gemini"),
        },
    )
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Gemini returned an unexpected response: {data}") from error


def _claude_generate_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    assistant_prefill: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    system, claude_messages = _claude_messages(messages)
    if assistant_prefill:
        claude_messages.append({"role": "assistant", "content": assistant_prefill})
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": claude_messages,
    }
    if system:
        payload["system"] = system

    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": _api_key("claude"),
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        content = "".join(
            block.get("text", "")
            for block in data["content"]
            if block.get("type") == "text"
        )
        if assistant_prefill and not content.lstrip().startswith("{"):
            return assistant_prefill + content
        return content
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Claude returned an unexpected response: {data}") from error


def _gemini_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, object]]]:
    system_parts = []
    contents = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
        )
    return "\n\n".join(system_parts), contents


def _claude_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts = []
    claude_messages = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
            continue
        claude_messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": message["content"],
            }
        )
    return "\n\n".join(system_parts), claude_messages


def _post_json(
    url: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str],
) -> dict[str, object]:
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=encoded,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider request failed with HTTP {exc.code}: {body}") from exc
