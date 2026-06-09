"""Provider-neutral JSON generation helpers for LLM-backed methods."""

import json
import os
from typing import Literal
from urllib import error, request

from openai import OpenAI

from layout_spatial_reasoning.config import load_env


LLMProvider = Literal["openai", "gemini", "claude", "local_hf"]

_LOCAL_HF_CACHE = {}

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "local_hf": "",
}

PROVIDER_MODEL_ENV_KEYS = {
    "openai": "OPENAI_LAYOUT_MODEL",
    "gemini": "GEMINI_LAYOUT_MODEL",
    "claude": "CLAUDE_LAYOUT_MODEL",
    "local_hf": "LOCAL_HF_LAYOUT_MODEL",
}

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "claude": "claude-sonnet-4-5",
    "local_hf": "Qwen/Qwen3-4B-Instruct-2507",
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
            max_tokens=max_tokens,
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
    if provider_name == "local_hf":
        return _local_hf_generate_json(
            model=model_name,
            messages=messages,
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
    max_tokens: int,
) -> str:
    client_kwargs = {"api_key": _api_key("openai")}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    request_kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    request_kwargs.update(_openai_token_limit_kwargs(model, max_tokens))
    response_mode = os.environ.get("OPENAI_RESPONSE_FORMAT_MODE", "auto").lower()
    extra_body = _openai_extra_body_for_response_format(response_format, response_mode)
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    elif response_mode != "disabled":
        request_kwargs["response_format"] = response_format or {"type": "json_object"}
    response = client.chat.completions.create(
        **request_kwargs,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned an empty response.")
    return content


def _openai_token_limit_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    token_parameter = os.environ.get("OPENAI_TOKEN_PARAMETER", "auto").lower()
    if token_parameter in {"max_completion_tokens", "completion"}:
        return {"max_completion_tokens": max_tokens}
    if token_parameter in {"max_tokens", "tokens"}:
        return {"max_tokens": max_tokens}

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if base_url and "api.openai.com" not in base_url:
        return {"max_tokens": max_tokens}
    if _openai_model_prefers_max_completion_tokens(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _openai_model_prefers_max_completion_tokens(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _openai_extra_body_for_response_format(
    response_format: dict[str, object] | None,
    response_mode: str,
) -> dict[str, object] | None:
    if not response_format:
        return None
    if response_format.get("type") != "json_schema":
        return None
    json_schema = response_format.get("json_schema", {})
    if not isinstance(json_schema, dict) or "schema" not in json_schema:
        return None
    schema = json_schema["schema"]
    if response_mode == "vllm_guided_json":
        return {"guided_json": schema}
    if response_mode == "vllm_structured_outputs":
        return {"structured_outputs": {"json": schema}}
    return None


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


def _local_hf_generate_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Local Hugging Face inference dependencies are missing. Install them with "
            "`uv sync --extra lora --extra lora-quantized`."
        ) from error

    max_new_tokens = int(os.environ.get("LOCAL_HF_MAX_NEW_TOKENS", str(max_tokens)))
    load_in_4bit = os.environ.get("LOCAL_HF_LOAD_IN_4BIT", "true").lower() == "true"
    cache_key = (model, load_in_4bit)
    if cache_key in _LOCAL_HF_CACHE:
        tokenizer, causal_model = _LOCAL_HF_CACHE[cache_key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        causal_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        causal_model.eval()
        _LOCAL_HF_CACHE[cache_key] = (tokenizer, causal_model)

    prompt = _local_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = next(causal_model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = causal_model.generate(**generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _local_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = []
    for message in messages:
        role = message["role"].upper()
        rendered.append(f"{role}:\n{message['content']}")
    rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


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
