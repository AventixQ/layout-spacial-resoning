"""Single-invocation LLM layout generation."""

import json
from pathlib import Path
from typing import Literal

from layout_spatial_reasoning.llm.providers import generate_json, provider_from_env
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout


PromptVariant = Literal[
    "zero_shot",
    "few_shot",
    "cot",
    "structured_output",
]

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = {
    "zero_shot": PROMPT_DIR / "single_zero_shot.txt",
    "few_shot": PROMPT_DIR / "single_few_shot.txt",
    "cot": PROMPT_DIR / "single_cot.txt",
    "structured_output": PROMPT_DIR / "single_structured_output.txt",
}


def generate_layout(
    controls: list[Control],
    *,
    variant: PromptVariant = "structured_output",
    examples: list[tuple[list[Control], Layout]] | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Layout:
    """Generate a complete form layout in one LLM invocation."""
    return generate_layout_llm(
        controls,
        variant=variant,
        examples=examples or [],
        model=model,
        provider=provider,
    )


def generate_layout_openai(
    controls: list[Control],
    *,
    variant: PromptVariant = "structured_output",
    examples: list[tuple[list[Control], Layout]] | None = None,
    model: str | None = None,
) -> Layout:
    """OpenAI implementation of Method 1."""
    return generate_layout_llm(
        controls,
        variant=variant,
        examples=examples,
        model=model,
        provider="openai",
    )


def generate_layout_llm(
    controls: list[Control],
    *,
    variant: PromptVariant = "structured_output",
    examples: list[tuple[list[Control], Layout]] | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Layout:
    """Provider-neutral implementation of Method 1."""
    provider_name = provider or provider_from_env()
    content = generate_json(
        provider_name,
        model=model,
        response_format=_response_format(variant),
        messages=[
            {"role": "system", "content": build_system_prompt(variant, examples or [])},
            {"role": "user", "content": controls_to_json(controls)},
        ],
    )
    return parse_layout_response(content)


def build_system_prompt(
    variant: PromptVariant,
    examples: list[tuple[list[Control], Layout]] | None = None,
) -> str:
    """Build the system prompt for the selected Method 1 variant."""
    if variant not in PROMPT_FILES:
        raise ValueError(f"Unsupported Method 1 prompt variant: {variant}")

    prompt = PROMPT_FILES[variant].read_text(encoding="utf-8")
    examples_json = _examples_to_json(examples or [])
    return prompt.replace("{{EXAMPLES_JSON}}", examples_json)


def controls_to_json(controls: list[Control]) -> str:
    """Serialize input controls using the thesis input schema."""
    return json.dumps(
        {
            "controls": [
                {
                    "id": control.id,
                    "label": control.label,
                    "type": control.type,
                    "help_text": control.help_text,
                }
                for control in controls
            ]
        },
        ensure_ascii=False,
    )


def parse_layout_response(content: str) -> Layout:
    """Parse a model response into the strict output Layout schema."""
    return Layout.model_validate_json(_extract_json_object(content))


def _response_format(variant: PromptVariant) -> dict[str, object]:
    if variant == "structured_output":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "form_layout",
                "strict": True,
                "schema": _layout_json_schema(),
            },
        }
    return {"type": "json_object"}


def _layout_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sections"],
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["section_id", "section_name", "rows"],
                    "properties": {
                        "section_id": {"type": "string"},
                        "section_name": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["row_id", "controls"],
                                "properties": {
                                    "row_id": {"type": "string"},
                                    "controls": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["id", "colStart", "colSpan"],
                                            "properties": {
                                                "id": {"type": "string"},
                                                "colStart": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "maximum": 12,
                                                },
                                                "colSpan": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "maximum": 12,
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in Method 1 response.")
    return stripped[start : end + 1]


def _examples_to_json(examples: list[tuple[list[Control], Layout]]) -> str:
    if not examples:
        return "[]"

    return json.dumps(
        [
            {
                "input": json.loads(controls_to_json(example_controls)),
                "output": layout.model_dump(mode="json"),
            }
            for example_controls, layout in examples
        ],
        ensure_ascii=False,
        indent=2,
    )
