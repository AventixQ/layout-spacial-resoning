"""LLM OrderExtractor for reading-order constraints."""

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.llm.providers import generate_json
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.form import OrderConstraint


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "order_extractor.txt"


class _RawOrderConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: str = Field(min_length=1)
    after: str = Field(min_length=1)


class _RawOrderConstraintResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: list[_RawOrderConstraint] = Field(default_factory=list)


def extract_order_constraints_openai(
    controls: list[Control],
    *,
    model: str | None = None,
) -> list[OrderConstraint]:
    """Extract reading-order constraints with an OpenAI chat model."""
    return extract_order_constraints_llm(controls, provider="openai", model=model)


def extract_order_constraints_llm(
    controls: list[Control],
    *,
    provider: str = "gemini",
    model: str | None = None,
) -> list[OrderConstraint]:
    """Extract reading-order constraints with a configured LLM provider."""
    load_env()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    provider_name = provider.lower().strip()
    model_name = model or _default_order_model(provider_name)
    content = generate_json(
        provider_name,
        model=model_name,
        temperature=0,
        max_tokens=2048,
        response_format=_order_constraint_response_format(),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": _controls_payload(controls)},
        ],
    )

    raw_response = _RawOrderConstraintResponse.model_validate_json(content)
    raw_constraints = [
        OrderConstraint(before=item.before, after=item.after)
        for item in raw_response.constraints
    ]
    return validate_order_constraints(raw_constraints, controls)


def _default_order_model(provider: str) -> str:
    if provider == "gemini":
        return os.environ.get("GEMINI_ORDER_MODEL", "gemini-3.1-flash-lite")
    if provider == "openai":
        return os.environ.get("OPENAI_ORDER_MODEL", "gpt-5.4-mini")
    if provider == "claude":
        return os.environ.get("CLAUDE_ORDER_MODEL", "claude-haiku-4-5")
    return ""


def _order_constraint_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "order_constraints",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "before": {"type": "string", "minLength": 1},
                                "after": {"type": "string", "minLength": 1},
                            },
                            "required": ["before", "after"],
                        },
                    }
                },
                "required": ["constraints"],
            },
        },
    }


def validate_order_constraints(
    constraints: list[OrderConstraint],
    controls: list[Control],
) -> list[OrderConstraint]:
    """Remove invalid, duplicated, reflexive, and conflicting constraints."""
    control_ids = {control.id for control in controls}
    validated: list[OrderConstraint] = []
    seen: set[tuple[str, str]] = set()

    for constraint in constraints:
        pair = (constraint.before, constraint.after)
        reverse_pair = (constraint.after, constraint.before)
        if constraint.before not in control_ids or constraint.after not in control_ids:
            continue
        if constraint.before == constraint.after:
            continue
        if pair in seen or reverse_pair in seen:
            continue
        seen.add(pair)
        validated.append(constraint)

    return validated


def _controls_payload(controls: list[Control]) -> str:
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
