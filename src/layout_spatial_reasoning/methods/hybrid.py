"""Hybrid graph grouping plus LLM refinement method."""

import json
import os
from pathlib import Path
from typing import Callable

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.config import env_float, env_int, env_str, load_env
from layout_spatial_reasoning.embeddings.provider import (
    embed_texts,
    embedding_function_from_env,
)
from layout_spatial_reasoning.methods.graph_community import (
    build_similarity_graph,
    detect_communities,
)
from layout_spatial_reasoning.methods.llm_single import (
    _extract_json_object,
    controls_to_json,
)
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout
from layout_spatial_reasoning.schemas.validation import validate_layout


EmbeddingFunction = Callable[[list[str]], list[list[float]]]

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
HYBRID_PROMPT = PROMPT_DIR / "hybrid_refinement.txt"


class PreliminarySection(BaseModel):
    """Graph-derived section proposal for Method 5."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(min_length=1)
    section_name: str = Field(min_length=1)
    control_ids: list[str] = Field(min_length=1)


class PreliminaryDivision(BaseModel):
    """Graph-derived preliminary division into sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections: list[PreliminarySection] = Field(default_factory=list)


def generate_layout(
    controls: list[Control],
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    similarity_threshold: float = 0.25,
    community_algorithm: str = "leiden",
    seed: int = 42,
    model: str | None = None,
) -> Layout:
    """Generate a layout by graph grouping followed by LLM refinement."""
    preliminary_division = build_preliminary_division(
        controls,
        embedding_function=embedding_function,
        similarity_threshold=similarity_threshold,
        community_algorithm=community_algorithm,
        seed=seed,
    )
    return refine_and_arrange_openai(
        controls,
        preliminary_division,
        model=model,
    )


def generate_layout_from_env(controls: list[Control]) -> Layout:
    """Generate Method 5 layout using environment configuration."""
    return generate_layout(
        controls,
        embedding_function=embedding_function_from_env(),
        similarity_threshold=env_float("GRAPH_SIMILARITY_THRESHOLD", 0.25),
        community_algorithm=env_str("GRAPH_COMMUNITY_ALGORITHM", "leiden") or "leiden",
        seed=env_int("GRAPH_RANDOM_SEED", 42),
        model=env_str("OPENAI_LAYOUT_MODEL", "gpt-4.1"),
    )


def build_preliminary_division(
    controls: list[Control],
    *,
    embedding_function: EmbeddingFunction = embed_texts,
    similarity_threshold: float = 0.25,
    community_algorithm: str = "leiden",
    seed: int = 42,
) -> PreliminaryDivision:
    """Build graph communities without producing a grid layout."""
    if not controls:
        return PreliminaryDivision(sections=[])

    graph = build_similarity_graph(
        controls,
        embedding_function=embedding_function,
        similarity_threshold=similarity_threshold,
    )
    communities = detect_communities(graph, algorithm=community_algorithm, seed=seed)
    sections = []
    for index, community_controls in enumerate(communities, start=1):
        sections.append(
            PreliminarySection(
                section_id=f"s{index:03d}",
                section_name=_central_control_section_name(
                    graph,
                    community_controls,
                ),
                control_ids=[control.id for control in community_controls],
            )
        )
    return PreliminaryDivision(sections=sections)


def refine_and_arrange_openai(
    controls: list[Control],
    preliminary_division: PreliminaryDivision,
    *,
    model: str | None = None,
) -> Layout:
    """Run the Method 5 LLM refinement and grid arrangement step."""
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Method 5.")

    model_name = model or os.environ.get("OPENAI_LAYOUT_MODEL", "gpt-4.1")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": HYBRID_PROMPT.read_text(encoding="utf-8")},
            {
                "role": "user",
                "content": hybrid_payload_to_json(controls, preliminary_division),
            },
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Method 5 returned an empty response.")

    layout = parse_hybrid_response(content)
    errors = validate_layout(controls, layout)
    if errors:
        raise ValueError(f"Invalid Method 5 layout: {'; '.join(errors)}")
    return layout


def hybrid_payload_to_json(
    controls: list[Control],
    preliminary_division: PreliminaryDivision,
) -> str:
    """Serialize the Method 5 refinement payload."""
    return json.dumps(
        {
            "controls": json.loads(controls_to_json(controls))["controls"],
            "preliminary_sections": preliminary_division.model_dump(mode="json")[
                "sections"
            ],
        },
        ensure_ascii=False,
    )


def parse_hybrid_response(content: str) -> Layout:
    """Parse a Method 5 response into the strict layout schema."""
    return Layout.model_validate_json(_extract_json_object(content))


def _central_control_section_name(graph, controls: list[Control]) -> str:
    central_control = max(
        controls,
        key=lambda control: (
            _weighted_internal_degree(graph, control.id, {item.id for item in controls}),
            -controls.index(control),
        ),
    )
    return f"{central_control.label} information"


def _weighted_internal_degree(graph, control_id: str, community_ids: set[str]) -> float:
    total = 0.0
    for neighbor_id in graph.neighbors(control_id):
        if neighbor_id in community_ids:
            total += float(graph.edges[control_id, neighbor_id].get("weight", 1.0))
    return total
