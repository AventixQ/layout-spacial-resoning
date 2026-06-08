import json

from layout_spatial_reasoning.methods.hybrid import (
    PreliminaryDivision,
    PreliminarySection,
    build_preliminary_division,
    generate_layout,
    hybrid_payload_to_json,
    parse_hybrid_response,
)
from layout_spatial_reasoning.schemas import Control


def test_build_preliminary_division_uses_graph_communities():
    controls = [
        Control(id="c01", label="First name", type="text"),
        Control(id="c02", label="Last name", type="text"),
        Control(id="c03", label="Invoice required", type="boolean"),
    ]

    division = build_preliminary_division(
        controls,
        embedding_function=lambda _labels: [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        similarity_threshold=0.8,
    )

    assert len(division.sections) == 2
    assert division.sections[0].control_ids == ["c01", "c02"]
    assert division.sections[1].control_ids == ["c03"]
    assert division.sections[0].section_name.endswith("information")


def test_hybrid_payload_to_json_contains_controls_and_preliminary_sections():
    controls = [Control(id="c01", label="Email", type="text")]
    division = PreliminaryDivision(
        sections=[
            PreliminarySection(
                section_id="s001",
                section_name="Email information",
                control_ids=["c01"],
            )
        ]
    )

    payload = json.loads(hybrid_payload_to_json(controls, division))

    assert payload["controls"][0]["id"] == "c01"
    assert payload["preliminary_sections"][0]["control_ids"] == ["c01"]


def test_parse_hybrid_response_accepts_json_fence():
    layout = parse_hybrid_response(
        """```json
        {"sections":[{"section_id":"s001","section_name":"Contact","rows":[]}]}
        ```"""
    )

    assert layout.sections[0].section_name == "Contact"


def test_generate_layout_forwards_provider_to_refinement(monkeypatch):
    controls = [Control(id="c01", label="Email", type="text")]
    seen = {}

    def fake_generate_json(provider, *, model, messages, **_kwargs):
        seen["provider"] = provider
        seen["model"] = model
        seen["payload"] = json.loads(messages[-1]["content"])
        return json.dumps(
            {
                "sections": [
                    {
                        "section_id": "s001",
                        "section_name": "Contact",
                        "rows": [
                            {
                                "row_id": "s001_r001",
                                "controls": [
                                    {"id": "c01", "colStart": 1, "colSpan": 12}
                                ],
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "layout_spatial_reasoning.methods.hybrid.generate_json",
        fake_generate_json,
    )

    layout = generate_layout(
        controls,
        embedding_function=lambda _labels: [[1.0, 0.0]],
        provider="gemini",
        model="gemini-test",
    )

    assert seen["provider"] == "gemini"
    assert seen["model"] == "gemini-test"
    assert seen["payload"]["controls"][0]["id"] == "c01"
    assert layout.sections[0].rows[0].controls[0].id == "c01"
