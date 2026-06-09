from layout_spatial_reasoning.llm import order_extractor
from layout_spatial_reasoning.llm.order_extractor import (
    extract_order_constraints_llm,
    validate_order_constraints,
)
from layout_spatial_reasoning.schemas import Control, OrderConstraint


def test_validate_order_constraints_removes_invalid_pairs():
    controls = [
        Control(id="c01", label="Password", type="text"),
        Control(id="c02", label="Confirm password", type="text"),
    ]
    constraints = [
        OrderConstraint(before="c01", after="c02"),
        OrderConstraint(before="c01", after="c02"),
        OrderConstraint(before="c02", after="c01"),
        OrderConstraint(before="c01", after="c01"),
        OrderConstraint(before="missing", after="c02"),
    ]

    assert validate_order_constraints(constraints, controls) == [
        OrderConstraint(before="c01", after="c02")
    ]


def test_extract_order_constraints_llm_validates_provider_response(monkeypatch):
    controls = [
        Control(id="c01", label="Issue type", type="choice"),
        Control(id="c02", label="Issue description", type="long_text"),
    ]

    def fake_generate_json(provider, *, model, messages, **_kwargs):
        assert provider == "gemini"
        assert model == "gemini-test"
        assert '"controls"' in messages[-1]["content"]
        return '{"constraints":[{"before":"c01","after":"c02"},{"before":"missing","after":"c02"}]}'

    monkeypatch.setattr(order_extractor, "generate_json", fake_generate_json)

    assert extract_order_constraints_llm(
        controls,
        provider="gemini",
        model="gemini-test",
    ) == [OrderConstraint(before="c01", after="c02")]
