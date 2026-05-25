import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_full_experiment",
    Path(__file__).resolve().parents[1] / "experiments" / "run_full_experiment.py",
)
assert _SPEC is not None
run_full_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(run_full_experiment)


def test_build_experiment_plan_includes_all_provider_llm_methods(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_LLM_PROVIDERS", "openai,gemini,claude")
    monkeypatch.setenv("OPENAI_LAYOUT_MODEL", "gpt-test")
    monkeypatch.setenv("GEMINI_LAYOUT_MODEL", "gemini-test")
    monkeypatch.setenv("CLAUDE_LAYOUT_MODEL", "claude-test")
    monkeypatch.setenv("INCLUDE_FINE_TUNED_MODEL", "true")
    monkeypatch.setenv("FINE_TUNED_LAYOUT_MODEL", "ft-model")
    monkeypatch.setenv("INCLUDE_LOCAL_LORA", "true")
    monkeypatch.setenv("LORA_OUTPUT_DIR", "outputs/lora/method3")

    methods = {plan["method"] for plan in run_full_experiment.build_experiment_plan()}

    assert "sequential" in methods
    assert "graph_community" in methods
    assert "openai_gpt_test_llm_single_zero_shot" in methods
    assert "gemini_gemini_test_llm_multi_agent" in methods
    assert "claude_claude_test_hybrid" in methods
    assert "openai_ft_model_fine_tuned_model" in methods
    assert "local_outputs_lora_method3_fine_tuned_model" in methods


def test_slug_normalizes_model_names():
    assert run_full_experiment.slug("Qwen/Qwen3-4B-Instruct-2507") == (
        "qwen_qwen3_4b_instruct_2507"
    )


def test_missing_api_key_records_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan = {
        "method": "openai_gpt_test_llm_multi_agent",
        "provider": "openai",
        "model": "gpt-test",
        "kind": "llm_multi_agent",
    }

    records, errors, latency = run_full_experiment.run_generation_plan([], plan)

    assert records == []
    assert latency == []
    assert errors[0]["error_type"] == "MissingApiKey"
