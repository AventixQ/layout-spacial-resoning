import importlib.util
from pathlib import Path

from layout_spatial_reasoning.methods.sequential_baseline import generate_layout
from layout_spatial_reasoning.schemas import Control, FormSpec

_SPEC = importlib.util.spec_from_file_location(
    "streamlit_app",
    Path(__file__).resolve().parents[1] / "demo" / "streamlit_app.py",
)
assert _SPEC is not None
streamlit_app = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(streamlit_app)


def test_safe_evaluate_falls_back_to_local_embeddings(monkeypatch):
    controls = [Control(id="c01", label="Email", type="text")]
    form = FormSpec(form_id="contact", domain="Contact", controls=controls)
    layout = generate_layout(controls)

    def broken_openai(_texts):
        raise RuntimeError("network issue")

    monkeypatch.setattr(streamlit_app, "openai_embed_texts", broken_openai)

    _record, provider = streamlit_app._safe_evaluate(
        "sequential",
        form,
        controls,
        layout,
        "openai",
    )

    assert provider == "local"
