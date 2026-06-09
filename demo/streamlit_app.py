"""Interactive demo for form layout generation methods."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from layout_spatial_reasoning.config import env_str
from layout_spatial_reasoning.dataset import load_forms_jsonl
from layout_spatial_reasoning.embeddings.provider import embed_texts, openai_embed_texts
from layout_spatial_reasoning.evaluation.metrics import evaluate_generated_layout
from layout_spatial_reasoning.llm.order_extractor import extract_order_constraints_llm
from layout_spatial_reasoning.methods.fine_tuned_model import (
    generate_layout as fine_tuned_layout,
)
from layout_spatial_reasoning.methods.graph_community import generate_layout as graph_layout
from layout_spatial_reasoning.methods.hybrid import generate_layout as hybrid_layout
from layout_spatial_reasoning.methods.llm_multi_agent import (
    generate_layout as llm_multi_agent_layout,
)
from layout_spatial_reasoning.methods.llm_single import generate_layout as llm_single_layout
from layout_spatial_reasoning.methods.random_baseline import generate_layout as random_layout
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout as sequential_layout
from layout_spatial_reasoning.rendering.html_renderer import render_html
from layout_spatial_reasoning.schemas import (
    Control,
    FormSpec,
    GeneratedLayoutRecord,
    OrderConstraint,
)


CONTROL_TYPES = [
    "text",
    "long_text",
    "choice",
    "multichoice",
    "boolean",
    "date",
    "number",
    "file",
]

METHODS = [
    "sequential",
    "random",
    "graph_community",
    "llm_single_zero_shot",
    "llm_single_few_shot",
    "llm_single_cot",
    "llm_single_structured_output",
    "llm_multi_agent",
    "fine_tuned_model",
    "hybrid",
]

LOCAL_METHODS = ["sequential", "random", "graph_community"]
LLM_METHODS = [
    "llm_single_zero_shot",
    "llm_single_few_shot",
    "llm_single_cot",
    "llm_single_structured_output",
    "llm_multi_agent",
    "fine_tuned_model",
    "hybrid",
]


def main() -> None:
    st.set_page_config(page_title="Layout Spatial Reasoning Demo", layout="wide")
    st.title("Form Layout Generation Demo")

    forms = load_forms_jsonl("data/processed/sample_forms.jsonl")
    selected_form = _select_form(forms)
    controls = _control_editor(selected_form)

    with st.sidebar:
        st.header("Generation")
        method_group = st.radio(
            "Method group",
            ["local / graph", "OpenAI LLM"],
            horizontal=False,
        )
        available_methods = LOCAL_METHODS if method_group == "local / graph" else LLM_METHODS
        method = st.selectbox("Method", available_methods, index=2 if method_group == "local / graph" else 0)
        graph_embedding_provider = "local"
        graph_threshold = 0.25
        llm_model = None
        enable_openai_llm = False

        if method == "random":
            random_seed = st.number_input("Random seed", min_value=0, value=1, step=1)
        else:
            random_seed = 1

        if method in {"graph_community", "hybrid"}:
            st.subheader("Graph settings")
            graph_embedding_provider = st.selectbox(
                "Graph embeddings",
                ["local", "openai"],
                index=0,
            )
            graph_threshold = st.slider(
                "Graph threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )

        if _is_llm_method(method):
            st.subheader("LLM settings")
            enable_openai_llm = st.checkbox(
                "Enable OpenAI LLM call",
                value=False,
                help="LLM methods require a network API call. Keep unchecked while editing controls.",
            )
            llm_model = st.text_input(
                "OpenAI layout model",
                value=_default_llm_model(method),
                disabled=not enable_openai_llm,
            )

        st.subheader("Evaluation settings")
        order_constraint_mode = st.selectbox(
            "Reading-order constraints",
            ["extract with Gemini", "use sample constraints", "none"],
            index=0,
        )
        order_model = st.text_input(
            "Gemini order model",
            value=env_str("GEMINI_ORDER_MODEL", "gemini-3.1-flash-lite")
            or "gemini-3.1-flash-lite",
            disabled=order_constraint_mode != "extract with Gemini",
        )
        metric_embedding_provider = st.selectbox(
            "Metric embeddings",
            ["local", "openai"],
            index=0,
        )
        st.caption("Only OpenAI options make network API calls.")
        generate = st.button("Generate layout", type="primary")

    if generate:
        if _is_llm_method(method) and not enable_openai_llm:
            st.warning("Enable the OpenAI LLM call checkbox before generating with an LLM method.")
            return
        with st.spinner("Generating layout..."):
            try:
                layout = _generate_layout(
                    method,
                    controls,
                    int(random_seed),
                    graph_embedding_provider,
                    graph_threshold,
                    llm_model,
                )
                order_constraints, order_warning = _order_constraints_for_demo(
                    selected_form,
                    controls,
                    order_constraint_mode,
                    order_model,
                )
                effective_graph_provider = graph_embedding_provider
            except Exception as error:  # noqa: BLE001 - user-facing demo.
                if method not in {"graph_community", "hybrid"} or graph_embedding_provider != "openai":
                    st.error(_friendly_error(error))
                    return
                st.warning(
                    "OpenAI graph embeddings failed, so the layout was generated with local embeddings."
                )
                layout = _generate_layout(
                    method,
                    controls,
                    int(random_seed),
                    "local",
                    graph_threshold,
                    llm_model,
                )
                order_constraints, order_warning = _order_constraints_for_demo(
                    selected_form,
                    controls,
                    order_constraint_mode,
                    order_model,
                )
                effective_graph_provider = "local"
            if order_warning:
                st.warning(order_warning)
            st.session_state["layout"] = layout
            st.session_state["method"] = method
            st.session_state["controls"] = controls
            st.session_state["order_constraints"] = order_constraints
            st.session_state["metric_embedding_provider"] = metric_embedding_provider
            st.session_state["graph_embedding_provider"] = effective_graph_provider

    if "layout" not in st.session_state:
        st.info("Choose controls and a method, then generate a layout.")
        return

    layout = st.session_state["layout"]
    method = st.session_state["method"]
    controls = st.session_state["controls"]
    order_constraints = st.session_state.get("order_constraints", [])
    metric_embedding_provider = st.session_state.get(
        "metric_embedding_provider",
        metric_embedding_provider,
    )

    preview_col, metrics_col = st.columns([2, 1], gap="large")
    with preview_col:
        st.subheader("Rendered layout")
        st.html(render_html(controls, layout))

    with metrics_col:
        st.subheader("Metrics")
        record, effective_provider = _safe_evaluate(
            method,
            selected_form,
            controls,
            layout,
            metric_embedding_provider,
            order_constraints,
        )
        if effective_provider != metric_embedding_provider:
            st.warning("OpenAI metric embeddings failed, so metrics were computed with local embeddings.")
            st.session_state["metric_embedding_provider"] = effective_provider
        st.dataframe(_metrics_frame(record), hide_index=True, width="stretch")

        st.subheader("Layout JSON")
        st.json(layout.model_dump(mode="json"))


def _select_form(forms: list[FormSpec]) -> FormSpec:
    form_by_label = {f"{form.form_id} ({form.domain})": form for form in forms}
    label = st.sidebar.selectbox("Sample form", list(form_by_label))
    return form_by_label[label]


def _control_editor(form: FormSpec) -> list[Control]:
    st.subheader("Controls")
    initial_rows = [
        {
            "id": control.id,
            "label": control.label,
            "type": control.type,
            "help_text": control.help_text,
        }
        for control in form.controls
    ]
    edited = st.data_editor(
        pd.DataFrame(initial_rows),
        num_rows="dynamic",
        width="stretch",
        column_config={
            "type": st.column_config.SelectboxColumn("type", options=CONTROL_TYPES),
        },
    )
    controls = []
    for row in edited.fillna("").to_dict(orient="records"):
        if not row["id"] or not row["label"]:
            continue
        controls.append(
            Control(
                id=str(row["id"]),
                label=str(row["label"]),
                type=row["type"] if row["type"] in CONTROL_TYPES else "text",
                help_text=str(row.get("help_text", "")),
            )
        )
    return controls


def _generate_layout(
    method: str,
    controls: list[Control],
    random_seed: int,
    graph_embedding_provider: str,
    graph_threshold: float,
    llm_model: str | None,
):
    if method == "sequential":
        return sequential_layout(controls)
    if method == "random":
        return random_layout(controls, seed=random_seed)
    if method == "graph_community":
        return graph_layout(
            controls,
            embedding_function=_embedding_function(graph_embedding_provider),
            similarity_threshold=graph_threshold,
        )
    if method == "hybrid":
        return hybrid_layout(
            controls,
            embedding_function=_embedding_function(graph_embedding_provider),
            similarity_threshold=graph_threshold,
            model=llm_model,
        )
    if method.startswith("llm_single_"):
        variant = method.removeprefix("llm_single_")
        return llm_single_layout(controls, variant=variant, model=llm_model)
    if method == "llm_multi_agent":
        return llm_multi_agent_layout(controls, model=llm_model)
    if method == "fine_tuned_model":
        return fine_tuned_layout(controls, model=llm_model)
    raise ValueError(f"Unsupported method: {method}")


def _evaluate(
    method: str,
    form: FormSpec,
    controls: list[Control],
    layout,
    embedding_provider: str,
    order_constraints: list[OrderConstraint] | None = None,
):
    demo_form = form.model_copy(
        update={
            "controls": controls,
            "order_constraints": order_constraints or [],
        }
    )
    generated = GeneratedLayoutRecord(
        form_id=demo_form.form_id,
        method=method,
        layout=layout,
    )
    return evaluate_generated_layout(
        demo_form,
        generated,
        embedding_function=_embedding_function(embedding_provider),
    )


def _safe_evaluate(
    method: str,
    form: FormSpec,
    controls: list[Control],
    layout,
    embedding_provider: str,
    order_constraints: list[OrderConstraint] | None = None,
):
    try:
        return (
            _evaluate(
                method,
                form,
                controls,
                layout,
                embedding_provider,
                order_constraints,
            ),
            embedding_provider,
        )
    except Exception as error:
        if embedding_provider != "openai":
            raise
        st.caption(f"Metric embedding fallback reason: {type(error).__name__}")
        return (
            _evaluate(method, form, controls, layout, "local", order_constraints),
            "local",
        )


def _order_constraints_for_demo(
    form: FormSpec,
    controls: list[Control],
    mode: str,
    model: str | None,
) -> tuple[list[OrderConstraint], str | None]:
    if mode == "none":
        return [], None
    if mode == "use sample constraints":
        return _valid_sample_order_constraints(form, controls), None
    try:
        constraints = extract_order_constraints_llm(
            controls,
            provider="gemini",
            model=model or None,
        )
        return constraints, None
    except Exception as error:  # noqa: BLE001 - user-facing demo fallback.
        return [], (
            "Gemini reading-order extraction failed, so reading-order metrics "
            f"were computed without constraints. Original error: {type(error).__name__}: {error}"
        )


def _valid_sample_order_constraints(
    form: FormSpec,
    controls: list[Control],
) -> list[OrderConstraint]:
    control_ids = {control.id for control in controls}
    return [
        constraint
        for constraint in form.order_constraints
        if constraint.before in control_ids and constraint.after in control_ids
    ]


def _embedding_function(provider: str):
    if provider == "openai":
        return openai_embed_texts
    return embed_texts


def _friendly_error(error: Exception) -> str:
    if error.__class__.__name__ == "SSLError" or "SSL" in str(error):
        return (
            "OpenAI call failed because the Streamlit process hit a local SSL error. "
            "Use local methods/embeddings in the demo, or run the CLI pipeline for OpenAI calls. "
            f"Original error: {type(error).__name__}: {error}"
        )
    return f"{type(error).__name__}: {error}"


def _is_llm_method(method: str) -> bool:
    return (
        method.startswith("llm_single_")
        or method == "llm_multi_agent"
        or method == "fine_tuned_model"
        or method == "hybrid"
    )


def _default_llm_model(method: str) -> str:
    if method == "fine_tuned_model":
        return env_str("FINE_TUNED_LAYOUT_MODEL", "") or ""
    return env_str("OPENAI_LAYOUT_MODEL", "gpt-4.1") or "gpt-4.1"


def _metrics_frame(record) -> pd.DataFrame:
    values = record.model_dump()
    rows = [
        {"metric": key, "value": value}
        for key, value in values.items()
        if key not in {"form_id", "method"}
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
