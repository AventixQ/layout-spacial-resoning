"""Join experiment metrics/errors with form metadata and summarize failure patterns."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from layout_spatial_reasoning.dataset import load_forms_jsonl


DEFAULT_RUN_DIR = Path("outputs/experiments/latest")
DEFAULT_FORMS_PATH = Path("data/splits/val.jsonl")


def main() -> None:
    run_dir = Path(os.environ.get("EXPERIMENT_RUN_DIR", str(DEFAULT_RUN_DIR)))
    forms_path = Path(os.environ.get("FORMS_PATH", str(DEFAULT_FORMS_PATH)))
    output_dir = run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    form_metadata = load_form_metadata(forms_path)
    metrics = read_metrics(run_dir, form_metadata)
    errors = read_errors(run_dir, form_metadata)

    metrics.to_csv(output_dir / "metrics_with_form_metadata.csv", index=False)
    errors.to_csv(output_dir / "errors_with_form_metadata.csv", index=False)

    summaries = {
        "method_by_domain": summarize_group(metrics, ["method", "domain"]),
        "method_by_category": summarize_group(metrics, ["method", "category"]),
        "method_by_length_bucket": summarize_group(metrics, ["method", "length_bucket"]),
        "hardest_forms": hardest_forms(metrics, errors),
        "error_summary": summarize_errors(errors),
    }
    for name, frame in summaries.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    write_markdown(output_dir, run_dir, forms_path, summaries)
    print(f"Wrote diagnostics to {output_dir}.")


def load_form_metadata(forms_path: Path) -> pd.DataFrame:
    rows = []
    for form in load_forms_jsonl(forms_path):
        control_count = len(form.controls)
        rows.append(
            {
                "form_id": form.form_id,
                "domain": form.domain,
                "category": category_from_form_id(form.form_id),
                "control_count": control_count,
                "length_bucket": length_bucket(control_count),
                "order_constraint_count": len(form.order_constraints),
                "control_types": ",".join(sorted({control.type for control in form.controls})),
                "long_text_count": sum(1 for control in form.controls if control.type == "long_text"),
                "file_count": sum(1 for control in form.controls if control.type == "file"),
            }
        )
    return pd.DataFrame(rows)


def read_metrics(run_dir: Path, form_metadata: pd.DataFrame) -> pd.DataFrame:
    metrics_path = run_dir / "metrics" / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_path)
    return metrics.merge(form_metadata, on="form_id", how="left")


def read_errors(run_dir: Path, form_metadata: pd.DataFrame) -> pd.DataFrame:
    errors_path = run_dir / "errors" / "all_errors.jsonl"
    rows = []
    if errors_path.exists():
        with errors_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
    errors = pd.DataFrame(rows)
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "form_id",
                "method",
                "provider",
                "model",
                "error_type",
                "error",
                *[column for column in form_metadata.columns if column != "form_id"],
            ]
        )
    return errors.merge(form_metadata, on="form_id", how="left")


def summarize_group(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(group_columns, dropna=False)
        .agg(
            form_count=("form_id", "count"),
            mean_control_count=("control_count", "mean"),
            mean_grid_utilization=("grid_utilization", "mean"),
            mean_row_coherence=("semantic_coherence_row", "mean"),
            mean_section_coherence=("semantic_coherence_section", "mean"),
            mean_row_underutilization=("row_underutilization_count", "mean"),
            mean_orphan_fields=("orphan_field_count", "mean"),
            mean_section_boundary=("section_boundary_misplacement_score", "mean"),
            form_level_grid_violation=("has_grid_constraint_violation", "mean"),
            form_level_orphan_field=("has_orphan_field", "mean"),
            form_level_reading_order_violation=("has_reading_order_violation", "mean"),
            mean_reading_order_violation_rate=("reading_order_violation_rate", "mean"),
            mean_missing_controls=("missing_control_count", "mean"),
            mean_duplicated_controls=("duplicated_control_count", "mean"),
            mean_unknown_controls=("unknown_control_count", "mean"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def hardest_forms(metrics: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    failure_columns = [
        "has_grid_constraint_violation",
        "has_row_underutilization",
        "has_orphan_field",
        "has_section_boundary_misplacement",
        "has_reading_order_violation",
    ]
    frame = metrics.copy()
    for column in failure_columns:
        frame[column] = frame[column].fillna(False).astype(bool)
    frame["failure_flags"] = frame[failure_columns].sum(axis=1)
    summary = (
        frame.groupby(
            [
                "form_id",
                "domain",
                "category",
                "control_count",
                "length_bucket",
                "order_constraint_count",
            ],
            dropna=False,
        )
        .agg(
            evaluated_methods=("method", "count"),
            methods_with_any_issue=("failure_flags", lambda values: int((values > 0).sum())),
            mean_failure_flags=("failure_flags", "mean"),
            mean_orphan_fields=("orphan_field_count", "mean"),
            mean_reading_order_violation_rate=("reading_order_violation_rate", "mean"),
            mean_grid_utilization=("grid_utilization", "mean"),
        )
        .reset_index()
    )
    if not errors.empty:
        error_counts = errors.groupby("form_id").size().reset_index(name="generation_errors")
        summary = summary.merge(error_counts, on="form_id", how="left")
    else:
        summary["generation_errors"] = 0
    summary["generation_errors"] = summary["generation_errors"].fillna(0).astype(int)
    return summary.sort_values(
        ["generation_errors", "methods_with_any_issue", "mean_failure_flags"],
        ascending=[False, False, False],
    )


def summarize_errors(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame()
    return (
        errors.groupby(["method", "error_type", "domain", "category", "length_bucket"], dropna=False)
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )


def write_markdown(
    output_dir: Path,
    run_dir: Path,
    forms_path: Path,
    summaries: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Experiment Diagnostics",
        "",
        f"Run: `{run_dir}`",
        f"Forms: `{forms_path}`",
        "",
    ]
    for title, name in [
        ("Method By Length Bucket", "method_by_length_bucket"),
        ("Method By Domain", "method_by_domain"),
        ("Hardest Forms", "hardest_forms"),
        ("Error Summary", "error_summary"),
    ]:
        frame = summaries[name]
        lines.extend([f"## {title}", ""])
        if frame.empty:
            lines.extend(["No records.", ""])
        else:
            lines.extend([markdown_table(frame.head(40)), ""])
    (output_dir / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def category_from_form_id(form_id: str) -> str:
    return re.sub(r"_\d+$", "", form_id)


def length_bucket(control_count: int) -> str:
    if control_count <= 10:
        return "01_short_<=10"
    if control_count <= 20:
        return "02_medium_11_20"
    if control_count <= 35:
        return "03_long_21_35"
    return "04_very_long_36+"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(format_markdown_value(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(rows)


def format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
