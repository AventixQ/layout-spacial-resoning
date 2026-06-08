"""Summarize validation/test split distributions for sanity checks."""

from __future__ import annotations

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from layout_spatial_reasoning.dataset import load_forms_jsonl


DEFAULT_SPLITS = {
    "val": Path("data/splits/val.jsonl"),
    "test": Path("data/splits/test.jsonl"),
}
OUTPUT_DIR = Path("outputs/dataset_split_summary")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for split, path in DEFAULT_SPLITS.items():
        for form in load_forms_jsonl(path):
            records.append(
                {
                    "split": split,
                    "form_id": form.form_id,
                    "domain": form.domain,
                    "category": category_from_form_id(form.form_id),
                    "control_count": len(form.controls),
                    "order_constraint_count": len(form.order_constraints),
                }
            )

    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT_DIR / "form_distribution_records.csv", index=False)

    domain_counts = count_table(frame, "domain")
    category_counts = count_table(frame, "category")
    length_counts = count_table(frame, "control_count")
    length_summary = (
        frame.groupby("split")["control_count"]
        .agg(["count", "min", "median", "mean", "max"])
        .reset_index()
    )
    constraint_summary = (
        frame.groupby("split")["order_constraint_count"]
        .agg(["min", "median", "mean", "max"])
        .reset_index()
    )

    domain_counts.to_csv(OUTPUT_DIR / "domain_counts.csv", index=False)
    category_counts.to_csv(OUTPUT_DIR / "category_counts.csv", index=False)
    length_counts.to_csv(OUTPUT_DIR / "control_count_distribution.csv", index=False)
    length_summary.to_csv(OUTPUT_DIR / "control_count_summary.csv", index=False)
    constraint_summary.to_csv(OUTPUT_DIR / "order_constraint_summary.csv", index=False)

    write_markdown_summary(
        frame,
        domain_counts,
        category_counts,
        length_counts,
        length_summary,
        constraint_summary,
    )
    plot_domain_counts(domain_counts)
    plot_category_counts(category_counts)
    plot_control_count_distribution(length_counts)
    plot_control_count_boxplot(frame)

    print(f"Wrote split distribution summary to {OUTPUT_DIR}.")


def category_from_form_id(form_id: str) -> str:
    """Return the stable category prefix, e.g. contact_business_021 -> contact_business."""
    return re.sub(r"_\d+$", "", form_id)


def count_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = (
        frame.groupby(["split", column])
        .size()
        .reset_index(name="count")
        .pivot(index=column, columns="split", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    for split in DEFAULT_SPLITS:
        if split not in counts:
            counts[split] = 0
    counts["total"] = counts[list(DEFAULT_SPLITS)].sum(axis=1)
    counts["abs_diff"] = (counts["val"] - counts["test"]).abs()
    return counts.sort_values(["abs_diff", "total", column], ascending=[False, False, True])


def write_markdown_summary(
    frame: pd.DataFrame,
    domain_counts: pd.DataFrame,
    category_counts: pd.DataFrame,
    length_counts: pd.DataFrame,
    length_summary: pd.DataFrame,
    constraint_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Validation/Test Split Distribution",
        "",
        "## Overall",
        "",
        markdown_table(
            frame.groupby("split")
            .agg(
                forms=("form_id", "count"),
                domains=("domain", "nunique"),
                categories=("category", "nunique"),
                min_controls=("control_count", "min"),
                median_controls=("control_count", "median"),
                mean_controls=("control_count", "mean"),
                max_controls=("control_count", "max"),
            )
            .reset_index()
        ),
        "",
        "## Domain Counts",
        "",
        markdown_table(domain_counts),
        "",
        "## Category Counts",
        "",
        markdown_table(category_counts),
        "",
        "## Control Count Distribution",
        "",
        markdown_table(length_counts),
        "",
        "## Control Count Summary",
        "",
        markdown_table(length_summary),
        "",
        "## Order Constraint Summary",
        "",
        markdown_table(constraint_summary),
        "",
        "## Figures",
        "",
        "- `domain_counts.png`",
        "- `category_counts.png`",
        "- `control_count_distribution.png`",
        "- `control_count_boxplot.png`",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


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
        return f"{value:.2f}"
    return str(value)


def plot_domain_counts(domain_counts: pd.DataFrame) -> None:
    ordered = domain_counts.sort_values("total", ascending=False)
    plot_grouped_bars(
        ordered,
        category_col="domain",
        title="Validation/test forms by domain",
        output_path=OUTPUT_DIR / "domain_counts.png",
    )


def plot_category_counts(category_counts: pd.DataFrame) -> None:
    ordered = category_counts.sort_values("total", ascending=True)
    height = max(6, 0.28 * len(ordered))
    fig, ax = plt.subplots(figsize=(10, height))
    y = range(len(ordered))
    ax.barh([item - 0.18 for item in y], ordered["val"], height=0.36, label="val")
    ax.barh([item + 0.18 for item in y], ordered["test"], height=0.36, label="test")
    ax.set_yticks(list(y), ordered["category"])
    ax.set_xlabel("Forms")
    ax.set_title("Validation/test forms by category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "category_counts.png", dpi=160)
    plt.close(fig)


def plot_control_count_distribution(length_counts: pd.DataFrame) -> None:
    ordered = length_counts.sort_values("control_count")
    plot_grouped_bars(
        ordered,
        category_col="control_count",
        title="Validation/test forms by number of controls",
        output_path=OUTPUT_DIR / "control_count_distribution.png",
    )


def plot_grouped_bars(
    frame: pd.DataFrame,
    *,
    category_col: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(frame))
    ax.bar([item - 0.18 for item in x], frame["val"], width=0.36, label="val")
    ax.bar([item + 0.18 for item in x], frame["test"], width=0.36, label="test")
    ax.set_xticks(list(x), frame[category_col].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Forms")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_control_count_boxplot(frame: pd.DataFrame) -> None:
    values = [
        frame.loc[frame["split"] == split, "control_count"].tolist()
        for split in DEFAULT_SPLITS
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values, tick_labels=list(DEFAULT_SPLITS), showmeans=True)
    ax.set_ylabel("Controls per form")
    ax.set_title("Control count distribution by split")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "control_count_boxplot.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
