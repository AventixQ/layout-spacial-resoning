"""Statistical tests for experiment results."""

from itertools import combinations
import math
from typing import Any

import pandas as pd
from scipy import stats

from layout_spatial_reasoning.schemas.results import EvaluationRecord


QUALITY_METRICS = [
    "grid_utilization",
    "semantic_coherence_row",
    "semantic_coherence_section",
]

ERROR_FORM_LEVEL_COLUMNS = {
    "grid_constraint_violation": "has_grid_constraint_violation",
    "row_underutilization": "has_row_underutilization",
    "orphan_field": "has_orphan_field",
    "section_boundary_misplacement": "has_section_boundary_misplacement",
    "reading_order_violation": "has_reading_order_violation",
}


def run_statistical_tests(
    records: list[EvaluationRecord] | pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run the thesis statistical analysis plan on evaluation records."""
    frame = _records_to_frame(records)
    omnibus_test_count = len(QUALITY_METRICS) + 1
    corrected_alpha = alpha / omnibus_test_count
    return {
        "alpha": alpha,
        "bonferroni_corrected_alpha": corrected_alpha,
        "quality_metrics": {
            metric: friedman_quality_metric_test(
                frame,
                metric,
                alpha=corrected_alpha,
            )
            for metric in QUALITY_METRICS
        },
        "error_frequencies": chi_square_error_frequency_test(
            frame,
            alpha=corrected_alpha,
        ),
    }


def friedman_quality_metric_test(
    frame: pd.DataFrame,
    metric: str,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare methods on one repeated-measures quality metric."""
    pivot = (
        frame.pivot_table(
            index="form_id",
            columns="method",
            values=metric,
            aggfunc="mean",
        )
        .dropna(axis=0)
        .sort_index(axis=1)
    )
    methods = list(pivot.columns)
    if len(methods) < 2 or len(pivot) < 2:
        return _skipped_result("Friedman test requires at least two methods and forms.")

    statistic, p_value = stats.friedmanchisquare(
        *[pivot[method].to_numpy() for method in methods]
    )
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0).to_dict()
    posthoc = nemenyi_posthoc_from_ranks(
        ranks,
        alpha=alpha,
    )
    effect_sizes = {
        f"{left}__vs__{right}": cliffs_delta(
            pivot[left].tolist(),
            pivot[right].tolist(),
        )
        for left, right in combinations(methods, 2)
    }
    return {
        "test": "friedman",
        "metric": metric,
        "n_forms": int(len(pivot)),
        "methods": methods,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "mean_ranks": {key: float(value) for key, value in mean_ranks.items()},
        "nemenyi_posthoc": posthoc,
        "cliffs_delta": effect_sizes,
    }


def nemenyi_posthoc_from_ranks(
    ranks: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Run Nemenyi-style post-hoc comparisons from per-form method ranks."""
    methods = list(ranks.columns)
    method_count = len(methods)
    form_count = len(ranks)
    if method_count < 2 or form_count < 2:
        return []

    q_alpha = stats.studentized_range.ppf(
        1 - alpha,
        method_count,
        math.inf,
    ) / math.sqrt(2)
    critical_difference = q_alpha * math.sqrt(
        method_count * (method_count + 1) / (6 * form_count)
    )
    mean_ranks = ranks.mean(axis=0)
    return [
        {
            "method_a": left,
            "method_b": right,
            "mean_rank_difference": float(abs(mean_ranks[left] - mean_ranks[right])),
            "critical_difference": float(critical_difference),
            "significant": bool(
                abs(mean_ranks[left] - mean_ranks[right]) > critical_difference
            ),
        }
        for left, right in combinations(methods, 2)
    ]


def cliffs_delta(left_values: list[float], right_values: list[float]) -> float:
    """Compute Cliff's delta for two paired or unpaired value lists."""
    if not left_values or not right_values:
        return 0.0

    greater = 0
    lower = 0
    for left in left_values:
        for right in right_values:
            if left > right:
                greater += 1
            elif left < right:
                lower += 1
    return (greater - lower) / (len(left_values) * len(right_values))


def chi_square_error_frequency_test(
    frame: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare form-level error-category distributions between methods."""
    table = error_frequency_table(frame)
    if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() == 0:
        return _skipped_result("Chi-square test requires at least two methods/categories.")

    statistic, p_value, degrees_of_freedom, expected = stats.chi2_contingency(
        table.to_numpy(),
        correction=False,
    )
    expected_frame = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns,
    )
    residuals = (table - expected_frame) / expected_frame.pow(0.5)
    return {
        "test": "chi_square_independence",
        "observed": _nested_float_dict(table),
        "expected": _nested_float_dict(expected_frame),
        "standardized_residuals": _nested_float_dict(residuals),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def error_frequency_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build method x error-category table from form-level error flags."""
    columns = [
        column
        for column in ERROR_FORM_LEVEL_COLUMNS.values()
        if column in frame.columns
    ]
    table = frame[["method", *columns]].copy()
    for column in columns:
        table[column] = table[column].fillna(False).astype(bool).astype(int)
    table["no_error"] = (table[columns].sum(axis=1) == 0).astype(int)
    grouped = table.groupby("method", sort=True)[[*columns, "no_error"]].sum()
    renamed = grouped.rename(
        columns={
            column: category
            for category, column in ERROR_FORM_LEVEL_COLUMNS.items()
            if column in grouped.columns
        }
    )
    return renamed.loc[:, renamed.sum(axis=0) > 0]


def rank_methods(
    frame: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    """Rank methods by average metric value."""
    summary = (
        frame.groupby("method", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=not higher_is_better)
        .reset_index(drop=True)
    )
    return [
        {
            "rank": index + 1,
            "method": row["method"],
            "mean_value": float(row[metric]),
        }
        for index, row in summary.iterrows()
    ]


def _records_to_frame(records: list[EvaluationRecord] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame([record.model_dump() for record in records])


def _nested_float_dict(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(index): {
            str(column): float(value)
            for column, value in row.items()
        }
        for index, row in frame.iterrows()
    }


def _skipped_result(reason: str) -> dict[str, Any]:
    return {
        "skipped": True,
        "reason": reason,
        "significant": False,
    }
