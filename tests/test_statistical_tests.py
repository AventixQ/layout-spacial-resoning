import pandas as pd

from layout_spatial_reasoning.evaluation.statistical_tests import (
    chi_square_error_frequency_test,
    cliffs_delta,
    error_frequency_table,
    friedman_quality_metric_test,
    rank_methods,
    run_statistical_tests,
)


def test_cliffs_delta_detects_direction():
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_friedman_quality_metric_test_returns_posthoc_and_effect_sizes():
    frame = _metrics_frame()

    result = friedman_quality_metric_test(frame, "grid_utilization")

    assert result["test"] == "friedman"
    assert result["n_forms"] == 4
    assert result["methods"] == ["graph_community", "hybrid", "sequential"]
    assert "hybrid__vs__sequential" in result["cliffs_delta"]
    assert len(result["nemenyi_posthoc"]) == 3


def test_error_frequency_table_uses_form_level_error_flags():
    table = error_frequency_table(_metrics_frame())

    assert table.loc["sequential", "orphan_field"] == 4
    assert table.loc["hybrid", "orphan_field"] == 0
    assert table.loc["graph_community", "row_underutilization"] == 4
    assert table.loc["hybrid", "no_error"] == 4


def test_chi_square_error_frequency_test_returns_residuals():
    result = chi_square_error_frequency_test(_metrics_frame())

    assert result["test"] == "chi_square_independence"
    assert result["degrees_of_freedom"] > 0
    assert "standardized_residuals" in result


def test_rank_methods_sorts_by_direction():
    frame = _metrics_frame()

    quality_ranking = rank_methods(frame, "semantic_coherence_row", higher_is_better=True)
    error_ranking = rank_methods(frame, "orphan_field_count", higher_is_better=False)

    assert quality_ranking[0]["method"] == "hybrid"
    assert error_ranking[0]["method"] == "hybrid"


def test_run_statistical_tests_applies_bonferroni_correction():
    result = run_statistical_tests(_metrics_frame(), alpha=0.05)

    assert result["bonferroni_corrected_alpha"] == 0.0125
    assert set(result["quality_metrics"]) == {
        "grid_utilization",
        "semantic_coherence_row",
        "semantic_coherence_section",
    }
    assert result["error_frequencies"]["test"] == "chi_square_independence"


def test_run_statistical_tests_skips_empty_frame():
    result = run_statistical_tests(pd.DataFrame(), alpha=0.05)

    assert result["quality_metrics"]["grid_utilization"]["skipped"] is True
    assert result["error_frequencies"]["skipped"] is True


def test_friedman_quality_metric_test_skips_missing_metric_column():
    result = friedman_quality_metric_test(
        pd.DataFrame([{"form_id": "f1", "method": "m1"}]),
        "grid_utilization",
    )

    assert result["skipped"] is True


def _metrics_frame() -> pd.DataFrame:
    rows = []
    for index in range(1, 5):
        rows.extend(
            [
                _row(
                    form_id=f"f{index}",
                    method="sequential",
                    grid_utilization=1.0,
                    semantic_coherence_row=0.30,
                    semantic_coherence_section=0.35,
                    has_orphan_field=True,
                    orphan_field_count=2,
                ),
                _row(
                    form_id=f"f{index}",
                    method="graph_community",
                    grid_utilization=0.60,
                    semantic_coherence_row=0.70,
                    semantic_coherence_section=0.75,
                    has_row_underutilization=True,
                    has_orphan_field=True,
                    row_underutilization_count=1,
                    orphan_field_count=1,
                ),
                _row(
                    form_id=f"f{index}",
                    method="hybrid",
                    grid_utilization=0.90,
                    semantic_coherence_row=0.85,
                    semantic_coherence_section=0.90,
                ),
            ]
        )
    return pd.DataFrame(rows)


def _row(
    *,
    form_id: str,
    method: str,
    grid_utilization: float,
    semantic_coherence_row: float,
    semantic_coherence_section: float,
    has_grid_constraint_violation: bool = False,
    has_row_underutilization: bool = False,
    has_orphan_field: bool = False,
    has_section_boundary_misplacement: bool = False,
    has_reading_order_violation: bool = False,
    row_underutilization_count: int = 0,
    orphan_field_count: int = 0,
) -> dict[str, object]:
    return {
        "form_id": form_id,
        "method": method,
        "grid_utilization": grid_utilization,
        "semantic_coherence_row": semantic_coherence_row,
        "semantic_coherence_section": semantic_coherence_section,
        "has_grid_constraint_violation": has_grid_constraint_violation,
        "has_row_underutilization": has_row_underutilization,
        "has_orphan_field": has_orphan_field,
        "has_section_boundary_misplacement": has_section_boundary_misplacement,
        "has_reading_order_violation": has_reading_order_violation,
        "row_underutilization_count": row_underutilization_count,
        "orphan_field_count": orphan_field_count,
    }
