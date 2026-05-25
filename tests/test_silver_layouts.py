from layout_spatial_reasoning.schemas import Control, FormSpec, Layout, LayoutControl, Row, Section
from layout_spatial_reasoning.training.silver_layouts import (
    LayoutCandidate,
    _source_candidate_id,
    audit_candidate,
    deterministic_coverage_repair,
    filter_final_check_issues,
    parse_review_response,
    select_top_candidates,
)


def _form() -> FormSpec:
    return FormSpec(
        form_id="contact_basic",
        domain="Contact",
        controls=[
            Control(id="c01", label="First name", type="text"),
            Control(id="c02", label="Last name", type="text"),
            Control(id="c03", label="Message", type="long_text"),
        ],
    )


def _layout(width: int = 6) -> Layout:
    return Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact details",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=width),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                        ],
                    ),
                    Row(
                        row_id="s001_r002",
                        controls=[LayoutControl(id="c03", colStart=1, colSpan=12)],
                    ),
                ],
            )
        ]
    )


def test_audit_candidate_scores_valid_layout():
    candidate = LayoutCandidate(
        candidate_id="C01",
        author_provider="openai",
        author_model="test-model",
        layout=_layout(),
    )

    audit_candidate(_form(), candidate)

    assert candidate.validation_errors == []
    assert candidate.deterministic_score > 0


def test_select_top_candidates_prefers_valid_layout():
    valid = LayoutCandidate(
        candidate_id="C01",
        author_provider="openai",
        author_model="test-model",
        layout=_layout(),
        review_scores=[8, 8, 8],
    )
    invalid = LayoutCandidate(
        candidate_id="C02",
        author_provider="claude",
        author_model="test-model",
        layout=_layout(width=8),
        review_scores=[10, 10, 10],
    )
    audit_candidate(_form(), valid)
    audit_candidate(_form(), invalid)

    selected = select_top_candidates([invalid, valid], count=1)

    assert selected == [valid]


def test_select_top_candidates_prefers_review_consensus_over_soft_penalty():
    strong_review = LayoutCandidate(
        candidate_id="C01",
        author_provider="openai",
        author_model="test-model",
        layout=_layout(),
        review_scores=[9, 9],
    )
    weak_review = LayoutCandidate(
        candidate_id="C02",
        author_provider="graph_community",
        author_model="local",
        layout=_layout(),
        review_scores=[2, 2],
    )
    strong_review.deterministic_score = -1.5
    weak_review.deterministic_score = -0.1

    selected = select_top_candidates([weak_review, strong_review], count=1)

    assert selected == [strong_review]


def test_parse_review_response_accepts_nested_review_shape():
    response = parse_review_response(
        """
        {
          "review": {
            "reviews": [
              {
                "candidate_id": "C01",
                "score": 8,
                "issues": ["minor issue"],
                "suggestions": ["tighten grouping"]
              }
            ]
          },
          "top_two": ["C01"]
        }
        """,
        candidate_ids={"C01"},
    )

    assert response.reviews[0].candidate_id == "C01"
    assert response.reviews[0].score == 8
    assert response.top_two == ["C01"]


def test_parse_review_response_accepts_candidate_score_map():
    response = parse_review_response(
        """
        {
          "reviews": {
            "C01": {
              "score": 9,
              "blocking_issues": [],
              "repair_suggestions": ["keep concise"]
            },
            "C02": 6
          }
        }
        """,
        candidate_ids={"C01", "C02"},
    )

    assert [review.candidate_id for review in response.reviews] == ["C01", "C02"]
    assert [review.score for review in response.reviews] == [9, 6]
    assert response.top_two == ["C01", "C02"]


def test_filter_final_check_issues_ignores_false_order_blocker():
    blocking, ignored = filter_final_check_issues(
        ["Order constraint violation: c02 must appear earlier than c01."],
        {
            "grid_utilization": 1.0,
            "grid_constraint_violations": 0,
            "row_underutilization_count": 0,
            "orphan_field_count": 0,
            "reading_order_violation_count": 0,
            "validation_errors": [],
        },
    )

    assert blocking == []
    assert ignored


def test_filter_final_check_issues_ignores_false_missing_blocker_when_valid():
    blocking, ignored = filter_final_check_issues(
        [
            "Several controls are missing from the layout.",
            "The layout is not structurally valid because it is incomplete.",
        ],
        {
            "grid_utilization": 1.0,
            "grid_constraint_violations": 0,
            "row_underutilization_count": 0,
            "orphan_field_count": 10,
            "reading_order_violation_count": 0,
            "validation_errors": [],
        },
    )

    assert blocking == []
    assert ignored


def test_source_candidate_id_strips_repair_suffixes():
    assert _source_candidate_id("C02_C_R_O") == "C02"


def test_deterministic_coverage_repair_removes_unknowns_and_appends_missing():
    form = _form()
    broken = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact details",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c99", colStart=7, colSpan=6),
                        ],
                    ),
                    Row(
                        row_id="s001_r002",
                        controls=[LayoutControl(id="c01", colStart=1, colSpan=6)],
                    ),
                ],
            )
        ]
    )

    repaired = deterministic_coverage_repair(form, broken)
    placed = [
        control.id
        for section in repaired.sections
        for row in section.rows
        for control in row.controls
    ]

    assert placed == ["c01", "c02", "c03"]


def test_deterministic_coverage_repair_wraps_overwide_rows():
    form = _form()
    broken = Layout(
        sections=[
            Section(
                section_id="s001",
                section_name="Contact details",
                rows=[
                    Row(
                        row_id="s001_r001",
                        controls=[
                            LayoutControl(id="c01", colStart=1, colSpan=6),
                            LayoutControl(id="c02", colStart=7, colSpan=6),
                            LayoutControl(id="c03", colStart=1, colSpan=12),
                        ],
                    ),
                ],
            )
        ]
    )

    repaired = deterministic_coverage_repair(form, broken)
    rows = repaired.sections[0].rows

    assert len(rows) == 2
    assert [control.id for control in rows[0].controls] == ["c01", "c02"]
    assert [control.id for control in rows[1].controls] == ["c03"]
