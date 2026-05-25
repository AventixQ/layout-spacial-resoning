"""Multi-model silver reference layout construction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.dataset.io import load_forms_jsonl
from layout_spatial_reasoning.evaluation.grid_constraints import grid_constraint_violations
from layout_spatial_reasoning.evaluation.metrics import grid_utilization
from layout_spatial_reasoning.evaluation.orphan_field import orphan_field_count
from layout_spatial_reasoning.evaluation.reading_order import reading_order_violation_count
from layout_spatial_reasoning.evaluation.row_underutilization import row_underutilization_count
from layout_spatial_reasoning.llm.providers import PROVIDER_ENV_KEYS, generate_json
from layout_spatial_reasoning.methods.graph_community import (
    generate_layout_from_env as graph_layout,
)
from layout_spatial_reasoning.methods.llm_single import (
    _layout_json_schema,
    controls_to_json,
    parse_layout_response,
)
from layout_spatial_reasoning.schemas.form import FormSpec
from layout_spatial_reasoning.schemas.layout import Layout, LayoutControl, Row, Section
from layout_spatial_reasoning.schemas.validation import validate_layout


DEFAULT_SILVER_MODELS = {
    "openai": "gpt-5.4-mini",
    "claude": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-flash-lite",
}
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = {
    "generation": PROMPT_DIR / "silver_generation.txt",
    "review": PROMPT_DIR / "silver_review.txt",
    "repair": PROMPT_DIR / "silver_repair.txt",
    "final_check": PROMPT_DIR / "silver_final_check.txt",
}
MAX_REPAIR_ATTEMPTS = 3
MAX_FINAL_REPAIR_ATTEMPTS = 2
DEFAULT_PROVIDER_RETRIES = 2
DEFAULT_RETRY_DELAY_SECONDS = 2.0


class CandidateReview(BaseModel):
    """LLM review for one anonymized layout candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    score: int = Field(ge=1, le=10)
    blocking_issues: list[str] = Field(default_factory=list)
    repair_suggestions: list[str] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    """Structured response for cross-model layout review."""

    model_config = ConfigDict(extra="forbid")

    reviews: list[CandidateReview]
    top_two: list[str] = Field(min_length=1, max_length=2)


class FinalCheckResponse(BaseModel):
    """Structured response for final sanity checking."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    blocking_issues: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SilverProvider:
    """One LLM provider/model participating in silver-label construction."""

    provider: str
    model: str


@dataclass
class LayoutCandidate:
    """Generated layout candidate with deterministic audit fields."""

    candidate_id: str
    author_provider: str
    author_model: str
    layout: Layout
    validation_errors: list[str] = field(default_factory=list)
    deterministic_score: float = 0.0
    review_scores: list[int] = field(default_factory=list)
    reviewer_scores: dict[str, int] = field(default_factory=dict)
    reviewer_top_two_votes: dict[str, bool] = field(default_factory=dict)
    review_issues: list[str] = field(default_factory=list)
    repair_suggestions: list[str] = field(default_factory=list)
    selected_for_repair: bool = False
    final_selected: bool = False

    @property
    def mean_review_score(self) -> float:
        if not self.review_scores:
            return 0.0
        return statistics.mean(self.review_scores)

    @property
    def consensus_score(self) -> float:
        return self.mean_review_score + self.deterministic_score


@dataclass(frozen=True)
class SilverLayoutResult:
    """Final result for one form."""

    form: FormSpec
    target_layouts: list[Layout]
    candidates: list[LayoutCandidate]
    errors: list[dict[str, Any]]


def build_silver_reference_layouts(
    form: FormSpec,
    *,
    providers: list[SilverProvider] | None = None,
    include_graph_candidate: bool = True,
    target_layout_count: int = 2,
) -> SilverLayoutResult:
    """Generate, review, repair, and final-check silver target layouts."""
    active_providers = providers or providers_from_env()
    candidates, errors = generate_candidates(
        form,
        active_providers,
        include_graph_candidate=include_graph_candidate,
    )
    for candidate in candidates:
        audit_candidate(form, candidate)

    review_candidates(form, candidates, active_providers, errors)
    selected = select_top_candidates(candidates, count=target_layout_count)
    for candidate in selected:
        candidate.selected_for_repair = True
    final_layouts = []
    for index, candidate in enumerate(selected):
        repaired = repair_until_deterministically_valid(
            form,
            candidate,
            providers=active_providers,
            provider_offset=index,
            errors=errors,
        )
        accepted = final_accept_or_repair(
            form,
            repaired,
            providers=active_providers,
            provider_offset=index,
            errors=errors,
        )
        if accepted is None:
            continue
        final_layouts.append(accepted.layout)
        source_id = _source_candidate_id(accepted.candidate_id)
        for source_candidate in candidates:
            if source_candidate.candidate_id == source_id:
                source_candidate.final_selected = True
                break
        if len(final_layouts) >= target_layout_count:
            break

    return SilverLayoutResult(
        form=form.model_copy(update={"target_layouts": final_layouts}),
        target_layouts=final_layouts,
        candidates=candidates,
        errors=errors,
    )


def providers_from_env() -> list[SilverProvider]:
    """Build silver-label provider list from environment variables."""
    load_env()
    provider_names = [
        provider.strip()
        for provider in os.environ.get(
            "SILVER_LAYOUT_PROVIDERS",
            "openai,claude,gemini",
        ).split(",")
        if provider.strip()
    ]
    providers = [
        SilverProvider(
            provider=provider,
            model=os.environ.get(
                f"SILVER_{provider.upper()}_MODEL",
                DEFAULT_SILVER_MODELS.get(provider, ""),
            ),
        )
        for provider in provider_names
    ]
    return [provider for provider in providers if provider.model]


def _source_candidate_id(candidate_id: str) -> str:
    return candidate_id.split("_", 1)[0]


def generate_candidates(
    form: FormSpec,
    providers: list[SilverProvider],
    *,
    include_graph_candidate: bool,
) -> tuple[list[LayoutCandidate], list[dict[str, Any]]]:
    """Generate initial candidates from all configured providers."""
    candidates: list[LayoutCandidate] = []
    errors: list[dict[str, Any]] = []
    generation_jobs: list[tuple[int, SilverProvider]] = []
    for index, provider in enumerate(providers, start=1):
        if provider.provider in PROVIDER_ENV_KEYS and not os.environ.get(
            PROVIDER_ENV_KEYS[provider.provider]
        ):
            errors.append(
                {
                    "stage": "generation",
                    "provider": provider.provider,
                    "model": provider.model,
                    "error_type": "MissingApiKey",
                    "error": f"{PROVIDER_ENV_KEYS[provider.provider]} is not set.",
                }
            )
            continue
        generation_jobs.append((index, provider))

    workers = _provider_worker_count(len(generation_jobs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(generate_layout_candidate, form, provider): (index, provider)
            for index, provider in generation_jobs
        }
        for future in as_completed(future_to_job):
            index, provider = future_to_job[future]
            try:
                layout = future.result()
            except Exception as error:  # noqa: BLE001 - batch pipeline records failures.
                errors.append(
                    {
                        "stage": "generation",
                        "provider": provider.provider,
                        "model": provider.model,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            candidates.append(
                LayoutCandidate(
                    candidate_id=f"C{index:02d}",
                    author_provider=provider.provider,
                    author_model=provider.model,
                    layout=layout,
                )
            )
    candidates.sort(key=lambda candidate: candidate.candidate_id)

    if include_graph_candidate:
        try:
            candidates.append(
                LayoutCandidate(
                    candidate_id=f"C{len(providers) + 1:02d}",
                    author_provider="graph_community",
                    author_model="local",
                    layout=graph_layout(form.controls),
                )
            )
        except Exception as error:  # noqa: BLE001
            errors.append(
                {
                    "stage": "graph_candidate",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    return candidates, errors


def generate_layout_candidate(form: FormSpec, provider: SilverProvider) -> Layout:
    """Ask one provider to generate a high-quality reference layout."""
    return _with_provider_retries(
        lambda: parse_layout_lenient(
            generate_json(
                provider.provider,
                model=provider.model,
                response_format=_json_schema_response_format(
                    "silver_layout",
                    _layout_json_schema(),
                ),
                assistant_prefill=_assistant_prefill(provider, '{"sections":'),
                temperature=0,
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": _prompt("generation")},
                    {"role": "user", "content": _form_payload(form)},
                ],
            )
        ),
        stage="generation",
        provider=provider,
        form_id=form.form_id,
    )


def review_candidates(
    form: FormSpec,
    candidates: list[LayoutCandidate],
    providers: list[SilverProvider],
    errors: list[dict[str, Any]],
) -> None:
    """Run anonymous cross-review and attach scores/issues to candidates."""
    if not candidates:
        return
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    payload = _review_payload(form, candidates)
    workers = _provider_worker_count(len(providers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_provider = {
            executor.submit(review_layout_candidates, payload, provider): provider
            for provider in providers
        }
        review_results: list[tuple[SilverProvider, ReviewResponse]] = []
        for future in as_completed(future_to_provider):
            provider = future_to_provider[future]
            try:
                response = future.result()
            except Exception as error:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "review",
                        "provider": provider.provider,
                        "model": provider.model,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            review_results.append((provider, response))
    for provider, response in sorted(
        review_results,
        key=lambda item: (item[0].provider, item[0].model),
    ):
        for review in response.reviews:
            candidate = candidate_by_id.get(review.candidate_id)
            if candidate is None:
                continue
            candidate.review_scores.append(review.score)
            candidate.reviewer_scores[_provider_key(provider)] = review.score
            candidate.review_issues.extend(review.blocking_issues)
            candidate.repair_suggestions.extend(review.repair_suggestions)
        for candidate_id in response.top_two:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is not None:
                candidate.reviewer_top_two_votes[_provider_key(provider)] = True


def review_layout_candidates(payload: str, provider: SilverProvider) -> ReviewResponse:
    """Ask one provider to review anonymous layout candidates."""
    return _with_provider_retries(
        lambda: parse_review_response(
            generate_json(
                provider.provider,
                model=provider.model,
                response_format=_json_schema_response_format(
                    "silver_layout_review",
                    _review_json_schema(),
                ),
                assistant_prefill=_assistant_prefill(provider, '{"reviews":'),
                temperature=0,
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": _prompt("review")},
                    {"role": "user", "content": payload},
                ],
            ),
            candidate_ids=_candidate_ids_from_payload(payload),
        ),
        stage="review",
        provider=provider,
        form_id=_form_id_from_payload(payload),
    )


def repair_candidate(
    form: FormSpec,
    candidate: LayoutCandidate,
    *,
    repair_provider: SilverProvider,
    errors: list[dict[str, Any]],
) -> LayoutCandidate:
    """Repair a selected candidate using aggregated review issues."""
    issues = sorted(set(candidate.review_issues + candidate.repair_suggestions))
    payload = {
        "form": json.loads(_form_payload(form)),
        "candidate_id": candidate.candidate_id,
        "layout": candidate.layout.model_dump(mode="json"),
        "deterministic_validation_errors": candidate.validation_errors,
        "deterministic_metrics": deterministic_metrics(form, candidate.layout),
        "review_issues_and_suggestions": issues,
    }
    try:
        repaired_layout = _with_provider_retries(
            lambda: deterministic_coverage_repair(
                form,
                parse_layout_lenient(
                    generate_json(
                        repair_provider.provider,
                        model=repair_provider.model,
                        response_format=_json_schema_response_format(
                            "silver_layout_repair",
                            _layout_json_schema(),
                        ),
                        assistant_prefill=_assistant_prefill(
                            repair_provider,
                            '{"sections":',
                        ),
                        temperature=0,
                        max_tokens=8192,
                        messages=[
                            {"role": "system", "content": _prompt("repair")},
                            {
                                "role": "user",
                                "content": json.dumps(payload, ensure_ascii=False),
                            },
                        ],
                    )
                ),
            ),
            stage="repair",
            provider=repair_provider,
            form_id=form.form_id,
            candidate_id=candidate.candidate_id,
        )
    except Exception as error:  # noqa: BLE001
        errors.append(
            {
                "stage": "repair",
                "candidate_id": candidate.candidate_id,
                "provider": repair_provider.provider,
                "model": repair_provider.model,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        repaired_layout = candidate.layout

    return LayoutCandidate(
        candidate_id=f"{candidate.candidate_id}_R",
        author_provider=f"repair:{repair_provider.provider}",
        author_model=repair_provider.model,
        layout=repaired_layout,
        review_scores=list(candidate.review_scores),
        review_issues=list(candidate.review_issues),
        repair_suggestions=list(candidate.repair_suggestions),
    )


def deterministic_coverage_repair(form: FormSpec, layout: Layout) -> Layout:
    """Drop invalid/duplicate placements and append omitted controls deterministically."""
    controls_by_id = {control.id: control for control in form.controls}
    seen: set[str] = set()
    sections: list[Section] = []
    for section in layout.sections:
        rows: list[Row] = []
        for row in section.rows:
            row_controls: list[LayoutControl] = []
            for control in row.controls:
                if control.id not in controls_by_id or control.id in seen:
                    continue
                seen.add(control.id)
                row_controls.append(
                    LayoutControl(
                        id=control.id,
                        colStart=control.colStart,
                        colSpan=_normalized_span(controls_by_id[control.id].type),
                    )
                )
            if row_controls:
                rows.append(Row(row_id=row.row_id, controls=row_controls))
        if rows:
            sections.append(
                Section(
                    section_id=section.section_id,
                    section_name=section.section_name,
                    rows=rows,
                )
            )
    if not sections:
        sections.append(
            Section(
                section_id="s001",
                section_name=_fallback_section_name(form),
                rows=[],
            )
        )

    missing_ids = [control.id for control in form.controls if control.id not in seen]
    if missing_ids:
        target_section_index = _best_append_section_index(form, sections, missing_ids)
        rows = list(sections[target_section_index].rows)
        for control_id in missing_ids:
            control = controls_by_id[control_id]
            rows.append(
                Row(
                    row_id="tmp",
                    controls=[
                        LayoutControl(
                            id=control.id,
                            colStart=1,
                            colSpan=_normalized_span(control.type),
                        )
                    ],
                )
            )
        sections[target_section_index] = Section(
            section_id=sections[target_section_index].section_id,
            section_name=sections[target_section_index].section_name,
            rows=rows,
        )
    return _renumber_layout(Layout(sections=sections))


def repair_until_deterministically_valid(
    form: FormSpec,
    candidate: LayoutCandidate,
    *,
    providers: list[SilverProvider],
    provider_offset: int,
    errors: list[dict[str, Any]],
) -> LayoutCandidate:
    """Iteratively repair until hard deterministic gates pass."""
    current = candidate
    for attempt in range(MAX_REPAIR_ATTEMPTS):
        current = deterministic_coverage_candidate_repair(form, current)
        current = deterministic_order_repair(form, current)
        audit_candidate(form, current)
        metrics = deterministic_metrics(form, current.layout)
        if is_deterministically_acceptable(metrics):
            return current
        errors.append(
            {
                "stage": "repair_loop",
                "candidate_id": current.candidate_id,
                "attempt": attempt,
                "metrics": metrics,
            }
        )
        current = repair_candidate(
            form,
            current,
            repair_provider=providers[(provider_offset + attempt) % len(providers)],
            errors=errors,
        )
    audit_candidate(form, current)
    return current


def deterministic_coverage_candidate_repair(
    form: FormSpec,
    candidate: LayoutCandidate,
) -> LayoutCandidate:
    repaired = deterministic_coverage_repair(form, candidate.layout)
    if repaired == candidate.layout:
        return candidate
    return LayoutCandidate(
        candidate_id=f"{candidate.candidate_id}_C",
        author_provider=f"coverage_repair:{candidate.author_provider}",
        author_model=candidate.author_model,
        layout=repaired,
        review_scores=list(candidate.review_scores),
        reviewer_scores=dict(candidate.reviewer_scores),
        reviewer_top_two_votes=dict(candidate.reviewer_top_two_votes),
        review_issues=list(candidate.review_issues),
        repair_suggestions=list(candidate.repair_suggestions),
        selected_for_repair=candidate.selected_for_repair,
    )


def _normalized_span(control_type: str) -> int:
    if control_type in {"long_text", "file"}:
        return 12
    return 6


def _best_append_section_index(
    form: FormSpec,
    sections: list[Section],
    missing_ids: list[str],
) -> int:
    controls_by_id = {control.id: control for control in form.controls}
    first_missing = controls_by_id[missing_ids[0]]
    keywords = _control_keywords(first_missing)
    if keywords:
        best_index = 0
        best_score = -1
        for index, section in enumerate(sections):
            name = section.section_name.lower()
            score = sum(1 for keyword in keywords if keyword in name)
            if score > best_score:
                best_score = score
                best_index = index
        if best_score > 0:
            return best_index
    return len(sections) - 1


def _control_keywords(control: Any) -> set[str]:
    text = f"{control.label} {control.type}".lower()
    groups = {
        "personal": {"name", "birth", "email", "phone", "contact", "address"},
        "policy": {"policy", "claim", "incident", "loss", "settlement"},
        "payment": {"bank", "sort", "account", "payment"},
        "employment": {"employer", "occupation", "income", "job"},
        "consent": {"consent", "declaration", "terms", "fatca", "marketing"},
        "attachments": {"file", "photo", "document", "quote", "receipt"},
    }
    return {
        group
        for group, tokens in groups.items()
        if any(token in text for token in tokens)
    }


def _fallback_section_name(form: FormSpec) -> str:
    domain = form.domain.strip()
    if domain:
        return f"{domain} details"
    return "Form details"


def final_accept_or_repair(
    form: FormSpec,
    candidate: LayoutCandidate,
    *,
    providers: list[SilverProvider],
    provider_offset: int,
    errors: list[dict[str, Any]],
) -> LayoutCandidate | None:
    """Run final checks and repair blocking issues before giving up."""
    current = candidate
    best_usable: LayoutCandidate | None = None
    best_usable_metrics: dict[str, Any] | None = None
    for attempt in range(MAX_FINAL_REPAIR_ATTEMPTS + 1):
        current = deterministic_coverage_candidate_repair(form, current)
        current = deterministic_order_repair(form, current)
        audit_candidate(form, current)
        metrics = deterministic_metrics(form, current.layout)
        if is_minimum_usable(form, current.layout):
            best_usable = current
            best_usable_metrics = metrics
        if not is_deterministically_acceptable(metrics):
            errors.append(
                {
                    "stage": "final_deterministic_quality_gate",
                    "candidate_id": current.candidate_id,
                    "attempt": attempt,
                    "metrics": metrics,
                }
            )
        else:
            passed, blocking_issues = final_sanity_check(
                form,
                current,
                providers,
                errors=errors,
            )
            if passed:
                return current
            current.review_issues.extend(blocking_issues)

        if attempt >= MAX_FINAL_REPAIR_ATTEMPTS:
            break
        current = repair_candidate(
            form,
            current,
            repair_provider=providers[(provider_offset + attempt) % len(providers)],
            errors=errors,
        )

    if best_usable is not None:
        errors.append(
            {
                "stage": "accepted_after_repair_with_llm_warnings",
                "candidate_id": best_usable.candidate_id,
                "metrics": best_usable_metrics
                or deterministic_metrics(form, best_usable.layout),
            }
        )
        return best_usable
    return None


def deterministic_order_repair(form: FormSpec, candidate: LayoutCandidate) -> LayoutCandidate:
    """Repair reading-order violations by moving constrained controls earlier."""
    if not form.order_constraints:
        return candidate
    layout = candidate.layout.model_copy(deep=True)
    for _ in range(len(form.order_constraints) * 2):
        changed = False
        positions = _layout_positions(layout)
        for constraint in form.order_constraints:
            before_position = positions.get(constraint.before)
            after_position = positions.get(constraint.after)
            if before_position is None or after_position is None:
                continue
            if before_position > after_position:
                _move_control_before(layout, constraint.before, constraint.after)
                changed = True
                positions = _layout_positions(layout)
        if not changed:
            break
    if layout == candidate.layout:
        return candidate
    return LayoutCandidate(
        candidate_id=f"{candidate.candidate_id}_O",
        author_provider=f"order_repair:{candidate.author_provider}",
        author_model=candidate.author_model,
        layout=_renumber_layout(layout),
        review_scores=list(candidate.review_scores),
        reviewer_scores=dict(candidate.reviewer_scores),
        reviewer_top_two_votes=dict(candidate.reviewer_top_two_votes),
        review_issues=list(candidate.review_issues),
        repair_suggestions=list(candidate.repair_suggestions),
        selected_for_repair=candidate.selected_for_repair,
    )


def _layout_positions(layout: Layout) -> dict[str, tuple[int, int, int, int]]:
    positions = {}
    flat_index = 0
    for section_index, section in enumerate(layout.sections):
        for row_index, row in enumerate(section.rows):
            ordered = sorted(enumerate(row.controls), key=lambda item: item[1].colStart)
            for control_index, control in ordered:
                positions[control.id] = (flat_index, section_index, row_index, control_index)
                flat_index += 1
    return positions


def _move_control_before(layout: Layout, moving_id: str, target_id: str) -> None:
    moving = _pop_control(layout, moving_id)
    if moving is None:
        return
    target_location = _find_control(layout, target_id)
    if target_location is None:
        return
    section_index, row_index, target_control_index = target_location
    target_row = layout.sections[section_index].rows[row_index]
    row_width = sum(control.colSpan for control in target_row.controls)
    if row_width + moving.colSpan <= 12:
        target_row.controls.insert(target_control_index, moving)
        return
    layout.sections[section_index].rows.insert(
        row_index,
        Row(row_id="tmp", controls=[moving]),
    )


def _pop_control(layout: Layout, control_id: str):
    for section in layout.sections:
        for row in section.rows:
            for index, control in enumerate(row.controls):
                if control.id == control_id:
                    removed = row.controls.pop(index)
                    _remove_empty_rows(layout)
                    return removed
    return None


def _find_control(layout: Layout, control_id: str) -> tuple[int, int, int] | None:
    for section_index, section in enumerate(layout.sections):
        for row_index, row in enumerate(section.rows):
            for control_index, control in enumerate(row.controls):
                if control.id == control_id:
                    return section_index, row_index, control_index
    return None


def _remove_empty_rows(layout: Layout) -> None:
    for section in layout.sections:
        section.rows[:] = [row for row in section.rows if row.controls]


def _reflow_row(row: Row) -> None:
    col_start = 1
    controls = []
    for control in row.controls:
        if col_start + control.colSpan - 1 > 12:
            break
        controls.append(
            LayoutControl(
                id=control.id,
                colStart=col_start,
                colSpan=control.colSpan,
            )
        )
        col_start += control.colSpan
    row.controls[:] = controls


def _renumber_layout(layout: Layout) -> Layout:
    sections = []
    for section_index, section in enumerate(layout.sections, start=1):
        rows = []
        row_index = 1
        for row in section.rows:
            col_start = 1
            controls = []
            for control in row.controls:
                if col_start + control.colSpan - 1 > 12 and controls:
                    rows.append(
                        Row(
                            row_id=f"s{section_index:03d}_r{row_index:03d}",
                            controls=controls,
                        )
                    )
                    row_index += 1
                    col_start = 1
                    controls = []
                col_span = min(control.colSpan, 12)
                controls.append(
                    LayoutControl(
                        id=control.id,
                        colStart=col_start,
                        colSpan=col_span,
                    )
                )
                col_start += col_span
            if controls:
                rows.append(
                    Row(
                        row_id=f"s{section_index:03d}_r{row_index:03d}",
                        controls=controls,
                    )
                )
                row_index += 1
        sections.append(
            Section(
                section_id=f"s{section_index:03d}",
                section_name=section.section_name,
                rows=rows,
            )
        )
    return Layout(sections=sections)


def final_sanity_check(
    form: FormSpec,
    candidate: LayoutCandidate,
    providers: list[SilverProvider],
    *,
    errors: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Require each configured provider to pass the final candidate."""
    metrics = deterministic_metrics(form, candidate.layout)
    payload = {
        "form": json.loads(_form_payload(form)),
        "layout": candidate.layout.model_dump(mode="json"),
        "deterministic_metrics": metrics,
    }
    all_blocking_issues: list[str] = []
    workers = _provider_worker_count(len(providers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_provider = {
            executor.submit(final_check_layout, payload, provider): provider
            for provider in providers
        }
        final_results: list[tuple[SilverProvider, FinalCheckResponse]] = []
        for future in as_completed(future_to_provider):
            provider = future_to_provider[future]
            try:
                response = future.result()
            except Exception as error:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "final_check",
                        "candidate_id": candidate.candidate_id,
                        "provider": provider.provider,
                        "model": provider.model,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                all_blocking_issues.append(str(error))
                continue
            final_results.append((provider, response))
    for provider, response in sorted(
        final_results,
        key=lambda item: (item[0].provider, item[0].model),
    ):
        if not response.passed:
            blocking_issues, ignored_issues = filter_final_check_issues(
                response.blocking_issues,
                metrics,
            )
            all_blocking_issues.extend(blocking_issues)
            errors.append(
                {
                    "stage": "final_check",
                    "candidate_id": candidate.candidate_id,
                    "provider": provider.provider,
                    "model": provider.model,
                    "blocking_issues": blocking_issues,
                    "ignored_issues": ignored_issues,
                }
            )
    return not all_blocking_issues, all_blocking_issues


def filter_final_check_issues(
    issues: list[str],
    metrics: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Drop LLM final-check complaints contradicted by deterministic metrics."""
    blocking: list[str] = []
    ignored: list[str] = []
    for issue in issues:
        lowered = issue.lower()
        is_order_issue = any(
            token in lowered
            for token in ("order constraint", "order_constraints", "reading order")
        )
        if is_order_issue and metrics["reading_order_violation_count"] == 0:
            ignored.append(issue)
            continue
        is_grid_issue = any(
            token in lowered
            for token in ("grid", "overlap", "column", "colstart", "colspan")
        )
        if (
            is_grid_issue
            and metrics["grid_constraint_violations"] == 0
            and not metrics["validation_errors"]
        ):
            ignored.append(issue)
            continue
        is_missing_issue = any(
            token in lowered
            for token in (
                "missing control",
                "missing controls",
                "controls are missing",
                "not placed exactly once",
                "more than once",
                "duplicate",
                "duplicated",
                "miscounted",
                "control count",
                "placed controls",
                "include all form controls",
                "do not appear exactly once",
                "does not include all",
                "incomplete",
                "not structurally valid",
                "omitted",
                "orphan field",
                "orphan fields",
                "unresolved controls",
            )
        )
        if is_missing_issue and not metrics["validation_errors"]:
            ignored.append(issue)
            continue
        blocking.append(issue)
    return blocking, ignored


def final_check_layout(payload: dict[str, Any], provider: SilverProvider) -> FinalCheckResponse:
    """Ask one provider for a final pass/fail sanity check."""
    return _with_provider_retries(
        lambda: parse_final_check_response(
            generate_json(
                provider.provider,
                model=provider.model,
                response_format=_json_schema_response_format(
                    "silver_layout_final_check",
                    _final_check_json_schema(),
                ),
                assistant_prefill=_assistant_prefill(provider, '{"passed":'),
                temperature=0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": _prompt("final_check")},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
        ),
        stage="final_check",
        provider=provider,
        form_id=payload.get("form", {}).get("form_id", ""),
    )


def audit_candidate(form: FormSpec, candidate: LayoutCandidate) -> None:
    """Attach deterministic validation errors and score."""
    candidate.validation_errors = validate_layout(form.controls, candidate.layout)
    metrics = deterministic_metrics(form, candidate.layout)
    score = metrics["grid_utilization"]
    score -= 2.0 * len(candidate.validation_errors)
    score -= 0.5 * metrics["grid_constraint_violations"]
    score -= 0.25 * metrics["row_underutilization_count"]
    score -= 0.25 * metrics["orphan_field_count"]
    score -= 0.5 * metrics["reading_order_violation_count"]
    candidate.deterministic_score = score


def parse_layout_lenient(content: str) -> Layout:
    """Parse provider layout JSON, stripping common extra metadata fields."""
    data = json.loads(_extract_json(content))
    if "layout" in data and isinstance(data["layout"], dict):
        data = data["layout"]
    if "sections" not in data and any(key in data for key in ("form_id", "domain")):
        data = {"sections": data.get("sections", [])}
    for key in list(data):
        if key != "sections":
            data.pop(key, None)
    for section in data.get("sections", []):
        for key in list(section):
            if key not in {"section_id", "section_name", "rows"}:
                section.pop(key, None)
        for row in section.get("rows", []):
            for key in list(row):
                if key not in {"row_id", "controls"}:
                    row.pop(key, None)
            for control in row.get("controls", []):
                for key in list(control):
                    if key not in {"id", "colStart", "colSpan"}:
                        control.pop(key, None)
    return Layout.model_validate(data)


def parse_review_response(content: str, *, candidate_ids: set[str]) -> ReviewResponse:
    """Parse reviewer JSON with small normalizations for weaker models."""
    data = json.loads(_extract_json(content))
    if "reviews" not in data and "review" in data:
        review = data["review"]
        if isinstance(review, dict) and "reviews" in review:
            data["reviews"] = review["reviews"]
        elif isinstance(review, dict) and "candidates" in review:
            data["reviews"] = review["candidates"]
        elif isinstance(review, list):
            data["reviews"] = review
    if isinstance(data.get("reviews"), dict):
        data["reviews"] = [
            {"candidate_id": candidate_id, **review}
            if isinstance(review, dict)
            else {"candidate_id": candidate_id, "score": review}
            for candidate_id, review in data["reviews"].items()
        ]
    reviews = []
    for item in data.get("reviews", []):
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id") or item.get("id") or item.get("candidate")
        if candidate_id not in candidate_ids:
            continue
        reviews.append(
            {
                "candidate_id": candidate_id,
                "score": int(item.get("score", item.get("rating", 5))),
                "blocking_issues": _string_list(
                    item.get("blocking_issues", item.get("issues", []))
                ),
                "repair_suggestions": _string_list(
                    item.get("repair_suggestions", item.get("suggestions", []))
                ),
            }
        )
    top_two = data.get("top_two") or data.get("top_2") or data.get("best_candidates")
    if not top_two:
        top_two = [
            item["candidate_id"]
            for item in sorted(reviews, key=lambda review: -review["score"])[:2]
        ]
    normalized = {
        "reviews": reviews,
        "top_two": [candidate_id for candidate_id in top_two if candidate_id in candidate_ids][:2],
    }
    if not normalized["top_two"] and reviews:
        normalized["top_two"] = [reviews[0]["candidate_id"]]
    return ReviewResponse.model_validate(normalized)


def parse_final_check_response(content: str) -> FinalCheckResponse:
    """Parse final-check JSON with small normalizations for weaker models."""
    data = json.loads(_extract_json(content))
    if "passed" not in data:
        if "pass" in data:
            data["passed"] = data["pass"]
        elif "status" in data:
            data["passed"] = str(data["status"]).lower() in {"pass", "passed", "ok"}
        else:
            data["passed"] = False
    if "blocking_issues" not in data:
        data["blocking_issues"] = _string_list(
            data.get("issues", data.get("notes", []))
        )
    return FinalCheckResponse.model_validate(
        {
            "passed": data["passed"],
            "blocking_issues": _string_list(data["blocking_issues"]),
        }
    )


def deterministic_metrics(form: FormSpec, layout: Layout) -> dict[str, Any]:
    """Compute cheap deterministic checks used in selection and final review."""
    return {
        "grid_utilization": grid_utilization(layout),
        "grid_constraint_violations": len(grid_constraint_violations(layout)),
        "row_underutilization_count": row_underutilization_count(form.controls, layout),
        "orphan_field_count": orphan_field_count(form.controls, layout),
        "reading_order_violation_count": reading_order_violation_count(
            layout,
            form.order_constraints,
        ),
        "validation_errors": validate_layout(form.controls, layout),
    }


def is_deterministically_acceptable(metrics: dict[str, Any]) -> bool:
    """Hard gates that cannot be waived for silver labels."""
    return (
        not metrics["validation_errors"]
        and metrics["grid_constraint_violations"] == 0
        and metrics["reading_order_violation_count"] == 0
    )


def is_minimum_usable(form: FormSpec, layout: Layout) -> bool:
    """Fallback acceptance if LLM sanity is too strict after repairs."""
    metrics = deterministic_metrics(form, layout)
    return (
        is_deterministically_acceptable(metrics)
        and metrics["orphan_field_count"] <= 2
        and metrics["row_underutilization_count"] <= 2
    )


def select_top_candidates(
    candidates: list[LayoutCandidate],
    *,
    count: int,
) -> list[LayoutCandidate]:
    """Select top candidates by consensus while preferring valid layouts."""
    return sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate.validation_errors),
            -candidate.consensus_score,
            candidate_quality_penalty(candidate),
            candidate.candidate_id,
        ),
    )[:count]


def candidate_quality_penalty(candidate: LayoutCandidate) -> float:
    """Soft deterministic penalty used only for ordering candidates."""
    return max(0.0, -candidate.deterministic_score)


def build_silver_dataset_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    start: int = 0,
    audit_path: str | Path | None = None,
) -> dict[str, int]:
    """Build a JSONL file containing forms with generated silver target layouts."""
    forms = load_forms_jsonl(input_path)
    selected_forms = forms[start : None if limit is None else start + limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_file = None
    if audit_path is not None:
        audit_output = Path(audit_path)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_file = audit_output.open("w", encoding="utf-8")
    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "forms_requested": len(selected_forms),
        "forms_written": 0,
        "forms_failed": 0,
    }
    try:
        with output.open("w", encoding="utf-8") as file:
            for result in _build_silver_results(selected_forms):
                if len(result.target_layouts) < 2:
                    manifest["forms_failed"] += 1
                file.write(result.form.model_dump_json())
                file.write("\n")
                if audit_file is not None:
                    audit_file.write(json.dumps(_result_audit_record(result), ensure_ascii=True))
                    audit_file.write("\n")
                manifest["forms_written"] += 1
    finally:
        if audit_file is not None:
            audit_file.close()
    return manifest


def _build_silver_results(forms: list[FormSpec]) -> list[SilverLayoutResult]:
    workers = _form_worker_count(len(forms))
    if workers <= 1:
        results = []
        for index, form in enumerate(forms, start=1):
            results.append(_build_silver_result_with_logging(index, len(forms), form))
        return results
    results: list[SilverLayoutResult | None] = [None] * len(forms)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _build_silver_result_with_logging,
                index + 1,
                len(forms),
                form,
            ): index
            for index, form in enumerate(forms)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


def _build_silver_result_with_logging(
    index: int,
    total: int,
    form: FormSpec,
) -> SilverLayoutResult:
    _log_form_start(index, total, form)
    started = time.monotonic()
    result = build_silver_reference_layouts(form)
    _log_form_done(index, total, result, started)
    return result


def _with_provider_retries(
    operation,
    *,
    stage: str,
    provider: SilverProvider,
    form_id: str,
    candidate_id: str | None = None,
):
    attempts = _provider_retry_count() + 1
    delay = _retry_delay_seconds()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - caller records final failure.
            last_error = error
            if attempt >= attempts:
                break
            context = f" form={form_id}" if form_id else ""
            if candidate_id:
                context += f" candidate={candidate_id}"
            print(
                "[silver] retry "
                f"{stage} {provider.provider}:{provider.model}"
                f"{context} attempt={attempt + 1}/{attempts} "
                f"after {type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(delay * attempt)
    if last_error is None:
        raise RuntimeError("Provider retry loop failed without an exception.")
    raise last_error


def _log_form_start(index: int, total: int, form: FormSpec) -> None:
    print(
        "[silver] start "
        f"{index}/{total} {form.form_id} "
        f"domain={form.domain} controls={len(form.controls)} "
        f"constraints={len(form.order_constraints)}",
        flush=True,
    )


def _log_form_done(
    index: int,
    total: int,
    result: SilverLayoutResult,
    started: float,
) -> None:
    elapsed = time.monotonic() - started
    status = "ok" if len(result.target_layouts) >= 2 else "failed"
    print(
        "[silver] done  "
        f"{index}/{total} {result.form.form_id} "
        f"targets={len(result.target_layouts)} "
        f"errors={len(result.errors)} status={status} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def _form_id_from_payload(payload: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    form = data.get("form", {})
    if isinstance(form, dict):
        return str(form.get("form_id", ""))
    return ""


def _provider_retry_count() -> int:
    return int(os.environ.get("SILVER_PROVIDER_RETRIES", DEFAULT_PROVIDER_RETRIES))


def _retry_delay_seconds() -> float:
    return float(
        os.environ.get("SILVER_RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY_SECONDS)
    )


def _provider_worker_count(job_count: int) -> int:
    default = max(1, job_count)
    return _bounded_worker_count(
        os.environ.get("SILVER_PROVIDER_WORKERS"),
        default=default,
        job_count=job_count,
    )


def _form_worker_count(job_count: int) -> int:
    return _bounded_worker_count(
        os.environ.get("SILVER_FORM_WORKERS"),
        default=1,
        job_count=job_count,
    )


def _bounded_worker_count(
    value: str | None,
    *,
    default: int,
    job_count: int,
) -> int:
    if job_count <= 0:
        return 1
    if value is None or value == "":
        requested = default
    else:
        requested = int(value)
    return max(1, min(requested, job_count))


def _result_audit_record(result: SilverLayoutResult) -> dict[str, Any]:
    return {
        "form_id": result.form.form_id,
        "target_layout_count": len(result.target_layouts),
        "errors": result.errors,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "author_provider": candidate.author_provider,
                "author_model": candidate.author_model,
                "validation_errors": candidate.validation_errors,
                "deterministic_score": candidate.deterministic_score,
                "review_scores": candidate.review_scores,
                "reviewer_scores": candidate.reviewer_scores,
                "reviewer_top_two_votes": candidate.reviewer_top_two_votes,
                "mean_review_score": candidate.mean_review_score,
                "consensus_score": candidate.consensus_score,
                "selected_for_repair": candidate.selected_for_repair,
                "final_selected": candidate.final_selected,
                "review_issue_count": len(candidate.review_issues),
                "repair_suggestion_count": len(candidate.repair_suggestions),
            }
            for candidate in result.candidates
        ],
    }


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _provider_key(provider: SilverProvider) -> str:
    return f"{provider.provider}:{provider.model}"


def _assistant_prefill(provider: SilverProvider, value: str) -> str | None:
    if provider.provider == "claude":
        return value
    return None


def _review_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews", "top_two"],
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "candidate_id",
                        "score",
                        "blocking_issues",
                        "repair_suggestions",
                    ],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 10},
                        "blocking_issues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "repair_suggestions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "top_two": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {"type": "string"},
            },
        },
    }


def _final_check_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["passed", "blocking_issues"],
        "properties": {
            "passed": {"type": "boolean"},
            "blocking_issues": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _form_payload(form: FormSpec) -> str:
    payload = json.loads(controls_to_json(form.controls))
    payload["form_id"] = form.form_id
    payload["domain"] = form.domain
    payload["order_constraints"] = [
        constraint.model_dump(mode="json")
        for constraint in form.order_constraints
    ]
    payload["order_constraints_explained"] = _order_constraint_explanations(form)
    return json.dumps(payload, ensure_ascii=False)


def _order_constraint_explanations(form: FormSpec) -> list[str]:
    labels = {control.id: control.label for control in form.controls}
    return [
        (
            f"{constraint.before} ({labels.get(constraint.before, constraint.before)}) "
            f"must appear earlier in visual reading order than "
            f"{constraint.after} ({labels.get(constraint.after, constraint.after)})."
        )
        for constraint in form.order_constraints
    ]


def _review_payload(form: FormSpec, candidates: list[LayoutCandidate]) -> str:
    payload = {
        "form": json.loads(_form_payload(form)),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "layout": candidate.layout.model_dump(mode="json"),
                "deterministic_validation_errors": candidate.validation_errors,
                "deterministic_metrics": deterministic_metrics(form, candidate.layout),
            }
            for candidate in candidates
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _candidate_ids_from_payload(payload: str) -> set[str]:
    data = json.loads(payload)
    return {
        candidate["candidate_id"]
        for candidate in data.get("candidates", [])
        if "candidate_id" in candidate
    }


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            _, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        return stripped[start : start + end]
    raise ValueError("No JSON object found in silver layout response.")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _prompt(name: str) -> str:
    return PROMPT_FILES[name].read_text(encoding="utf-8")
