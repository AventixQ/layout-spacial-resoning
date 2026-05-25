"""Quality checks for unsorted input control lists."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Any

from layout_spatial_reasoning.dataset.build_dataset import (
    FORMS_PER_SCENARIO,
    _SCENARIOS,
)
from layout_spatial_reasoning.dataset.io import load_forms_jsonl
from layout_spatial_reasoning.schemas.control import ControlType
from layout_spatial_reasoning.schemas.form import FormSpec


EXPECTED_DOMAINS = {
    domain: len(scenarios) * FORMS_PER_SCENARIO
    for domain_scenarios in _SCENARIOS.values()
    for domain in {domain_scenarios[0].domain}
    for scenarios in [domain_scenarios]
}
ALLOWED_CONTROL_TYPES = set(ControlType.__args__)
ASCII_TEXT_RE = re.compile(r"^[\x20-\x7e]*$")
CONTROL_ID_RE = re.compile(r"^c\d{2}$")


@dataclass(frozen=True)
class AuditIssue:
    """One dataset audit issue."""

    severity: str
    category: str
    message: str
    form_id: str | None = None
    control_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimilarFormPair:
    """Pair of forms with very similar control label sets."""

    left_form_id: str
    right_form_id: str
    left_domain: str
    right_domain: str
    jaccard: float
    sequence_ratio: float
    shared_label_count: int
    left_control_count: int
    right_control_count: int


@dataclass(frozen=True)
class ControlListAuditReport:
    """Audit report for unsorted control lists."""

    path: str
    form_count: int
    domain_counts: dict[str, int]
    scenario_counts: dict[str, int]
    control_count_distribution: dict[int, int]
    control_type_counts: dict[str, int]
    unique_label_count: int
    order_constraint_count: int
    issues: list[AuditIssue]
    similar_form_pairs: list[SimilarFormPair]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "form_count": self.form_count,
            "domain_counts": self.domain_counts,
            "scenario_counts": self.scenario_counts,
            "control_count_distribution": self.control_count_distribution,
            "control_type_counts": self.control_type_counts,
            "unique_label_count": self.unique_label_count,
            "order_constraint_count": self.order_constraint_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.__dict__ for issue in self.issues],
            "similar_form_pairs": [pair.__dict__ for pair in self.similar_form_pairs],
        }


def audit_control_lists(
    forms: list[FormSpec],
    *,
    path: str = "",
    high_similarity_threshold: float = 0.9,
) -> ControlListAuditReport:
    """Audit unsorted control lists in the generated FLB dataset."""
    issues: list[AuditIssue] = []
    scenario_by_id = _scenario_registry()
    label_type_by_scenario = _label_type_registry()

    domain_counts = Counter(form.domain for form in forms)
    scenario_counts = Counter(_scenario_id(form.form_id) for form in forms)
    control_count_distribution = Counter(len(form.controls) for form in forms)
    control_type_counts = Counter(
        control.type
        for form in forms
        for control in form.controls
    )
    unique_labels = {
        control.label
        for form in forms
        for control in form.controls
    }
    order_constraint_count = sum(len(form.order_constraints) for form in forms)

    _audit_dataset_shape(forms, domain_counts, scenario_counts, issues)
    for form in forms:
        scenario_id = _scenario_id(form.form_id)
        scenario = scenario_by_id.get(scenario_id)
        _audit_form_identity(form, scenario, issues)
        _audit_controls(
            form,
            scenario,
            label_type_by_scenario.get(scenario_id, {}),
            issues,
        )
        _audit_order_constraints(form, issues)

    similar_pairs = find_similar_forms(
        forms,
        threshold=high_similarity_threshold,
    )
    for pair in similar_pairs:
        severity = "error" if pair.jaccard == 1.0 else "warning"
        issues.append(
            AuditIssue(
                severity=severity,
                category="form_similarity",
                message=(
                    "Forms have highly similar control label sets "
                    f"(Jaccard={pair.jaccard:.3f})."
                ),
                form_id=pair.left_form_id,
                details={
                    "other_form_id": pair.right_form_id,
                    "left_domain": pair.left_domain,
                    "right_domain": pair.right_domain,
                    "sequence_ratio": pair.sequence_ratio,
                    "shared_label_count": pair.shared_label_count,
                },
            )
        )

    return ControlListAuditReport(
        path=path,
        form_count=len(forms),
        domain_counts=dict(sorted(domain_counts.items())),
        scenario_counts=dict(sorted(scenario_counts.items())),
        control_count_distribution=dict(sorted(control_count_distribution.items())),
        control_type_counts=dict(sorted(control_type_counts.items())),
        unique_label_count=len(unique_labels),
        order_constraint_count=order_constraint_count,
        issues=issues,
        similar_form_pairs=similar_pairs,
    )


def audit_control_lists_file(
    input_path: str | Path,
    *,
    high_similarity_threshold: float = 0.9,
) -> ControlListAuditReport:
    """Load and audit a JSONL file with form specs."""
    path = Path(input_path)
    return audit_control_lists(
        load_forms_jsonl(path),
        path=str(path),
        high_similarity_threshold=high_similarity_threshold,
    )


def write_audit_report(
    report: ControlListAuditReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write machine-readable and human-readable audit reports."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: ControlListAuditReport) -> str:
    """Render a concise Markdown report."""
    lines = [
        "# Control list dataset audit",
        "",
        f"Source: `{report.path}`",
        f"Forms: {report.form_count}",
        f"Errors: {report.error_count}",
        f"Warnings: {report.warning_count}",
        f"Unique labels: {report.unique_label_count}",
        f"Order constraints: {report.order_constraint_count}",
        "",
        "## Domain counts",
        _table(["domain", "forms"], report.domain_counts.items()),
        "",
        "## Control count distribution",
        _table(
            ["controls_per_form", "forms"],
            ((str(count), total) for count, total in report.control_count_distribution.items()),
        ),
        "",
        "## Control type counts",
        _table(["type", "controls"], report.control_type_counts.items()),
        "",
        "## Scenario counts",
        _table(["scenario", "forms"], report.scenario_counts.items()),
        "",
        "## Issues by category",
        _table(
            ["category", "count"],
            Counter(issue.category for issue in report.issues).most_common(),
        ),
        "",
        "## Similar form pairs",
        _similar_pairs_table(report.similar_form_pairs[:100]),
        "",
        "## Issue details",
        _issues_table(report.issues[:200]),
    ]
    return "\n".join(lines).strip() + "\n"


def find_similar_forms(
    forms: list[FormSpec],
    *,
    threshold: float = 0.9,
) -> list[SimilarFormPair]:
    """Find highly similar forms by label-set Jaccard similarity."""
    signatures = [
        (
            form,
            frozenset(_normalized_label(control.label) for control in form.controls),
            " | ".join(sorted(_normalized_label(control.label) for control in form.controls)),
        )
        for form in forms
    ]
    pairs: list[SimilarFormPair] = []
    for left_index, (left_form, left_labels, left_text) in enumerate(signatures):
        for right_form, right_labels, right_text in signatures[left_index + 1:]:
            if not left_labels or not right_labels:
                continue
            shared = left_labels.intersection(right_labels)
            jaccard = len(shared) / len(left_labels.union(right_labels))
            if jaccard < threshold:
                continue
            pairs.append(
                SimilarFormPair(
                    left_form_id=left_form.form_id,
                    right_form_id=right_form.form_id,
                    left_domain=left_form.domain,
                    right_domain=right_form.domain,
                    jaccard=jaccard,
                    sequence_ratio=SequenceMatcher(None, left_text, right_text).ratio(),
                    shared_label_count=len(shared),
                    left_control_count=len(left_labels),
                    right_control_count=len(right_labels),
                )
            )
    return sorted(
        pairs,
        key=lambda pair: (
            -pair.jaccard,
            -pair.sequence_ratio,
            pair.left_form_id,
            pair.right_form_id,
        ),
    )


def _audit_dataset_shape(
    forms: list[FormSpec],
    domain_counts: Counter[str],
    scenario_counts: Counter[str],
    issues: list[AuditIssue],
) -> None:
    if len(forms) != sum(EXPECTED_DOMAINS.values()):
        issues.append(
            AuditIssue(
                severity="warning",
                category="dataset_size",
                message=(
                    f"Dataset contains {len(forms)} forms; expected "
                    f"{sum(EXPECTED_DOMAINS.values())} for the full generated FLB."
                ),
            )
        )

    for domain, expected_count in EXPECTED_DOMAINS.items():
        actual_count = domain_counts.get(domain, 0)
        if actual_count != expected_count:
            issues.append(
                AuditIssue(
                    severity="warning",
                    category="domain_balance",
                    message=(
                        f"Domain {domain!r} has {actual_count} forms; "
                        f"expected {expected_count}."
                    ),
                    details={"domain": domain, "actual": actual_count, "expected": expected_count},
                )
            )

    expected_scenarios = _scenario_registry()
    for scenario_id in expected_scenarios:
        actual_count = scenario_counts.get(scenario_id, 0)
        if actual_count != FORMS_PER_SCENARIO:
            issues.append(
                AuditIssue(
                    severity="warning",
                    category="scenario_balance",
                    message=(
                        f"Scenario {scenario_id!r} has {actual_count} forms; "
                        f"expected {FORMS_PER_SCENARIO}."
                    ),
                    details={
                        "scenario_id": scenario_id,
                        "actual": actual_count,
                        "expected": FORMS_PER_SCENARIO,
                    },
                )
            )


def _audit_form_identity(
    form: FormSpec,
    scenario: Any | None,
    issues: list[AuditIssue],
) -> None:
    if scenario is None:
        issues.append(
            AuditIssue(
                severity="error",
                category="scenario_link",
                message="Form id does not match any known dataset scenario.",
                form_id=form.form_id,
            )
        )
        return

    if form.domain != scenario.domain:
        issues.append(
            AuditIssue(
                severity="error",
                category="domain_scenario_mismatch",
                message=(
                    f"Form domain {form.domain!r} does not match scenario domain "
                    f"{scenario.domain!r}."
                ),
                form_id=form.form_id,
            )
        )


def _audit_controls(
    form: FormSpec,
    scenario: Any | None,
    expected_label_types: dict[str, str],
    issues: list[AuditIssue],
) -> None:
    ids = [control.id for control in form.controls]
    id_counts = Counter(ids)
    expected_ids = [f"c{index + 1:02d}" for index in range(len(form.controls))]
    if ids != expected_ids:
        issues.append(
            AuditIssue(
                severity="error",
                category="control_id_sequence",
                message="Control ids should be consecutive c01..cNN in the unsorted list.",
                form_id=form.form_id,
                details={"actual": ids, "expected": expected_ids},
            )
        )
    for control_id, count in id_counts.items():
        if count > 1:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="duplicate_control_id",
                    message=f"Control id {control_id!r} appears {count} times.",
                    form_id=form.form_id,
                    control_id=control_id,
                )
            )

    label_counts = Counter(_normalized_label(control.label) for control in form.controls)
    for control in form.controls:
        _audit_control_text(form, control.id, control.label, "label", issues)
        _audit_control_text(form, control.id, control.help_text, "help_text", issues)
        if not CONTROL_ID_RE.match(control.id):
            issues.append(
                AuditIssue(
                    severity="error",
                    category="control_id_format",
                    message=f"Control id {control.id!r} does not match cNN format.",
                    form_id=form.form_id,
                    control_id=control.id,
                )
            )
        if control.type not in ALLOWED_CONTROL_TYPES:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="control_type",
                    message=f"Control type {control.type!r} is not allowed.",
                    form_id=form.form_id,
                    control_id=control.id,
                )
            )
        normalized = _normalized_label(control.label)
        if label_counts[normalized] > 1:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="duplicate_control_label",
                    message=f"Control label {control.label!r} appears more than once in the form.",
                    form_id=form.form_id,
                    control_id=control.id,
                )
            )
        expected_type = expected_label_types.get(control.label)
        if scenario is not None and expected_type is None:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="off_topic_control",
                    message="Control label is not present in the form's scenario template.",
                    form_id=form.form_id,
                    control_id=control.id,
                    details={"label": control.label, "scenario_id": scenario.id},
                )
            )
        elif expected_type is not None and expected_type != control.type:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="control_type_mismatch",
                    message=(
                        f"Control label {control.label!r} has type {control.type!r}; "
                        f"expected {expected_type!r}."
                    ),
                    form_id=form.form_id,
                    control_id=control.id,
                    details={"label": control.label},
                )
            )

    if scenario is not None:
        control_count = len(form.controls)
        if control_count < scenario.min_fields or control_count > scenario.max_fields:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="control_count",
                    message=(
                        f"Form has {control_count} controls; scenario allows "
                        f"{scenario.min_fields}-{scenario.max_fields}."
                    ),
                    form_id=form.form_id,
                )
            )
        labels = {control.label for control in form.controls}
        required_labels = {control.label for control in scenario.required}
        missing_required = sorted(required_labels.difference(labels))
        if missing_required:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="missing_required_control",
                    message="Form is missing required scenario controls.",
                    form_id=form.form_id,
                    details={"missing_labels": missing_required},
                )
            )


def _audit_control_text(
    form: FormSpec,
    control_id: str,
    text: str,
    field_name: str,
    issues: list[AuditIssue],
) -> None:
    if text != text.strip():
        issues.append(
            AuditIssue(
                severity="error",
                category="control_text_whitespace",
                message=f"Control {field_name} has leading or trailing whitespace.",
                form_id=form.form_id,
                control_id=control_id,
            )
        )
    if "\n" in text or "\r" in text or "\t" in text:
        issues.append(
            AuditIssue(
                severity="error",
                category="control_text_whitespace",
                message=f"Control {field_name} contains newline or tab whitespace.",
                form_id=form.form_id,
                control_id=control_id,
            )
        )
    if field_name == "label" and not text:
        issues.append(
            AuditIssue(
                severity="error",
                category="empty_label",
                message="Control label must not be empty.",
                form_id=form.form_id,
                control_id=control_id,
            )
        )
    if not ASCII_TEXT_RE.match(text):
        issues.append(
            AuditIssue(
                severity="error",
                category="language_ascii",
                message=(
                    f"Control {field_name} contains non-ASCII characters; "
                    "dataset labels/help text should be English-only ASCII."
                ),
                form_id=form.form_id,
                control_id=control_id,
                details={"text": text},
            )
        )


def _audit_order_constraints(form: FormSpec, issues: list[AuditIssue]) -> None:
    control_ids = {control.id for control in form.controls}
    for constraint in form.order_constraints:
        if constraint.before not in control_ids or constraint.after not in control_ids:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="order_constraint_reference",
                    message="Order constraint references a missing control id.",
                    form_id=form.form_id,
                    details={
                        "before": constraint.before,
                        "after": constraint.after,
                    },
                )
            )
        if constraint.before == constraint.after:
            issues.append(
                AuditIssue(
                    severity="error",
                    category="order_constraint_self_reference",
                    message="Order constraint points a control to itself.",
                    form_id=form.form_id,
                    details={
                        "before": constraint.before,
                        "after": constraint.after,
                    },
                )
            )


def _scenario_registry() -> dict[str, Any]:
    return {
        scenario.id: scenario
        for scenarios in _SCENARIOS.values()
        for scenario in scenarios
    }


def _label_type_registry() -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    for scenario_id, scenario in _scenario_registry().items():
        expected: dict[str, str] = {}
        for control in [*scenario.required, *scenario.optional]:
            expected[control.label] = control.type
        registry[scenario_id] = expected
    return registry


def _scenario_id(form_id: str) -> str:
    parts = form_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else form_id


def _normalized_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _table(headers: list[str], rows: Any) -> str:
    materialized = list(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in materialized:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _similar_pairs_table(pairs: list[SimilarFormPair]) -> str:
    rows = [
        (
            pair.left_form_id,
            pair.right_form_id,
            f"{pair.jaccard:.3f}",
            f"{pair.sequence_ratio:.3f}",
            pair.shared_label_count,
        )
        for pair in pairs
    ]
    return _table(
        ["left_form", "right_form", "jaccard", "sequence_ratio", "shared_labels"],
        rows,
    )


def _issues_table(issues: list[AuditIssue]) -> str:
    rows = [
        (
            issue.severity,
            issue.category,
            issue.form_id or "",
            issue.control_id or "",
            issue.message,
        )
        for issue in issues
    ]
    return _table(["severity", "category", "form", "control", "message"], rows)
