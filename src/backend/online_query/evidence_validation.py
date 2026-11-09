"""Bounded checks on the evidence actually prepared for a video answer."""
from __future__ import annotations

import json
from typing import Any

from online_query_constraints import constraint_window, constraint_windows, reserve_time_coverage, actor_fit, overlaps
from online_evidence_coverage import coverage_features, GROUNDING_INSTRUCTION


def validate_evidence(question: str, events: list[dict[str, Any]], labels: list[dict[str, Any]], constraints: dict[str, Any] | None = None, *, source_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    constraints = constraints or {}
    window = constraint_window(constraints)
    requirements = constraints.get("requirements") or {}
    issues = []
    if not events and not labels:
        issues.append("no_retrieved_evidence")
    sources = {str(item.get("source_memory") or item.get("source") or "") for item in events + labels}
    if constraints.get("requires_current") and "M_cur" not in sources:
        issues.append("required_current_evidence_missing")
    if constraints.get("requires_history") and not sources.intersection({"M_st", "M_lt", "M_cache"}):
        issues.append("required_history_evidence_missing")
    tail = constraints.get("current_tail_range")
    if tail and not any(row.get("capture_time_seconds") is not None and tail[0] < row["capture_time_seconds"] <= tail[1] for row in labels):
        issues.append("current_tail_has_no_direct_image")
    windows = constraint_windows(constraints)
    uncovered_requested = [span for span in windows if not any(row.get("capture_time_seconds") is not None and overlaps(row["capture_time_seconds"], row["capture_time_seconds"], span) for row in labels)]
    if any(span[0] == span[1] for span in uncovered_requested):
        issues.append("requested_instant_has_no_direct_image")
    if uncovered_requested:
        issues.append("requested_time_window_has_no_direct_image")
    if window and any(row.get("capture_time_seconds") is not None and not overlaps(row["capture_time_seconds"], row["capture_time_seconds"], window) for row in labels):
        issues.append("sent_image_outside_requested_time")
    if events and all(actor_fit(question, json.dumps(event, ensure_ascii=False, default=str)) == 0 for event in events):
        issues.append("requested_actor_has_no_matching_evidence")
    if requirements.get("earliest"):
        issues.append("earliest_occurrence_requires_checking_earlier_history")
    if requirements.get("simultaneous"):
        issues.append("verify_conditions_on_same_actor_in_same_image_or_observation")
    if requirements.get("comparison") and len(events) < 2:
        issues.append("comparison_may_lack_one_side")
    if requirements.get("comparison") and len({row.get("capture_time_seconds") for row in labels
            if row.get("capture_time_seconds") is not None}) < 2:
        issues.append("comparison_has_fewer_than_two_observed_image_times")
    reference_ranges = constraints.get("referenced_time_ranges") or []
    uncovered = [span for span in reference_ranges if not any(row.get("capture_time_seconds") is not None and overlaps(row["capture_time_seconds"], row["capture_time_seconds"], [span["start"], span.get("end", span["start"])]) for row in labels)]
    if uncovered:
        issues.append("previous_occurrence_has_no_direct_image")
    missing_features = []
    if constraints.get("summary") and source_events is not None:
        expected = set().union(*(coverage_features(question, event) for event in source_events)) if source_events else set()
        retained = set().union(*(coverage_features(question, event) for event in events)) if events else set()
        missing_features = sorted(feature for feature in expected - retained if feature.startswith(("state:", "appearance:")))
        if missing_features:
            issues.append("summary_state_changes_lost_during_prompt_compression")
    # Passing bookkeeping checks cannot establish semantic answer sufficiency.
    # Keep the legacy field nullable for clients, without asserting True.
    return {"sufficient": False if issues else None, "mechanical_checks_passed": not issues,
            "assessment_scope": "Time/source checks only; semantic sufficiency is not established.",
            "issues": issues, "requested_time_range": constraints.get("time_range"), "requested_time_ranges": windows, "uncovered_requested_ranges": uncovered_requested,
            "referenced_time_ranges": reference_ranges, "uncovered_reference_ranges": uncovered,
            "state_features_lost_during_compression": missing_features,
            "sent_image_times": [row.get("capture_time_seconds") for row in labels],
            "instruction": "Use only direct evidence within the requested time. An unverified requirement is not a negative fact. For first/earliest say first observed in the supplied evidence unless earlier history was checked. Verify simultaneous conditions on the same actor and instant. If the requested instant, previous occurrence, or comparison side has no evidence, state that gap. " + GROUNDING_INSTRUCTION}
