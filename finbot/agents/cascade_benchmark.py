"""Cascade benchmark recording.

Turns a single cascade-scenario run (see `run_cascade_orchestrator`) into a
flat CSV row and appends it to the benchmark file. Every run is recorded,
including runs whose detected cascade type differs from the scenario's
expectation and runs that crashed -- both are useful benchmark signal.

Benchmark loop:

    1. take a prepared invoice from `cascade_scenarios.json`
    2. run it through the real agent chain
    3. classify the resulting cascade (`classify_cascade`)
    4. append the run to the CSV (`append_benchmark_row`)
"""

from __future__ import annotations

import csv
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "cascade_runs.csv"

# Bump when `classify_cascade` or the step heuristics change, so rows produced
# by different classifier logic can be told apart.
CLASSIFIER_VERSION = "1"

CSV_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "scenario_id",
    "scenario_title",
    "catalog_version",
    "classifier_version",
    "llm_provider",
    "llm_model",
    "invoice_number",
    "invoice_amount",
    "invoice_description",
    "due_date_offset_days",
    "expected_cascade_type",
    "detected_cascade_type",
    "matches_expected",
    "detected_severity",
    "cascade_failures_detected",
    "task_status",
    "agent_path",
    "first_error_agent",
    "failed_agents",
    "reached_payments",
    "reached_final_agent",
    "total_errors",
    "initial_confidence",
    "final_confidence",
    "confidence_degradation",
    "agent_chain_json",
    "run_error",
)


def build_benchmark_row(
    scenario: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    invoice_number: str = "",
    cascade_types: dict[str, Any] | None = None,
    catalog_version: Any = "",
    llm_provider: str = "",
    llm_model: str = "",
    run_error: str = "",
    run_id: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Flatten a scenario plus its run result into one CSV row.

    `result` is the dict returned by `run_cascade_orchestrator`; pass None
    together with `run_error` when the run raised before producing a result.
    """
    result = result or {}
    analysis = result.get("cascade_analysis") or {}
    chain = result.get("agent_chain") or []
    invoice = scenario.get("invoice", {})

    # An empty chain means no agent ran (typically every LLM call failed);
    # classify_cascade would report that as "none", a false clean result.
    if not run_error and not chain:
        run_error = "no agent steps recorded (LLM calls likely failed)"

    detected = analysis.get("cascade_type", "") if not run_error else "error"
    expected = scenario.get("expected_cascade_type", "")
    severity = (cascade_types or {}).get(detected, {}).get("severity", "")
    first_error_agent = next(
        (s["agent"] for s in chain if not s.get("success") or s.get("errors")), ""
    )

    return {
        "run_id": run_id or f"run_{secrets.token_hex(6)}",
        "timestamp_utc": (timestamp or datetime.now(UTC)).isoformat(timespec="seconds"),
        "scenario_id": scenario.get("id", ""),
        "scenario_title": scenario.get("title", ""),
        "catalog_version": catalog_version,
        "classifier_version": CLASSIFIER_VERSION,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "invoice_number": invoice_number,
        "invoice_amount": invoice.get("amount", ""),
        "invoice_description": invoice.get("description", ""),
        "due_date_offset_days": invoice.get("due_date_offset_days", ""),
        "expected_cascade_type": expected,
        "detected_cascade_type": detected,
        "matches_expected": detected == expected,
        "detected_severity": severity,
        "cascade_failures_detected": analysis.get("cascade_failures_detected", ""),
        "task_status": result.get("task_status", ""),
        "agent_path": ">".join(s["agent"] for s in chain),
        "first_error_agent": first_error_agent,
        "failed_agents": "|".join(analysis.get("failed_agents", [])),
        "reached_payments": any(s["agent"] == "payments_agent" for s in chain),
        "reached_final_agent": analysis.get("reached_final_agent", ""),
        "total_errors": analysis.get("total_errors", ""),
        "initial_confidence": analysis.get("initial_confidence", ""),
        "final_confidence": analysis.get("final_confidence", ""),
        "confidence_degradation": analysis.get("confidence_degradation", ""),
        "agent_chain_json": json.dumps(chain, ensure_ascii=False),
        "run_error": run_error,
    }


def append_benchmark_row(row: dict[str, Any], path: Path | None = None) -> Path:
    """Append one row to the benchmark CSV, writing the header for a new file."""
    target = path or BENCHMARK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    is_new = not target.exists() or target.stat().st_size == 0
    with open(target, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return target


def load_benchmark_rows(path: Path | None = None) -> list[dict[str, str]]:
    """Read the benchmark CSV back as a list of string dicts."""
    target = path or BENCHMARK_PATH
    if not target.exists():
        return []
    with open(target, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
