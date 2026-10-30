"""Baseline evaluation runner for un-fine-tuned models (Zero-Shot and Few-Shot)."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from rich.console import Console
from rich.table import Table

from src.schema import TriageOutput, validate_triage_json

console = Console()


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of records."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_predictions(
    ground_truths: List[Dict[str, Any]],
    predictions: List[str],
    latencies: List[float],
) -> Dict[str, Any]:
    """Computes JSON validity, field-level exact match, and latency statistics."""
    total = len(ground_truths)
    if total == 0:
        return {}

    valid_json_count = 0
    intent_match = 0
    sentiment_match = 0
    urgency_match = 0
    route_match = 0
    all_fields_exact_match = 0

    for gt_record, pred_text in zip(ground_truths, predictions):
        # Extract ground truth target from ChatML assistant message
        gt_assistant_content = gt_record["messages"][-1]["content"]
        gt_target = json.loads(gt_assistant_content)

        is_valid, parsed_pred, _ = validate_triage_json(pred_text)
        if not is_valid or parsed_pred is None:
            continue

        valid_json_count += 1
        pred_dict = parsed_pred.model_dump()

        curr_intent = pred_dict.get("intent") == gt_target.get("intent")
        curr_sentiment = pred_dict.get("sentiment") == gt_target.get("sentiment")
        curr_urgency = pred_dict.get("urgency") == gt_target.get("urgency")
        curr_route = pred_dict.get("route_to") == gt_target.get("route_to")

        if curr_intent:
            intent_match += 1
        if curr_sentiment:
            sentiment_match += 1
        if curr_urgency:
            urgency_match += 1
        if curr_route:
            route_match += 1

        if curr_intent and curr_sentiment and curr_urgency and curr_route:
            all_fields_exact_match += 1

    sorted_latencies = sorted(latencies) if latencies else [0.0]
    p50_latency = sorted_latencies[len(sorted_latencies) // 2] * 1000  # ms
    p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] * 1000  # ms

    metrics = {
        "total_samples": total,
        "valid_json_pct": (valid_json_count / total) * 100,
        "intent_accuracy_pct": (intent_match / total) * 100,
        "sentiment_accuracy_pct": (sentiment_match / total) * 100,
        "urgency_accuracy_pct": (urgency_match / total) * 100,
        "route_accuracy_pct": (route_match / total) * 100,
        "exact_match_all_pct": (all_fields_exact_match / total) * 100,
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
    }

    return metrics


def print_evaluation_table(title: str, metrics: Dict[str, Any]) -> None:
    """Pretty prints evaluation metrics to console using Rich."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")

    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), f"{v:.2f}%" if "pct" in k else str(v))

    console.print(table)


if __name__ == "__main__":
    console.print("[bold green]TriageLM Baseline Evaluator Initialized.[/bold green]")
