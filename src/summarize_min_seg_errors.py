from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _find_latest_error_report(model_dir: str) -> Path:
    path = Path(model_dir)
    if path.is_file():
        return path
    candidates = sorted(path.glob("error_analysis_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No error analysis files found under: {path}")
    return candidates[0]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _aggregate_error_stats(report: Dict[str, Any]) -> Dict[str, Any]:
    global_counts = report.get("global_counts_at_threshold") or {}
    global_fp = global_counts.get("fp")
    global_fn = global_counts.get("fn")

    error_docs = report.get("error_docs", [])
    total_docs = len(error_docs)
    total_fp = sum(int(doc.get("num_fp", 0)) for doc in error_docs)
    total_fn = sum(int(doc.get("num_fn", 0)) for doc in error_docs)

    fp_probs: List[float] = []
    fn_probs: List[float] = []
    for doc in error_docs:
        for example in doc.get("examples", []):
            prob = _safe_float(example.get("boundary_prob", 0.0), 0.0)
            if example.get("error_type") == "FP":
                fp_probs.append(prob)
            elif example.get("error_type") == "FN":
                fn_probs.append(prob)

    return {
        "num_error_docs": total_docs,
        "total_fp": int(global_fp) if global_fp is not None else total_fp,
        "total_fn": int(global_fn) if global_fn is not None else total_fn,
        "avg_fp_prob": mean(fp_probs) if fp_probs else None,
        "avg_fn_prob": mean(fn_probs) if fn_probs else None,
    }


def _threshold_recommendation(threshold: float, total_fp: int, total_fn: int) -> Tuple[str, float, str]:
    """
    Returns (direction, suggested_threshold, rationale).
    """
    threshold = max(0.0, min(1.0, float(threshold)))
    if total_fp == 0 and total_fn == 0:
        return "keep", threshold, "No FP/FN errors were recorded."

    if total_fn >= total_fp * 1.25:
        step = 0.05
        suggested = max(0.05, threshold - step)
        return "lower", suggested, "FN dominates FP, so recall likely needs help."

    if total_fp >= total_fn * 1.25:
        step = 0.05
        suggested = min(0.95, threshold + step)
        return "raise", suggested, "FP dominates FN, so precision likely needs help."

    return "keep", threshold, "FP and FN are balanced; threshold is likely not the main bottleneck."


def _metric_grade(f1: float, pk: float, wd: float) -> str:
    if f1 >= 0.62 and pk <= 0.24 and wd <= 0.28:
        return "good"
    if f1 >= 0.52 and pk <= 0.30 and wd <= 0.34:
        return "usable"
    return "needs_work"


def _next_tuning_actions(direction: str, f1: float, grade: str) -> List[str]:
    actions: List[str] = []
    if direction == "lower":
        actions.append("Decrease threshold by 0.03-0.07 and re-check FP/FN balance.")
        actions.append("Increase `max_sentences` if memory allows to improve long-range boundary recall.")
    elif direction == "raise":
        actions.append("Increase threshold by 0.03-0.07 and re-check FP/FN balance.")
        actions.append("Increase dropout slightly (e.g., +0.03) if FP spikes persist.")
    else:
        actions.append("Keep threshold fixed and focus on model capacity/data handling.")

    if grade == "needs_work":
        actions.append("Run one tuning pass with `context_layers=2` and lower `lr_encoder` to stabilize boundaries.")
        actions.append("Inspect top 20 FN contexts; if many are subtle shifts, increase training epochs by 2-4.")
    elif grade == "usable":
        actions.append("Start pipeline integration (`segment -> topic summary -> global summary`) and iterate with error analysis.")
    else:
        actions.append("Proceed to topic-aware summarization stage; keep periodic segmentation checks.")

    if f1 < 0.50:
        actions.append("Prioritize segmentation improvements before heavy summarization tuning.")

    return actions


def _print_compact_summary(summary: Dict[str, Any]) -> None:
    print("MiniSeg Error Analysis Summary")
    print(
        f"- Metrics: f1={summary['metrics']['f1']:.4f} "
        f"relaxed_f1={summary['metrics']['relaxed_f1']:.4f} "
        f"pk={summary['metrics']['pk']:.4f} wd={summary['metrics']['window_diff']:.4f} "
        f"(grade={summary['grade']})"
    )
    print(
        f"- Errors: FP={summary['errors']['total_fp']} FN={summary['errors']['total_fn']} "
        f"across {summary['errors']['num_error_docs']} docs"
    )
    print(
        f"- Threshold: current={summary['threshold']['current']:.2f} "
        f"suggestion={summary['threshold']['suggested']:.2f} ({summary['threshold']['direction']})"
    )
    print(f"- Why: {summary['threshold']['rationale']}")
    print("- Next tuning actions:")
    for action in summary["next_actions"]:
        print(f"  - {action}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Summarize MiniSeg error analysis and suggest next tuning steps.")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to error analysis JSON. If omitted, uses latest error_analysis_*.json in model-dir.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(repo_root / "models" / "miniseg"),
        help="Directory containing error analysis JSON files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path for compact summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report) if args.report else _find_latest_error_report(args.model_dir)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("validation_metrics_at_threshold", {})
    f1 = _safe_float(metrics.get("f1", 0.0))
    relaxed_f1 = _safe_float(metrics.get("relaxed_f1", 0.0))
    pk = _safe_float(metrics.get("pk", 1.0))
    wd = _safe_float(metrics.get("window_diff", 1.0))
    grade = _metric_grade(f1, pk, wd)

    threshold_current = _safe_float(report.get("threshold_used", 0.5), 0.5)
    errors = _aggregate_error_stats(report)
    direction, suggested_threshold, rationale = _threshold_recommendation(
        threshold_current,
        errors["total_fp"],
        errors["total_fn"],
    )
    actions = _next_tuning_actions(direction, f1, grade)

    summary = {
        "source_report": str(report_path.resolve()),
        "timestamp": report.get("timestamp"),
        "metrics": {
            "f1": f1,
            "relaxed_f1": relaxed_f1,
            "pk": pk,
            "window_diff": wd,
        },
        "grade": grade,
        "errors": errors,
        "threshold": {
            "current": threshold_current,
            "direction": direction,
            "suggested": suggested_threshold,
            "rationale": rationale,
        },
        "next_actions": actions,
    }

    _print_compact_summary(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved compact summary to: {out_path}")


if __name__ == "__main__":
    main()
