"""
evaluate.py — Unified evaluation pipeline for the final project setup.

Evaluates the canonical sparse baseline, TinyBERT, and the cascade on the
clean evaluation CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import DATA_EVAL_PATH, RANDOM_SEED

_RESULTS_DIR = Path(__file__).parent / "results" / "reruns"
_DEFAULT_TEST_PATH = DATA_EVAL_PATH
_TIMING_REPEATS = 3


def load_eval_data(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Evaluation CSV must contain columns {required}; found {set(df.columns)}")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    if not texts:
        raise ValueError("Evaluation CSV must contain at least one review.")
    if df["text"].isna().any() or df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Evaluation reviews must be non-empty strings.")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Evaluation labels must be 0 or 1.")
    return texts, labels


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, average="binary", zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, average="binary", zero_division=0)), 4),
        "n_samples": len(labels),
        "n_correct": int(sum(p == l for p, l in zip(preds, labels))),
    }


def _predict_and_time(
    predict_one: Callable[[str], int],
    texts: List[str],
    repeats: int = _TIMING_REPEATS,
) -> Tuple[List[int], Dict[str, float]]:
    if not texts:
        return [], {"timing_repeats": repeats, "inference_time_total_s": 0.0, "inference_time_per_sample_ms": 0.0}

    predict_one(texts[0])

    timings: List[float] = []
    reference_preds: List[int] | None = None
    for _ in range(repeats):
        run_preds: List[int] = []
        t0 = time.perf_counter()
        for text in texts:
            run_preds.append(int(predict_one(text)))
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        if reference_preds is None:
            reference_preds = run_preds
        elif run_preds != reference_preds:
            raise RuntimeError("Predictions changed across timing repeats.")

    mean_total = sum(timings) / len(timings)
    return reference_preds or [], {
        "timing_repeats": repeats,
        "inference_time_total_s": round(mean_total, 4),
        "inference_time_per_sample_ms": round(mean_total / len(texts) * 1000, 4),
    }


def evaluate_logreg(texts: List[str], labels: List[int]) -> Dict[str, Any]:
    from src.tfidf_logreg_model import TfidfLogRegModel

    print("[evaluate] Loading TfidfLogRegModel...")
    model = TfidfLogRegModel()
    preds, timing = _predict_and_time(model.predict, texts)
    metrics = compute_metrics(labels, preds)
    metrics.update(timing)
    return metrics


def evaluate_tinybert(texts: List[str], labels: List[int]) -> Dict[str, Any]:
    from src.tinybert_model import TinyBertModel

    print("[evaluate] Loading TinyBertModel...")
    model = TinyBertModel()
    preds, timing = _predict_and_time(model.predict, texts)
    metrics = compute_metrics(labels, preds)
    metrics.update(timing)
    return metrics


def evaluate_cascade(texts: List[str], labels: List[int]) -> Dict[str, Any]:
    from src.cascade_classifier import CascadeClassifier

    print("[evaluate] Loading CascadeClassifier...")
    model = CascadeClassifier()

    route_counts = {"logreg": 0, "tinybert": 0}

    def _predict(text: str) -> int:
        if model.logreg_confidence(text) >= model.PROB_THRESHOLD:
            route_counts["logreg"] += 1
            return model.logreg_model.predict(text)
        route_counts["tinybert"] += 1
        return model.bert_model.predict(text)

    preds, timing = _predict_and_time(_predict, texts)
    warmup_route = "logreg" if model.logreg_confidence(texts[0]) >= model.PROB_THRESHOLD else "tinybert"
    route_counts[warmup_route] -= 1
    metrics = compute_metrics(labels, preds)
    metrics.update(timing)
    metrics["routing"] = {
        "logreg_gate": route_counts["logreg"] // _TIMING_REPEATS,
        "tinybert_gate": route_counts["tinybert"] // _TIMING_REPEATS,
        "logreg_pct": round((route_counts["logreg"] / _TIMING_REPEATS) / len(texts) * 100, 1),
        "tinybert_pct": round((route_counts["tinybert"] / _TIMING_REPEATS) / len(texts) * 100, 1),
        "threshold": model.PROB_THRESHOLD,
    }
    return metrics


def save_results(results: Dict[str, Any], output_dir: Path = _RESULTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"evaluation_clean_{ts}.json"
    latest = output_dir / "evaluation_clean_latest.json"

    payload = json.dumps(results, indent=2)
    out_path.write_text(payload)
    latest.write_text(payload)
    return out_path


def print_summary(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print(f"EVALUATION SUMMARY — {results['timestamp']}")
    print(f"Eval file  : {results['eval_file']}  ({results['n_samples']} samples)")
    print("=" * 64)

    for model_name, metrics in results["models"].items():
        print(f"\n  {model_name.upper()}")
        if "error" in metrics:
            print(f"    ✗ SKIPPED: {metrics['error']}")
            continue
        print(f"    Accuracy  : {metrics['accuracy']:.4f}")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1        : {metrics['f1']:.4f}")
        print(f"    Time/sample: {metrics['inference_time_per_sample_ms']:.2f} ms")
        if "routing" in metrics:
            r = metrics["routing"]
            print(
                f"    Routing   : LogReg {r['logreg_pct']}% ({r['logreg_gate']} samples) "
                f"| TinyBERT {r['tinybert_pct']}% ({r['tinybert_gate']} samples) "
                f"| threshold={r['threshold']:.2f}"
            )

    print("\n" + "=" * 64)


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Evaluate the final project models on the clean evaluation split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test",
        default=_DEFAULT_TEST_PATH,
        help="Path to labelled evaluation CSV (columns: text, label)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "tinybert", "cascade"],
        default=["logreg", "tinybert", "cascade"],
        help="Which models to include in this evaluation run",
    )
    parser.add_argument("--out-dir", type=Path, default=_RESULTS_DIR,
                        help="Directory for new results; defaults to results/reruns")
    args = parser.parse_args()

    print(f"[evaluate] Eval file    : {args.test}")
    print(f"[evaluate] Models       : {args.models}")
    print(f"[evaluate] Random seed  : {RANDOM_SEED}")
    print(f"[evaluate] Timing setup : {_TIMING_REPEATS} repeated per-sample passes per model")

    texts, labels = load_eval_data(args.test)
    print(f"[evaluate] Loaded {len(texts)} samples  (pos={sum(labels)}  neg={len(labels) - sum(labels)})")

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "eval_file": str(args.test),
        "n_samples": len(texts),
        "timing_setup": {
            "mode": "per-sample loop",
            "repeats": _TIMING_REPEATS,
        },
        "models": {},
    }

    evaluators = {
        "logreg": evaluate_logreg,
        "tinybert": evaluate_tinybert,
        "cascade": evaluate_cascade,
    }

    for model_key in args.models:
        try:
            results["models"][model_key] = evaluators[model_key](texts, labels)
            print(
                f"[evaluate] {model_key}: done  "
                f"(F1={results['models'][model_key]['f1']:.4f}  "
                f"{results['models'][model_key]['inference_time_per_sample_ms']:.2f} ms/sample)"
            )
        except FileNotFoundError as exc:
            print(f"[evaluate] WARNING — {model_key} skipped: {exc}")
            results["models"][model_key] = {"error": str(exc)}
        except Exception as exc:
            print(f"[evaluate] ERROR   — {model_key} failed: {exc}")
            results["models"][model_key] = {"error": str(exc)}

    out_path = save_results(results, args.out_dir)
    print(f"\n[evaluate] Results saved → {out_path}")
    print(f"[evaluate] Latest       → {args.out_dir / 'evaluation_clean_latest.json'}")

    print_summary(results)
    return results


if __name__ == "__main__":
    final_results = main()
    if any("error" in metrics for metrics in final_results["models"].values()):
        raise SystemExit(1)
