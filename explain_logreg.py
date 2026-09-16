"""
explain_logreg.py — Sparse baseline explainability script.

Extracts and saves feature-importance explanations from the trained
TF-IDF + Logistic Regression model.

What it produces
----------------
results/logreg_features.json  (coefficient-based)
    {
      "global": {
        "top_positive": [{"word": ..., "weight": ...}, ...],
        "top_negative": [{"word": ..., "weight": ...}, ...]
      },
      "per_example": [...]   // only when --examples used
    }

results/logreg_lime.json  (LIME local explanations)
    [
      {
        "text": ...,
        "prediction": int,
        "prediction_label": "positive" | "negative",
        "lime_features": [{"word": ..., "weight": float}, ...]
      },
      ...
    ]

Usage
-----
# Global coefficient-based features only (Phase 3a):
python explain_logreg.py

# Global + per-example coefficient explanations:
python explain_logreg.py --examples data/eval_clean.csv --n-examples 5

# LIME local explanations (Phase 3b) — requires --examples:
python explain_logreg.py --lime --examples data/eval_clean.csv --n-examples 5

# All together:
python explain_logreg.py --examples data/eval_clean.csv --n-examples 5 --lime

Options
-------
--top-n          int   Number of top positive / negative global features (default: 20)
--examples       str   Path to a CSV with a 'text' column
--n-examples     int   How many rows from the CSV to explain (default: 5)
--top-n-example  int   Features per example for coefficient method (default: 10)
--lime           flag  Run LIME local explanations on the examples CSV
--top-n-lime     int   Number of LIME features per example (default: 10)
--lime-out       str   Output path for LIME JSON (default: results/logreg_lime.json)
--out            str   Output JSON path for coefficient explanations (default: results/logreg_features.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Logistic Regression feature explanations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top positive/negative global features to save",
    )
    parser.add_argument(
        "--examples", default=None,
        help="CSV file with a 'text' column; enables per-example explanations",
    )
    parser.add_argument(
        "--n-examples", type=int, default=5,
        help="Number of rows to explain from the examples CSV",
    )
    parser.add_argument(
        "--top-n-example", type=int, default=10,
        help="Number of top features to show per example",
    )
    parser.add_argument(
        "--lime", action="store_true",
        help="Run LIME local explanations (requires --examples)",
    )
    parser.add_argument(
        "--top-n-lime", type=int, default=10,
        help="Number of LIME features per example",
    )
    parser.add_argument(
        "--lime-out", default="results/logreg_lime.json",
        help="Output path for LIME explanations JSON",
    )
    parser.add_argument(
        "--out", default="results/logreg_features.json",
        help="Output path for the coefficient-based JSON file",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from src.tfidf_logreg_model import TfidfLogRegModel

    print("[explain_logreg] Loading TfidfLogRegModel...")
    model = TfidfLogRegModel()

    # ── Global feature importance ─────────────────────────────────────────────
    print(f"[explain_logreg] Extracting top {args.top_n} global features...")
    global_explanation = model.explain_global(n=args.top_n)

    print("\n  Top POSITIVE features (→ class 1 / positive sentiment):")
    for entry in global_explanation["top_positive"][:10]:
        print(f"    {entry['weight']:+.4f}  {entry['word']}")

    print("\n  Top NEGATIVE features (→ class 0 / negative sentiment):")
    for entry in global_explanation["top_negative"][:10]:
        print(f"    {entry['weight']:+.4f}  {entry['word']}")

    # ── Per-example explanations (optional) ───────────────────────────────────
    per_example: list = []
    if args.examples:
        csv_path = Path(args.examples)
        if not csv_path.exists():
            print(f"[explain_logreg] WARNING — examples file not found: {csv_path}; skipping.")
        else:
            df = pd.read_csv(csv_path)
            if "text" not in df.columns:
                print("[explain_logreg] WARNING — CSV has no 'text' column; skipping per-example.")
            else:
                texts = df["text"].astype(str).tolist()[: args.n_examples]
                print(f"\n[explain_logreg] Explaining {len(texts)} examples "
                      f"(top {args.top_n_example} features each)...")
                for i, text in enumerate(texts):
                    result = model.explain_example(text, n=args.top_n_example)
                    per_example.append(result)
                    label_str = "positive" if result["prediction"] == 1 else "negative"
                    print(f"\n  [{i}] Prediction: {label_str}")
                    print(f"       Text: {text[:80]}{'...' if len(text) > 80 else ''}")
                    for feat in result["top_features"][:5]:
                        print(f"         {feat['contribution']:+.4f}  {feat['word']}")

    # ── LIME local explanations (Phase 3b) ───────────────────────────────────
    if args.lime:
        if not args.examples:
            print("[explain_logreg] WARNING — --lime requires --examples; skipping LIME.")
        else:
            lime_csv = Path(args.examples)
            if not lime_csv.exists():
                print(f"[explain_logreg] WARNING — examples file not found: {lime_csv}; skipping LIME.")
            else:
                try:
                    from lime.lime_text import LimeTextExplainer  # type: ignore
                except ImportError:
                    print("[explain_logreg] ERROR — lime package not found. "
                          "Install it with: pip install lime")
                    lime_csv = None  # skip cleanly

                if lime_csv is not None:
                    df_lime = pd.read_csv(lime_csv)
                    if "text" not in df_lime.columns:
                        print("[explain_logreg] WARNING — CSV has no 'text' column; skipping LIME.")
                    else:
                        texts_lime = df_lime["text"].astype(str).tolist()[: args.n_examples]

                        def _predict_proba(batch: list) -> np.ndarray:
                            return model.pipeline.predict_proba(batch)

                        lime_explainer = LimeTextExplainer(
                            class_names=["negative", "positive"],
                        )

                        print(f"\n[explain_logreg] Running LIME on {len(texts_lime)} examples "
                              f"(top {args.top_n_lime} features each)...")

                        lime_results = []
                        for i, text in enumerate(texts_lime):
                            exp = lime_explainer.explain_instance(
                                text,
                                _predict_proba,
                                num_features=args.top_n_lime,
                                num_samples=500,
                            )
                            pred = model.predict(text)
                            label_str = "positive" if pred == 1 else "negative"
                            lime_features = [
                                {"word": word, "weight": round(float(weight), 4)}
                                for word, weight in exp.as_list()
                            ]
                            lime_results.append({
                                "text": text,
                                "prediction": pred,
                                "prediction_label": label_str,
                                "lime_features": lime_features,
                            })
                            print(f"  [{i}] {label_str}: {text[:70]}{'...' if len(text) > 70 else ''}")
                            for feat in lime_features[:5]:
                                print(f"        {feat['weight']:+.4f}  {feat['word']}")

                        lime_out_path = Path(args.lime_out)
                        lime_out_path.parent.mkdir(exist_ok=True)
                        lime_out_path.write_text(json.dumps(lime_results, indent=2))
                        print(f"\n[explain_logreg] LIME results saved → {lime_out_path}")

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    payload: dict = {"global": global_explanation}
    if per_example:
        payload["per_example"] = per_example

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[explain_logreg] Saved → {out_path}")


if __name__ == "__main__":
    main()
