"""Predict sentiment for one review using the retained project artifacts."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Movie review to classify")
    parser.add_argument("--model", choices=["logreg", "tinybert", "cascade"], default="cascade")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must contain a non-empty review")

    try:
        if args.model == "logreg":
            from src.tfidf_logreg_model import TfidfLogRegModel

            model = TfidfLogRegModel()
            prediction = model.predict(args.text)
            print(f"Maximum class probability: {model.confidence(args.text):.4f}")
        elif args.model == "tinybert":
            from src.tinybert_model import TinyBertModel

            prediction = TinyBertModel().predict(args.text)
        else:
            from src.cascade_classifier import CascadeClassifier

            model = CascadeClassifier()
            probability = model.logreg_confidence(args.text)
            route = "Logistic Regression" if probability >= model.PROB_THRESHOLD else "TinyBERT INT8"
            prediction = model.predict(args.text)
            print(f"Logistic Regression maximum probability: {probability:.4f}")
            print(f"Route: {route} (threshold {model.PROB_THRESHOLD:.2f})")
    except FileNotFoundError as exc:
        parser.exit(1, f"{exc}\nRestore the supplied model archive; see models/README.md.\n")

    print(f"Sentiment: {'positive' if prediction == 1 else 'negative'} (label {prediction})")


if __name__ == "__main__":
    main()
