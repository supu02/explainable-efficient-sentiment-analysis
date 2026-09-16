"""
CascadeClassifier: Logistic Regression → TinyBERT cascade for sentiment analysis.

Routes confident sparse-baseline predictions directly and sends the uncertain
cases to TinyBERT.
"""

from __future__ import annotations

try:
    from src.tfidf_logreg_model import TfidfLogRegModel
    from src.tinybert_model import TinyBertModel
except ImportError:
    from tfidf_logreg_model import TfidfLogRegModel
    from tinybert_model import TinyBertModel


class CascadeClassifier:
    """
    Two-stage cascade classifier.

    Stage 1 — TF-IDF + Logistic Regression:
      Fast and directly probabilistic, so the routing threshold has a clear
      interpretation.

    Stage 2 — Quantized TinyBERT:
      Used only when the sparse baseline is not confident enough.
      Delegates entirely to TinyBertModel.predict(), which handles tokenisation
      and head+tail truncation internally.
    """

    PROB_THRESHOLD = 0.80

    def __init__(self) -> None:
        print("[CascadeClassifier] Loading Logistic Regression + TinyBERT...")

        self.logreg_model = TfidfLogRegModel()
        self.bert_model = TinyBertModel()  # TINYBERT_MODEL_PATH + TINYBERT_TOKENIZER_PATH

        print("[CascadeClassifier] Ready ✔")

    def logreg_confidence(self, text: str) -> float:
        return self.logreg_model.confidence(text)

    def predict(self, text: str) -> int:
        """
        Cascade prediction.  Returns 0 (negative) or 1 (positive).

        Routing:
          - Logistic Regression confidence ≥ PROB_THRESHOLD → Logistic Regression
          - Otherwise                                       → TinyBERT
        """
        if self.logreg_confidence(text) >= self.PROB_THRESHOLD:
            return self.logreg_model.predict(text)

        return self.bert_model.predict(text)
