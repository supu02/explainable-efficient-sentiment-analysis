"""
TfidfLogRegModel: TF-IDF + LogisticRegression sentiment classifier.

The canonical sparse baseline for this project is the shipped logistic
regression artifact. This module matches that baseline so the code, saved
artifact, report, and slides all describe the same model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.config import DATA_TRAIN_PATH, TFIDF_LOGREG_MODEL_PATH, RANDOM_SEED
except ImportError:
    from config import DATA_TRAIN_PATH, TFIDF_LOGREG_MODEL_PATH, RANDOM_SEED


TextLike = Union[str, Sequence[str]]


@dataclass
class TfidfLogRegConfig:
    max_features: int = 20000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 1.0
    max_iter: int = 1000
    random_state: int = RANDOM_SEED


class TfidfLogRegModel:
    def __init__(
        self,
        model_path: Optional[str] = TFIDF_LOGREG_MODEL_PATH,
        config: Optional[TfidfLogRegConfig] = None,
        load: bool = True,
    ):
        self.config = config or TfidfLogRegConfig()
        self.pipeline: Optional[Pipeline] = None

        if load:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Trained model not found at {model_path}. "
                    "Train it once locally by running `python -m src.tfidf_logreg_model`."
                )
            self.load(model_path)

    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=True,
            sublinear_tf=False,
            strip_accents=None,
            stop_words="english",
        )

        clf = LogisticRegression(
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )

        return Pipeline(
            steps=[
                ("tfidf", tfidf),
                ("logreg", clf),
            ]
        )

    def train(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))

    def predict(self, x: TextLike) -> Union[int, List[int]]:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded / trained (pipeline is None).")

        if isinstance(x, str):
            preds = self.pipeline.predict([x])
            return int(preds[0])

        preds = self.pipeline.predict(list(x))
        return [int(p) for p in preds]

    def predict_proba(self, x: TextLike) -> Union[np.ndarray, List[List[float]]]:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded / trained (pipeline is None).")

        if isinstance(x, str):
            return self.pipeline.predict_proba([x])[0]

        probs = self.pipeline.predict_proba(list(x))
        return probs.tolist()

    def confidence(self, text: str) -> float:
        probs = self.predict_proba(text)
        return float(max(probs))

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save: pipeline is None.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        self.pipeline = joblib.load(path)

    def explain_global(self, n: int = 20) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded / trained (pipeline is None).")
        tfidf = self.pipeline.steps[0][1]
        clf = self.pipeline.steps[-1][1]
        feature_names = tfidf.get_feature_names_out()
        coef = clf.coef_[0]

        top_pos_idx = np.argsort(coef)[::-1][:n]
        top_neg_idx = np.argsort(coef)[:n]

        return {
            "top_positive": [
                {"word": str(feature_names[i]), "weight": round(float(coef[i]), 4)}
                for i in top_pos_idx
            ],
            "top_negative": [
                {"word": str(feature_names[i]), "weight": round(float(coef[i]), 4)}
                for i in top_neg_idx
            ],
        }

    def explain_example(self, text: str, n: int = 10) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded / trained (pipeline is None).")
        tfidf = self.pipeline.steps[0][1]
        clf = self.pipeline.steps[-1][1]
        feature_names = tfidf.get_feature_names_out()
        coef = clf.coef_[0]

        vec = tfidf.transform([text]).toarray()[0]
        contributions = vec * coef
        present_idx = np.where(vec != 0)[0]
        if len(present_idx) == 0:
            top_idx = np.array([], dtype=int)
        else:
            order = np.argsort(np.abs(contributions[present_idx]))[::-1][:n]
            top_idx = present_idx[order]

        return {
            "text": text,
            "prediction": self.predict(text),
            "top_features": [
                {
                    "word": str(feature_names[i]),
                    "contribution": round(float(contributions[i]), 4),
                    "tfidf_value": round(float(vec[i]), 4),
                }
                for i in top_idx
            ],
        }

    @classmethod
    def train_from_csv(
        cls,
        csv_path: str = DATA_TRAIN_PATH,
        model_path: str = TFIDF_LOGREG_MODEL_PATH,
        text_col: str = "text",
        label_col: str = "label",
        config: Optional[TfidfLogRegConfig] = None,
        val_size: float = 0.2,
        random_state: int = RANDOM_SEED,
    ) -> "TfidfLogRegModel":
        df = pd.read_csv(csv_path)
        assert {text_col, label_col}.issubset(df.columns), (
            f"CSV must contain columns: {text_col}, {label_col}"
        )

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=val_size,
            random_state=random_state,
            stratify=labels,
        )

        model = cls(model_path=None, config=config, load=False)
        model.train(X_train, y_train)

        y_val_pred = model.pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"[TfidfLogRegModel] Validation accuracy: {acc:.4f}")
        print(classification_report(y_val, y_val_pred))

        model.save(model_path)
        print(f"[TfidfLogRegModel] Saved model to {model_path}")
        return model

    @classmethod
    def train_full(
        cls,
        csv_path: str = DATA_TRAIN_PATH,
        model_path: str = TFIDF_LOGREG_MODEL_PATH,
        text_col: str = "text",
        label_col: str = "label",
        config: Optional[TfidfLogRegConfig] = None,
    ) -> "TfidfLogRegModel":
        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        model = cls(model_path=None, config=config, load=False)
        model.train(texts, labels)
        model.save(model_path)
        print(f"[TfidfLogRegModel] Trained on full data, saved to {model_path}")
        return model


if __name__ == "__main__":
    TfidfLogRegModel.train_from_csv()
