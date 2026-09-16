"""
src — active models for the XAI sentiment analysis project.

Exports the active model classes so they can be imported
directly from the package:
    from src import TfidfLogRegModel, TinyBertModel, CascadeClassifier

Individual submodule imports still work as before:
    from src.tfidf_logreg_model import TfidfLogRegModel
"""

# Import lightweight classes first so they are in sys.modules before
# CascadeClassifier (which imports them) is loaded.
from .tfidf_logreg_model import TfidfLogRegModel, TfidfLogRegConfig
from .tinybert_model import TinyBertModel, TinyBertConfig
from .cascade_classifier import CascadeClassifier

__all__ = [
    "TfidfLogRegModel",
    "TfidfLogRegConfig",
    "TinyBertModel",
    "TinyBertConfig",
    "CascadeClassifier",
]
