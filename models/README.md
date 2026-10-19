# Model artifacts

This GitHub repository contains the source code and a small architecture configuration, but **not the trained model weights or tokenizer files**. The original project handoff included a separate **`xai-sentiment-models.zip`** archive. There is currently no configured public download URL for that archive.

Extract the model ZIP into the repository root. It restores:

```text
models/
├── tfidf_logreg.joblib
├── tinybert_quantized.pt
├── architecture/config.json
└── tokenizer/
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

[artifact_manifest.json](artifact_manifest.json) records file sizes and SHA-256 hashes. The optional `tinybert_config.joblib` is not part of the original artifact set; the loader uses the project defaults when it is absent.

TinyBERT's retained checkpoint stores quantized weights. This portfolio includes the matching architecture configuration so loading it does not require a populated Hugging Face cache or a network download. The saved tokenizer is required too. Dynamic INT8 inference runs on CPU.

The checkpoint and scikit-learn pipeline use Python-based serialization. Load the supplied trusted project files; do not substitute untrusted pickled artifacts. Loading across library versions can fail, so use the checked environment pins in `requirements-lock.txt` when reproducing the saved artifacts.

For training instructions and reproducibility limits, see [Reproducibility](../docs/REPRODUCIBILITY.md).
