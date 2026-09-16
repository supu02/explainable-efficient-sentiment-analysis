# GitHub portfolio setup

## Repository name

`explainable-efficient-sentiment-analysis`

## About description

```text
Explainable IMDb sentiment analysis with Logistic Regression, INT8 TinyBERT, confidence-based routing, LIME, and reproducible CPU evaluation.
```

## Topics

```text
python machine-learning nlp sentiment-analysis explainable-ai lime tinybert transformers scikit-learn pytorch quantization model-efficiency imdb
```

## Upload the portfolio

Create an empty GitHub repository with the name above. Extract `explainable-efficient-sentiment-analysis.zip` and upload the contents of its repository folder. The README, report, figures, source code, and saved results belong at the repository root as packaged.

To use Git from a terminal, run the following **inside this portfolio directory**. The original workspace inherits a different home-directory Git repository, so initialize this directory before adding files:

```bash
git init -b main
git add .
git commit -m "Add explainable sentiment analysis portfolio"
```

Then follow GitHub's commands for adding the remote of the repository you created and pushing `main`. Do not reuse the unrelated remote inherited from the original workspace.

## Attach the trained models

The source archive is lightweight. `xai-sentiment-models.zip` contains the retained model weights and tokenizer, and can be attached as an asset to a GitHub Release. Once an actual Release exists, add its model download link to the README and `models/README.md`. The current documentation does not claim that such a release already exists.

The model ZIP extracts into `models/`. The `.gitignore` keeps large serialized weights, local training data, environments, and rerun outputs out of ordinary source commits.

The optional `xai-sentiment-training-data.zip` restores the exact original training CSV at `data/train.csv` for cleaning or retraining. Keep it as a separate artifact; data provenance and source attribution are documented in `data/README.md`.

## Feature it on your profile

Pin the repository. Use this short project description in your profile README:

> **Explainable & Efficient Sentiment Analysis** — Compared sparse and quantized transformer models for IMDb sentiment classification; investigated confidence-based routing and interpreted predictions with LIME and feature weights.

If mentioning the numerical result, keep its scope:

> The cascade reduced measured CPU latency by 47.5% relative to TinyBERT INT8 on a cleaned 39-review evaluation sample.

For an interview, explain why the linear baseline was strongest on this sample, how training/evaluation text overlap was removed, and why a larger independent evaluation is the next step.
