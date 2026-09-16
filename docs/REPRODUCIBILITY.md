# Reproducing the project

## What can be reproduced

| Workflow | Inputs | Scope |
|---|---|---|
| Inspect results / rebuild figures | Source archive, plotting dependencies | Recreates presentation of the recorded experiment |
| Run inference and evaluation | Source archive + model archive | Executes the retained models on the included 39-review sample |
| Rebuild sample cleaning | Original `data/train.csv` + candidate CSV | Repeats the exact-text exclusion operation |
| Train fresh models | Training CSV, training dependencies, network for pretrained TinyBERT | Produces a new experiment; historical weights/results are not guaranteed |

The headline results come from `results/evaluation_clean_latest.json`, timestamped 2026-04-16. The portfolio copy replaces its machine-specific evaluation path with `data/eval_clean.csv`; numerical results are unchanged.

## Environment

The packaging checks use Python 3.12.0. `requirements-lock.txt` records the installed packages in the existing project environment at packaging time. It is an environment snapshot, not a claim that all packages are necessary or that every operating system has been tested. `requirements.txt` retains the original broad dependency ranges.

The original saved latency run records CPU use and three repeated passes but not the CPU model, operating system, thread counts, or a complete environment snapshot. Local packaging verification cannot retroactively fill those gaps.

## Inference protocol

1. Load each model before timing.
2. Warm it up with the first evaluation review.
3. Predict each review in order, using the same per-sample loop.
4. Repeat the loop three times; require predictions to agree across passes.
5. Average the three elapsed times and divide by 39 to obtain ms/review.

The cascade timing includes its actual probability gate and prediction work. Model loading and explanations are outside the timed loop. F1, precision, and recall are binary scores for label `1`, not macro averages. Scores are rounded to four decimals.

```bash
python evaluate.py --models logreg tinybert cascade --out-dir results/reruns
```

New results are written under `results/reruns/` so the reference experiment remains available. Use `--test` to evaluate a different labeled CSV. That is a new evaluation, not a reproduction of the 39-review experiment.

## Explainability

```bash
python explain_logreg.py --examples data/eval_clean.csv --n-examples 5 --lime --out results/reruns/logreg_features.json --lime-out results/reruns/logreg_lime.json
python explain_tinybert.py --examples data/eval_clean.csv --n-examples 5 --method lime --out results/reruns/tinybert_lime.json
```

The commands explain the first five rows in their saved order. The original LIME workflow does not fix a perturbation seed, so exact coefficients can vary. The figure generator uses the saved reference explanations, not new rerun files.

Integrated Gradients is an unfinished path in the inherited code. The available checkpoint is quantized; no compatible fine-tuned float checkpoint is included. Use LIME for the completed workflow. A future IG implementation must retain the float model before quantization and use matching preprocessing.

## Training new models

Restore a CSV with `text` and binary `label` fields at `data/train.csv`. The baseline training helper makes a stratified 80/20 train/validation split with seed 42:

```bash
python -m src.tfidf_logreg_model
```

For TinyBERT, install the optional training dependency and run its training entry point:

```bash
python -m pip install -r requirements-training.txt
python src/tinybert_model.py
```

TinyBERT training downloads its pretrained starting model and requires substantially more compute than inference. These training commands are provided from the existing implementation; retraining was not performed during portfolio packaging. Fresh training overwrites the model paths, so retain the supplied artifacts separately if you need them again.

The preserved default architecture uses four transformer layers, 128-token inputs with 63 head and 63 tail tokens plus two special tokens, dynamic padding, and post-training dynamic INT8 quantization of linear layers. The code exposes training settings, but a complete original training log is not available.

## Figures and report

```bash
python scripts/generate_report_figures.py
cd docs/report-source
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
cp report.pdf ../report.pdf
```

Figures are exported as PNG for GitHub and PDF for the report, using the checked-in result JSON. The report copy corrects stale sample/threshold captions, local-example descriptions, and overly broad explanation claims from the supplied draft. The original workspace report remains available separately.

## Verification

`python scripts/verify_portfolio.py` checks Python syntax, local Markdown links, CSV identities and class balance, numeric consistency of recorded metrics, routing totals, explanation/example alignment, and available model hashes. It requires only the Python standard library, so the GitHub workflow can run without model weights.

The workflow checks the repository and recorded evidence. It does not run model training or certify model quality. Local model execution is documented separately in [Verification](VERIFICATION.md).
