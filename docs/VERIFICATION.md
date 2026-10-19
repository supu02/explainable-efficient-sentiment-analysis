# Local packaging verification

Date: **15 September 2026**. These checks validate the packaged workflows; the headline table continues to describe the saved **16 April 2026** experiment.

## Completed checks

- All three model evaluators ran on the packaged 39-review CSV with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and an empty Hugging Face cache location.
- Accuracy, precision, recall, binary F1, correct counts, and the 22/17 cascade routing split matched the reference JSON exactly.
- The single-review cascade demo ran and returned a positive prediction for the illustrative review in the README, routed through Logistic Regression.
- Both LIME commands ran on the first evaluation review and produced finite saved attribution weights. These were one-example smoke checks, not a new explanation-quality evaluation.
- The figure script generated five PNG/PDF pairs from the reference JSON. All five PNGs were visually inspected for labels, scales, and clipping.
- The revised seven-page report compiled and its rendered pages were visually checked.
- The standard-library verification script checks source syntax, relative links, CSV hashes, metrics, routing, and explanation order. It also checks model hashes when the model archive is present.

## Scope and observations

New inference timings differed from the April reference, as expected for wall-clock measurements. They were kept in ignored `results/reruns/` outputs and did not replace the portfolio's recorded results.

The existing environment emitted PyTorch deprecation/backend warnings and NumPy/scikit-learn numerical warnings during LIME. The commands completed and the exported explanation weights were checked for finite values. Those checks do not establish numerical stability or explanation faithfulness across repeated runs.

Training was not rerun. A clean dependency installation on another operating system and a hosted GitHub Actions run have not been performed. The workflow file is included for future pushes. These checks were performed during local packaging before GitHub publication. They are historical verification notes, not evidence that the current GitHub checkout has been tested again.
