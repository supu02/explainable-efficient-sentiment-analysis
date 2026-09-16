# Data provenance

Task: binary sentiment classification. Required CSV columns are `text` and `label`; the supplied files also retain `id`. Labels are `0 = negative`, `1 = positive`.

| File | Rows | Included in source archive? | Purpose |
|---|---:|---|---|
| `train.csv` | 30,000 | No | Original project training file, 15,000 reviews per class |
| `sample_test.csv` | 100 | Yes | Candidate sample before overlap removal |
| `eval_clean.csv` | 39 | Yes | Final sample, 20 negative and 19 positive reviews |

The candidate sample contains 61 rows whose raw text exactly matches a row in the local training file. Removing those rows leaves 39 unique review texts; the final duplicate-removal step removes zero additional rows.

The source is the [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The original dataset has official 25,000-review train/test partitions. The project's 30,000-review training file is a custom partition: the original extraction/split recipe is not retained, so downloading the official dataset does not reproduce this training file automatically.

## Rebuild the cleaned sample

The optional `xai-sentiment-training-data.zip` supplied with the handoff contains the original training CSV. Extract it from the repository root, then rebuild the cleaned sample:

```bash
unzip ../xai-sentiment-training-data.zip
python scripts/create_clean_eval_split.py --train data/train.csv --source data/sample_test.csv --out data/eval_clean.csv
```

[split_manifest.json](split_manifest.json) records row counts, exclusion counts, class balance, and SHA-256 hashes of the original CSV files. Exact hashes establish file identity; they do not certify the entire historical model-training process.

The cleaning checks exact raw-text overlap. Near-duplicate detection, provenance of earlier training checkpoints, and a full independent holdout audit are outside the completed experiment. The cleaned sample is exploratory and should not be described as the full IMDb test benchmark.
