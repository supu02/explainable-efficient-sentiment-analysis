from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean evaluation CSV by removing any train/test text overlap.",
    )
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--source", default="data/sample_test.csv")
    parser.add_argument("--out", default="data/eval_clean.csv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    source_df = pd.read_csv(args.source)

    train_texts = set(train_df["text"].astype(str))
    clean_df = (
        source_df.loc[~source_df["text"].astype(str).isin(train_texts)]
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    clean_df.to_csv(out_path, index=False)

    print(f"train rows       : {len(train_df)}")
    print(f"source rows      : {len(source_df)}")
    print(f"clean eval rows  : {len(clean_df)}")
    print(f"label counts     : {clean_df['label'].value_counts().sort_index().to_dict()}")
    print(f"saved            : {out_path}")


if __name__ == "__main__":
    main()
