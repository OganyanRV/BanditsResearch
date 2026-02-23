"""Two-stage dataset preparation pipeline.

Stage 1:
- read source TSV/Parquet
- preprocess columns into candidates/features lists + propensity
- split train/test
- shuffle+sort each split
- filter test to policy == random
- save interim splits to disk

Stage 2:
- load interim splits
- apply StandardScaler to features_list
- overwrite/save final prepared train/test files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from bandit_benchmark import apply_standard_scaler_to_features, preprocess_bandit_dataframe, split_train_test_by_date


def load_source(path: str) -> pl.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t", schema_overrides={"candidates": pl.String})


def stage1_make_splits(input_path: str, out_dir: str, test_ratio: float, seed: int) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_df = load_source(input_path)
    df = preprocess_bandit_dataframe(raw_df)
    train_df, test_df = split_train_test_by_date(df, test_ratio=test_ratio)

    train_df = train_df.sample(fraction=1.0, shuffle=True, seed=seed).sort("date")
    test_df = test_df.sample(fraction=1.0, shuffle=True, seed=seed).sort("date")
    test_df = test_df.filter(pl.col("policy") == "random")

    train_stage1 = out / "train_stage1.parquet"
    test_stage1 = out / "test_stage1.parquet"
    train_df.write_parquet(train_stage1)
    test_df.write_parquet(test_stage1)
    return train_stage1, test_stage1


def stage2_scale_features(train_stage1: Path, test_stage1: Path, out_dir: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pl.read_parquet(train_stage1)
    test_df = pl.read_parquet(test_stage1)

    train_df = apply_standard_scaler_to_features(train_df)
    test_df = apply_standard_scaler_to_features(test_df)

    train_final = out / "train_prepared.parquet"
    test_final = out / "test_prepared.parquet"
    train_df.write_parquet(train_final)
    test_df.write_parquet(test_final)
    return train_final, test_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test datasets in two stages")
    parser.add_argument("--input", required=True, help="Path to source tsv/parquet")
    parser.add_argument("--out-dir", default="artifacts/datasets")
    parser.add_argument("--test-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_s1, test_s1 = stage1_make_splits(args.input, args.out_dir, args.test_ratio, args.seed)
    print(f"stage1 train: {train_s1}")
    print(f"stage1 test:  {test_s1}")

    train_final, test_final = stage2_scale_features(train_s1, test_s1, args.out_dir)
    print(f"prepared train: {train_final}")
    print(f"prepared test:  {test_final}")


if __name__ == "__main__":
    main()
