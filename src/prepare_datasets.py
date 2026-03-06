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

from bandit_benchmark import (
    apply_standard_scaler_to_features,
    filter_test_by_train_candidate_coverage,
    preprocess_bandit_dataframe,
    split_train_test_by_date,
)


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


def _save_variant(train_df: pl.DataFrame, test_df: pl.DataFrame, out: Path, variant_name: str) -> tuple[Path, Path]:
    train_path = out / f"train_{variant_name}.parquet"
    test_path = out / f"test_{variant_name}.parquet"
    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)
    return train_path, test_path


def stage2_scale_features(train_stage1: Path, test_stage1: Path, out_dir: str) -> tuple[Path, Path]:
    """Build all requested preprocessing variants.

    Returns default prepared pair equivalent to variant_2 (scaled features).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_base = pl.read_parquet(train_stage1)
    test_base = pl.read_parquet(test_stage1)

    # 2) train/test with standardized features
    train_scaled = apply_standard_scaler_to_features(train_base)
    test_scaled = apply_standard_scaler_to_features(test_base)

    # 3) scaled + test rows with no unseen actions vs train
    test_scaled_known = filter_test_by_train_candidate_coverage(train_scaled, test_scaled)

    # 4) train/test without scaling
    train_raw = train_base
    test_raw = test_base

    # 5) raw + test rows with no unseen actions vs train
    test_raw_known = filter_test_by_train_candidate_coverage(train_raw, test_raw)

    variants = {
        # 1) test random-only filtering is already applied in stage1; keep explicit artifact
        "variant_1_random_test_only": (train_raw, test_raw),
        "variant_2_scaled": (train_scaled, test_scaled),
        "variant_3_scaled_test_known_actions": (train_scaled, test_scaled_known),
        "variant_5_raw": (train_raw, test_raw),
        "variant_6_raw_test_known_actions": (train_raw, test_raw_known),
    }

    for name, (tr, te) in variants.items():
        _save_variant(tr, te, out, name)

    # Backward-compatible default artifacts = variant 2
    train_final = out / "train_prepared.parquet"
    test_final = out / "test_prepared.parquet"
    train_scaled.write_parquet(train_final)
    test_scaled.write_parquet(test_final)
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
    print(f"prepared train (default variant_2_scaled): {train_final}")
    print(f"prepared test  (default variant_2_scaled): {test_final}")
    print("additional stage2 variants were also saved under out-dir")


if __name__ == "__main__":
    main()
