from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

NULL_FEATURE_FILL = 0.0


def _normalize_action_token(raw: object) -> str:
    return str(raw)


def _parse_candidates(raw: str) -> list[str]:
    if raw is None or raw == "":
        return []
    return [_normalize_action_token(x) for x in str(raw).split("\\t") if str(x) != ""]


def _parse_features(raw: str) -> list[float]:
    if raw is None or raw == "":
        return []

    vals: list[float] = []
    for x in str(raw).split("\\t"):
        token = str(x).strip().lower()
        if token == "":
            continue
        if token in {"null", "none", "nan"}:
            vals.append(NULL_FEATURE_FILL)
        else:
            vals.append(float(token))

    return vals


def preprocess_bandit_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    req = {"policy", "reward", "features", "show", "candidates", "date"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = df.with_columns(
        [
            pl.col("show").map_elements(_normalize_action_token, return_dtype=pl.String).alias("show"),
            pl.col("reward"),
            pl.col("date").str.to_datetime(strict=False),
            pl.col("candidates").map_elements(_parse_candidates, return_dtype=pl.List(pl.String)).alias("candidates_list"),
            pl.col("features").map_elements(_parse_features, return_dtype=pl.List(pl.Float64)).alias("features_list"),
        ]
    )

    return prepared.with_columns(
        (pl.lit(1.0) / pl.col("candidates_list").list.len().cast(pl.Float64)).alias("propensity")
    )


def _transform_features_with_scaler(df: pl.DataFrame, scaler) -> pl.DataFrame:
    import numpy as np

    feat_rows = df.select("features_list").to_series().to_list()
    non_empty_idx = [i for i, row in enumerate(feat_rows) if isinstance(row, list) and len(row) > 0]
    if not non_empty_idx:
        return df

    dim = len(feat_rows[non_empty_idx[0]])
    valid_idx = [i for i in non_empty_idx if len(feat_rows[i]) == dim]
    if not valid_idx:
        return df

    X = np.array([feat_rows[i] for i in valid_idx], dtype=float)
    Xs = scaler.transform(X)
    for j, i in enumerate(valid_idx):
        feat_rows[i] = [float(v) for v in Xs[j].tolist()]
    return df.with_columns(pl.Series("features_list", feat_rows))


def apply_standard_scaler_to_features(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    fit_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit scaler on train features and transform both train and test.

    If sklearn is unavailable or compatible features cannot be extracted, returns inputs unchanged.
    """
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        train_feat_rows = train_df.select("features_list").to_series().to_list()
        non_empty_idx = [i for i, row in enumerate(train_feat_rows) if isinstance(row, list) and len(row) > 0]
        if not non_empty_idx:
            return train_df, test_df

        dim = len(train_feat_rows[non_empty_idx[0]])
        valid_idx = [i for i in non_empty_idx if len(train_feat_rows[i]) == dim]
        if not valid_idx:
            return train_df, test_df

        fit_idx = valid_idx
        if 0.0 < fit_fraction < 1.0:
            rng = np.random.default_rng(seed)
            k = max(1, int(round(len(valid_idx) * fit_fraction)))
            fit_idx = sorted(rng.choice(valid_idx, size=k, replace=False).tolist())

        X_fit = np.array([train_feat_rows[i] for i in fit_idx], dtype=float)
        scaler = StandardScaler()
        scaler.fit(X_fit)

        train_scaled = _transform_features_with_scaler(train_df, scaler)
        test_scaled = _transform_features_with_scaler(test_df, scaler)
        return train_scaled, test_scaled
    except Exception:
        return train_df, test_df


def filter_test_by_train_candidate_coverage(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """Keep test rows whose candidate list is fully covered by train actions.

    A row is preserved only if every action in `candidates_list` exists in train `show` actions.
    """
    train_actions = {str(r["show"]) for r in train_df.iter_rows(named=True)}
    if not train_actions:
        return test_df.clear()
    if test_df.is_empty():
        return test_df.clear()

    keep_mask: list[bool] = []
    for row in test_df.iter_rows(named=True):
        candidates = row.get("candidates_list") or []
        keep_mask.append(all(str(a) in train_actions for a in candidates))

    if not keep_mask:
        return test_df.clear()

    return test_df.filter(pl.Series("_keep", keep_mask))


def split_train_test_by_date(df: pl.DataFrame, train_days: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by date: first `train_days` unique dates go to train, rest to test."""
    if train_days <= 0:
        raise ValueError("train_days must be > 0")

    ordered = df.sort("date")
    unique_dates = ordered.select(pl.col("date").unique().sort()).to_series().to_list()
    if not unique_dates:
        return ordered.clear(), ordered.clear()

    cutoff_idx = min(int(train_days), len(unique_dates))
    train_date_set = set(unique_dates[:cutoff_idx])

    train_df = ordered.filter(pl.col("date").is_in(train_date_set))
    test_df = ordered.filter(~pl.col("date").is_in(train_date_set))
    return train_df, test_df


def load_source(path: str) -> pl.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t", schema_overrides={"candidates": pl.String})


def stage1_make_splits(input_path: str, out_dir: str, train_days: int, seed: int) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_df = load_source(input_path)
    df = preprocess_bandit_dataframe(raw_df)
    train_df, test_df = split_train_test_by_date(df, train_days=train_days)

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

    Returns default prepared pair equivalent to variant_1_scaled.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_base = pl.read_parquet(train_stage1)
    test_base = pl.read_parquet(test_stage1)

    # 1) train/test with standardized features using scaler fit on train
    train_scaled, test_scaled = apply_standard_scaler_to_features(train_base, test_base)

    # 2) scaled + test rows with no unseen actions vs train
    test_scaled_known = filter_test_by_train_candidate_coverage(train_scaled, test_scaled)

    # 3) train/test without scaling
    train_raw = train_base
    test_raw = test_base

    # 4) raw + test rows with no unseen actions vs train
    test_raw_known = filter_test_by_train_candidate_coverage(train_raw, test_raw)

    variants = {
        "variant_1_scaled": (train_scaled, test_scaled),
        "variant_2_scaled_test_known_actions": (train_scaled, test_scaled_known),
        "variant_3_raw": (train_raw, test_raw),
        "variant_4_raw_test_known_actions": (train_raw, test_raw_known),
    }

    for name, (tr, te) in variants.items():
        _save_variant(tr, te, out, name)

    # Backward-compatible default artifacts = variant 1
    train_final = out / "train_prepared.parquet"
    test_final = out / "test_prepared.parquet"
    train_scaled.write_parquet(train_final)
    test_scaled.write_parquet(test_final)
    return train_final, test_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test datasets in two stages")
    parser.add_argument("--input", required=True, help="Path to source tsv/parquet")
    parser.add_argument("--out-dir", default="artifacts/datasets")
    parser.add_argument("--train-days", type=int, default=1, help="How many first unique dates go to train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_s1, test_s1 = stage1_make_splits(args.input, args.out_dir, args.train_days, args.seed)
    print(f"stage1 train: {train_s1}")
    print(f"stage1 test:  {test_s1}")

    train_final, test_final = stage2_scale_features(train_s1, test_s1, args.out_dir)
    print(f"prepared train (default variant_1_scaled): {train_final}")
    print(f"prepared test  (default variant_1_scaled): {test_final}")
    print("additional stage2 variants were also saved under out-dir")


if __name__ == "__main__":
    main()
