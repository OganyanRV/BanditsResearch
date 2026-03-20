"""CLI launcher for scenario-based bandit benchmark (polars input)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from prepare_datasets import preprocess_bandit_dataframe, split_train_test_by_date
from bandit_benchmark import (
    CatBoostPolicy,
    CatBoostOneTreePolicy,
    CatBoostTreeThompsonSamplingPolicyUpdateV1,
    EpsilonGreedyPolicy,
    LogisticTSLibPolicy,
    LaplaceThompsonViaBayesianLogRegPolicy,
    NeuralLaplaceThompsonViaBayesianLogRegPolicy,
    RandomPolicy,
    PartitionedTSLibPolicy,
    ThompsonSamplingPolicy,
    TreeThompsonSamplingPolicy,
    TreeThompsonSamplingPolicyDummyRefit,
    UCBPolicy,
    build_expected_reward_estimator,
    default_scenario,
    default_five_scenarios,
    make_simulated_environment,
    run_scenarios,
)


def load_dataset(path: str) -> pl.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t", schema_overrides={"candidates": pl.String})


def load_prepared_splits(train_path: str, test_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(train_path), pl.read_parquet(test_path)


def save_plots(history_df: pd.DataFrame, out_dir: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_paths: list[str] = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        return output_paths

    for scenario_name, part in history_df.groupby("scenario"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for algo, algo_df in part.groupby("algo"):
            max_step = int(algo_df["step"].max()) if len(algo_df) else 0
            stride = max(1, int(round(max_step * 0.05)))
            ds = algo_df.iloc[stride::stride] if len(algo_df) > stride else algo_df
            axes[0].plot(ds["step"], ds["avg_reward"], label=algo)
            axes[1].plot(ds["step"], ds["avg_regret"], label=algo)

        axes[0].set_title(f"{scenario_name}: average reward")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("avg_reward")

        axes[1].set_title(f"{scenario_name}: average regret")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("avg_regret")
        axes[1].set_yscale("log")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()

        out_path = str(Path(out_dir) / f"{scenario_name}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario-based benchmark for bandit policies")
    parser.add_argument("--input", help="Path to source dataset (tsv/parquet)")
    parser.add_argument("--train-path", default="artifacts/datasets/train_prepared.parquet", help="Prepared train parquet")
    parser.add_argument("--test-path", default="artifacts/datasets/test_prepared.parquet", help="Prepared test parquet")
    parser.add_argument("--train-days", type=int, default=1, help="How many first unique dates go to train")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulate", action="store_true", help="Use learned environment simulation")
    parser.add_argument("--stochastic-sim", action="store_true", help="In simulation, sample Bernoulli reward")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--full-scenarios", action="store_true", help="Run full five scenarios instead of core")
    parser.add_argument("--neural-hidden-dims", default="64,32", help="Comma-separated hidden layer sizes for neural_laplace_ts_logreg")
    parser.add_argument("--encoder-train-data-mode", choices=["all", "random_half", "time_half"], default="all")
    args = parser.parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    if train_path.exists() and test_path.exists():
        train_df, test_df = load_prepared_splits(str(train_path), str(test_path))
        print(f"loaded prepared splits: {train_path}, {test_path}")
    else:
        if not args.input:
            raise ValueError("Either provide --input or prepare datasets at --train-path/--test-path")
        raw_df = load_dataset(args.input)
        df = preprocess_bandit_dataframe(raw_df)
        train_df, test_df = split_train_test_by_date(df, train_days=args.train_days)
        train_df = train_df.sample(fraction=1.0, shuffle=True, seed=args.seed).sort("date")
        test_df = test_df.sample(fraction=1.0, shuffle=True, seed=args.seed).sort("date")
        test_df = test_df.filter(pl.col("policy") == "random")

    policy_factories = {
        "random": lambda: RandomPolicy(seed=args.seed),
        "epsilon_greedy": lambda: EpsilonGreedyPolicy(epsilon=args.epsilon, seed=args.seed),
        "ucb": lambda: UCBPolicy(),
        "thompson_sampling": lambda: ThompsonSamplingPolicy(seed=args.seed),
    }

    try:
        import sklearn  # noqa: F401
        policy_factories["tree_thompson_sampling_refit"] = lambda: TreeThompsonSamplingPolicyDummyRefit(random_state=args.seed)
    except Exception:
        print("sklearn is unavailable: skipping TreeThompsonSamplingPolicy")

    try:
        import scipy  # noqa: F401
        policy_factories["laplace_ts_logreg"] = lambda: LaplaceThompsonViaBayesianLogRegPolicy(seed=args.seed)
        try:
            import torch  # noqa: F401
            hidden_dims = [int(x.strip()) for x in args.neural_hidden_dims.split(",") if x.strip()]
            policy_factories["neural_laplace_ts_logreg"] = lambda: NeuralLaplaceThompsonViaBayesianLogRegPolicy(
                seed=args.seed,
                hidden_dims=hidden_dims,
                encoder_train_data_mode=args.encoder_train_data_mode,
            )
        except Exception:
            print("torch is unavailable: skipping NeuralLaplaceThompsonViaBayesianLogRegPolicy")
    except Exception:
        print("scipy is unavailable: skipping LaplaceThompsonViaBayesianLogRegPolicy")

    try:
        import catboost  # noqa: F401
        policy_factories["catboost"] = lambda: CatBoostPolicy(random_seed=args.seed)
        policy_factories["catboost_one_tree"] = lambda: CatBoostOneTreePolicy(random_seed=args.seed)
        policy_factories["catboost_tree_thompson_sampling_update_v1"] = (
            lambda: CatBoostTreeThompsonSamplingPolicyUpdateV1(random_state=args.seed)
        )
    except Exception:
        print("catboost is unavailable: skipping CatBoostPolicy")

    try:
        import contextualbandits  # noqa: F401
        policy_factories["logistic_ts"] = lambda: LogisticTSLibPolicy(random_seed=args.seed)
        policy_factories["partitioned_ts"] = lambda: PartitionedTSLibPolicy(random_seed=args.seed)
    except Exception:
        print("contextualbandits is unavailable: skipping LogisticTS/PartitionedTS")

    env_reward = None
    if args.simulate:
        expected_fn = build_expected_reward_estimator(train_df)
        env_reward = make_simulated_environment(proba_predictor=expected_fn, stochastic=args.stochastic_sim, seed=args.seed)

    scenarios = default_five_scenarios() if args.full_scenarios else default_scenario()

    result = run_scenarios(
        train_df=train_df,
        test_df=test_df,
        policy_factories=policy_factories,
        scenarios=scenarios,
        env_reward=env_reward,
        show_progress=not args.no_progress,
    )

    metrics_df = result["metrics"]
    history_df = result["history"]
    action_stats_df = result["action_stats"]
    action_daily_stats_df = result["action_daily_stats"]
    action_ips_stats_df = result["action_ips_stats"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    history_path = out_dir / "history.csv"
    action_stats_path = out_dir / "action_stats.csv"
    action_daily_stats_path = out_dir / "action_daily_stats.csv"
    action_ips_stats_path = out_dir / "action_ips_stats.csv"
    metrics_df.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)
    action_stats_df.to_csv(action_stats_path, index=False)
    action_daily_stats_df.to_csv(action_daily_stats_path, index=False)
    action_ips_stats_df.to_csv(action_ips_stats_path, index=False)

    print(f"saved metrics: {metrics_path}")
    print(f"saved history: {history_path}")
    print(f"saved action stats: {action_stats_path}")
    print(f"saved action daily stats: {action_daily_stats_path}")
    print(f"saved action IPS stats: {action_ips_stats_path}")
    print(metrics_df)

    main_cols = ["scenario", "algo", "ips_ctr", "snips_ctr"]
    if set(main_cols).issubset(metrics_df.columns):
        print("IPS/SNIPS metrics:")
        print(metrics_df[main_cols].sort_values(["scenario", "algo"]).to_string(index=False))

    if args.simulate:
        print("action stats (date-change checkpoints):")
        print(action_stats_df)
        print("action daily stats (impressions by action/day):")
        print(action_daily_stats_df)
        print("action IPS stats:")
        print(action_ips_stats_df)

    plot_paths = save_plots(history_df, str(out_dir / "plots"))
    if plot_paths:
        print("saved plots:")
        for p in plot_paths:
            print(f" - {p}")
    else:
        print("plots were not generated (matplotlib is unavailable or no history)")


if __name__ == "__main__":
    main()
