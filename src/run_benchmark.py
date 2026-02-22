"""CLI launcher for scenario-based bandit benchmark (polars input)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from bandit_benchmark import (
    CatBoostPolicy,
    EpsilonGreedyPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
    build_expected_reward_estimator,
    default_five_ips_scenarios,
    default_five_scenarios,
    make_simulated_environment,
    preprocess_bandit_dataframe,
    run_scenarios,
    split_train_test_by_date,
)


def load_dataset(path: str) -> pl.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t", schema_overrides={"candidates": pl.String})


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
            stride = max(1, int(round(max_step * 0.02)))
            ds = algo_df.iloc[stride::stride] if len(algo_df) > stride else algo_df
            axes[0].plot(ds["step"], ds["avg_reward"], label=algo)
            axes[1].plot(ds["step"], ds["avg_regret"], label=algo)

        axes[0].set_title(f"{scenario_name}: average reward")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("avg_reward")

        axes[1].set_title(f"{scenario_name}: average regret")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("avg_regret")

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
    parser.add_argument("--input", required=True, help="Path to source dataset (tsv/parquet)")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulate", action="store_true", help="Use learned environment simulation")
    parser.add_argument("--stochastic-sim", action="store_true", help="In simulation, sample Bernoulli reward")
    parser.add_argument("--ips-scenarios", action="store_true", help="Run IPS-oriented five scenarios naming")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    raw_df = load_dataset(args.input)
    df = preprocess_bandit_dataframe(raw_df)
    train_df, test_df = split_train_test_by_date(df, test_ratio=args.test_ratio)

    train_df = train_df.sample(fraction=1.0, shuffle=True, seed=args.seed).sort("date")
    test_df = test_df.sample(fraction=1.0, shuffle=True, seed=args.seed).sort("date")

    test_df = test_df.filter(pl.col("policy") == "random")

    policy_factories = {
        "epsilon_greedy": lambda: EpsilonGreedyPolicy(epsilon=args.epsilon, seed=args.seed),
        "ucb": lambda: UCBPolicy(),
        "thompson_sampling": lambda: ThompsonSamplingPolicy(seed=args.seed),
    }

    try:
        import catboost  # noqa: F401
        policy_factories["catboost"] = lambda: CatBoostPolicy(random_seed=args.seed)
    except Exception:
        print("catboost is unavailable: skipping CatBoostPolicy")

    env_reward = None
    if args.simulate:
        expected_fn = build_expected_reward_estimator(train_df)
        env_reward = make_simulated_environment(proba_predictor=expected_fn, stochastic=args.stochastic_sim, seed=args.seed)

    scenarios = default_five_ips_scenarios() if args.ips_scenarios else default_five_scenarios()

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    history_path = out_dir / "history.csv"
    metrics_df.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)

    print(f"saved metrics: {metrics_path}")
    print(f"saved history: {history_path}")
    print(metrics_df)

    plot_paths = save_plots(history_df, str(out_dir / "plots"))
    if plot_paths:
        print("saved plots:")
        for p in plot_paths:
            print(f" - {p}")
    else:
        print("plots were not generated (matplotlib is unavailable or no history)")


if __name__ == "__main__":
    main()
