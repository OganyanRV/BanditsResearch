"""CLI launcher for scenario-based bandit benchmark (polars input)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl

from bandit_benchmark import (
    Action,
    EpsilonGreedyPolicy,
    ScenarioConfig,
    ThompsonSamplingPolicy,
    UCBPolicy,
    build_expected_reward_estimator,
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


def default_scenarios_from_args() -> list[ScenarioConfig]:
    return default_five_scenarios()


def make_live_plot_callback(enabled: bool, every_steps: int):
    if not enabled:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def callback(run_name: str, step: int, history_df: pd.DataFrame) -> None:
        if step % every_steps != 0 and step != 1:
            return
        if history_df.empty:
            return
        axes[0].clear()
        axes[1].clear()

        axes[0].plot(history_df["step"], history_df["avg_reward"])
        axes[0].set_title(f"{run_name}: avg_reward")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("avg_reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history_df["step"], history_df["avg_regret"])
        axes[1].set_title(f"{run_name}: avg_regret")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("avg_regret")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.001)

    return callback


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
            axes[0].plot(algo_df["step"], algo_df["avg_reward"], label=algo)
            axes[1].plot(algo_df["step"], algo_df["avg_regret"], label=algo)

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
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--live-plots", action="store_true", help="Show live plots during evaluation")
    parser.add_argument("--plot-every", type=int, default=50, help="Update live plot every N used steps")
    args = parser.parse_args()

    raw_df = load_dataset(args.input)
    df = preprocess_bandit_dataframe(raw_df)
    train_df, test_df = split_train_test_by_date(df, test_ratio=args.test_ratio)

    policy_factories = {
        "epsilon_greedy": lambda: EpsilonGreedyPolicy(epsilon=args.epsilon, seed=args.seed),
        "ucb": lambda: UCBPolicy(),
        "thompson_sampling": lambda: ThompsonSamplingPolicy(seed=args.seed),
    }

    env_reward = None
    if args.simulate:
        expected_fn = build_expected_reward_estimator(train_df)
        env_reward = make_simulated_environment(proba_predictor=expected_fn, stochastic=args.stochastic_sim, seed=args.seed)

    scenarios = default_scenarios_from_args()
    live_cb = make_live_plot_callback(enabled=args.live_plots, every_steps=max(args.plot_every, 1))

    result = run_scenarios(
        train_df=train_df,
        test_df=test_df,
        policy_factories=policy_factories,
        scenarios=scenarios,
        env_reward=env_reward,
        show_progress=not args.no_progress,
        on_step=live_cb,
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
