"""Benchmark helpers for logged ad bandit data (polars processing + pandas metrics)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Literal

import pandas as pd
import polars as pl

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable=None, **kwargs):  # type: ignore
        del kwargs
        return iterable

Action = int
NULL_FEATURE_FILL = -1e-6


@dataclass
class ScenarioConfig:
    name: str
    pretrain_source: Literal["random", "all", "none"]
    online_update: bool


class BasePolicy:
    can_update_online: bool = True

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del candidates, features, row
        raise NotImplementedError

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del action, reward, features

    def fit(self, train_df: pl.DataFrame) -> None:
        for row in train_df.iter_rows(named=True):
            self.update(int(row["show"]), float(row["reward"]), row["features_list"])


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts: dict[Action, int] = {}
        self.values: dict[Action, float] = {}

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del features, row
        if not candidates:
            raise ValueError("Empty candidate set")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)
        return max(candidates, key=lambda a: self.values.get(a, 0.0))

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        n = self.counts.get(action, 0) + 1
        v = self.values.get(action, 0.0)
        self.values[action] = v + (reward - v) / n
        self.counts[action] = n


class UCBPolicy(BasePolicy):
    def __init__(self, exploration: float = 2.0):
        self.exploration = exploration
        self.t = 0
        self.counts: dict[Action, int] = {}
        self.values: dict[Action, float] = {}

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del features, row
        if not candidates:
            raise ValueError("Empty candidate set")
        for a in candidates:
            if self.counts.get(a, 0) == 0:
                return a
        log_t = math.log(max(self.t, 1))
        return max(candidates, key=lambda a: self.values[a] + math.sqrt(self.exploration * log_t / self.counts[a]))

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        self.t += 1
        n = self.counts.get(action, 0) + 1
        v = self.values.get(action, 0.0)
        self.values[action] = v + (reward - v) / n
        self.counts[action] = n


class ThompsonSamplingPolicy(BasePolicy):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, seed: int = 42):
        self.alpha0 = alpha
        self.beta0 = beta
        self.rng = random.Random(seed)
        self.alpha: dict[Action, float] = {}
        self.beta: dict[Action, float] = {}

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del features, row
        if not candidates:
            raise ValueError("Empty candidate set")
        return max(candidates, key=lambda a: self.rng.betavariate(self.alpha.get(a, self.alpha0), self.beta.get(a, self.beta0)))

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        self.alpha[action] = self.alpha.get(action, self.alpha0) + reward
        self.beta[action] = self.beta.get(action, self.beta0) + (1.0 - reward)


class ContextualBanditPlaceholder(BasePolicy):
    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del candidates, features, row
        raise NotImplementedError("Contextual bandits are intentionally not implemented yet")


def _parse_candidates(raw: str) -> list[int]:
    if raw is None or raw == "":
        return []
    return [int(x) for x in str(raw).split("\t") if str(x) != ""]


def _parse_features(raw: str) -> list[float]:
    if raw is None or raw == "":
        return []

    vals: list[float] = []
    for x in str(raw).split("\t"):
        token = str(x).strip().lower()
        if token == "":
            continue
        if token in {"null", "none", "nan"}:
            vals.append(NULL_FEATURE_FILL)
        else:
            vals.append(float(token))
    return vals


def preprocess_bandit_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    req = {"policy", "reward", "puid", "features", "show", "candidates", "date"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.with_columns(
        [
            pl.col("show").cast(pl.Int64),
            pl.col("reward"),
            pl.col("date").str.to_datetime(strict=False),
            pl.col("candidates").map_elements(_parse_candidates, return_dtype=pl.List(pl.Int64)).alias("candidates_list"),
            pl.col("features").map_elements(_parse_features, return_dtype=pl.List(pl.Float64)).alias("features_list"),
        ]
    )


def split_train_test_by_date(df: pl.DataFrame, test_ratio: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0,1)")
    ordered = df.sort("date")
    split_idx = int(ordered.height * (1.0 - test_ratio))
    return ordered.slice(0, split_idx), ordered.slice(split_idx, ordered.height - split_idx)


def select_pretrain_data(train_df: pl.DataFrame, source: Literal["random", "all", "none"]) -> pl.DataFrame:
    if source == "none":
        return train_df.clear()
    if source == "all":
        return train_df
    if source == "random":
        return train_df.filter(pl.col("policy") == "random")
    raise ValueError(f"Unknown pretrain source: {source}")


def build_expected_reward_estimator(train_df: pl.DataFrame) -> Callable[[dict[str, object], Action], float]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in train_df.iter_rows(named=True):
        a = int(row["show"])
        r = float(row["reward"])
        sums[a] = sums.get(a, 0.0) + r
        counts[a] = counts.get(a, 0) + 1

    total_sum = sum(sums.values())
    total_n = sum(counts.values())
    global_mean = (total_sum / total_n) if total_n > 0 else 0.0

    def estimate(_row: dict[str, object], action: Action) -> float:
        n = counts.get(action, 0)
        if n == 0:
            return global_mean
        return sums[action] / n

    return estimate


def evaluate_policy(
    policy: BasePolicy,
    test_df: pl.DataFrame,
    online_update: bool,
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    expected_reward_fn: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
    progress_desc: str = "evaluate",
    progress_position: int = 1,
    on_step: Callable[[str, int, pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_reward = 0.0
    used = 0
    replay_matches = 0
    cumulative_regret = 0.0
    history_rows: list[dict[str, float | int]] = []

    rows = test_df.iter_rows(named=True)
    rows_iter = tqdm(rows, total=test_df.height, desc=progress_desc, leave=False, disable=not show_progress, position=progress_position)

    for step, row in enumerate(rows_iter, start=1):
        candidates = row["candidates_list"]
        features = row["features_list"]
        action = int(policy.select(candidates, features, row))

        if env_reward is None:
            if action != int(row["show"]):
                continue
            reward = float(row["reward"])
            replay_matches += 1
            if expected_reward_fn is None:
                regret = 0.0
            else:
                mu_chosen = float(expected_reward_fn(row, action))
                mu_best = max(float(expected_reward_fn(row, a)) for a in candidates) if candidates else mu_chosen
                regret = mu_best - mu_chosen
        else:
            reward = float(env_reward(row, action))
            candidate_rewards = [float(env_reward(row, a)) for a in candidates] if candidates else [reward]
            best_reward = max(candidate_rewards) if candidate_rewards else reward
            regret = best_reward - reward
            if action == int(row["show"]):
                replay_matches += 1

        total_reward += reward
        cumulative_regret += regret
        used += 1

        if online_update and policy.can_update_online:
            policy.update(action, reward, features)

        history_rows.append(
            {
                "step": step,
                "reward": reward,
                "avg_reward": total_reward / used,
                "cumulative_regret": cumulative_regret,
                "avg_regret": cumulative_regret / used,
            }
        )

        if on_step is not None:
            on_step(progress_desc, step, pd.DataFrame(history_rows))

    ctr = total_reward / used if used else 0.0
    match_rate = replay_matches / test_df.height if test_df.height else 0.0
    metrics_df = pd.DataFrame(
        [
            {
                "impressions_total": test_df.height,
                "impressions_used": used,
                "total_reward": total_reward,
                "ctr": ctr,
                "replay_match_rate": match_rate,
            }
        ]
    )
    history_df = pd.DataFrame(history_rows)
    return metrics_df, history_df


def run_scenarios(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    policy_factories: dict[str, Callable[[], BasePolicy]],
    scenarios: list[ScenarioConfig],
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
    on_step: Callable[[str, int, pd.DataFrame], None] | None = None,
) -> dict[str, pd.DataFrame]:
    metrics_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []

    scenario_iter = tqdm(scenarios, desc="scenarios", leave=False, position=0, disable=not show_progress)
    for scenario in scenario_iter:
        scenario_iter.set_description(f"scenario={scenario.name}")
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)
        expected_reward_fn = build_expected_reward_estimator(pretrain_df if pretrain_df.height > 0 else train_df)

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_df.height > 0:
                policy.fit(pretrain_df)

            metrics_df, history_df = evaluate_policy(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                env_reward=env_reward,
                expected_reward_fn=expected_reward_fn,
                show_progress=show_progress,
                progress_desc=f"{scenario.name}/{algo_name}",
                progress_position=1,
                on_step=on_step,
            )
            metrics_df["scenario"] = scenario.name
            metrics_df["algo"] = algo_name
            metrics_parts.append(metrics_df)

            if not history_df.empty:
                history_df["scenario"] = scenario.name
                history_df["algo"] = algo_name
                history_parts.append(history_df)

    out_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    out_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    return {"metrics": out_metrics, "history": out_history}


def make_simulated_environment(
    proba_predictor: Callable[[dict[str, object], Action], float],
    stochastic: bool = True,
    seed: int = 42,
) -> Callable[[dict[str, object], Action], float]:
    rng = random.Random(seed)

    def env_reward(row: dict[str, object], action: Action) -> float:
        p = max(0.0, min(1.0, float(proba_predictor(row, action))))
        if not stochastic:
            return p
        return 1.0 if rng.random() < p else 0.0

    return env_reward


def default_five_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("case_1_random_pretrain_predict_only", "random", False),
        ScenarioConfig("case_2_random_pretrain_online_update", "random", True),
        ScenarioConfig("case_3_all_pretrain_predict_only", "all", False),
        ScenarioConfig("case_4_all_pretrain_online_update", "all", True),
        ScenarioConfig("case_5_no_pretrain_online_update", "none", True),
    ]
