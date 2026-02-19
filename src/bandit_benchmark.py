"""Benchmark helpers for logged ad bandit data (pandas-first version)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Literal

import pandas as pd

Action = int


@dataclass
class ScenarioConfig:
    name: str
    pretrain_source: Literal["random", "all", "none"]
    online_update: bool


class BasePolicy:
    can_update_online: bool = True

    def select(self, candidates: list[Action], features: list[float], row: pd.Series) -> Action:
        del candidates, features, row
        raise NotImplementedError

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del action, reward, features

    def fit(self, train_df: pd.DataFrame) -> None:
        for _, row in train_df.iterrows():
            self.update(int(row["show"]), float(row["reward"]), row["features_list"])


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts: dict[Action, int] = {}
        self.values: dict[Action, float] = {}

    def select(self, candidates: list[Action], features: list[float], row: pd.Series) -> Action:
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

    def select(self, candidates: list[Action], features: list[float], row: pd.Series) -> Action:
        del features, row
        if not candidates:
            raise ValueError("Empty candidate set")
        for a in candidates:
            if self.counts.get(a, 0) == 0:
                return a

        log_t = math.log(max(self.t, 1))
        return max(
            candidates,
            key=lambda a: self.values[a] + math.sqrt(self.exploration * log_t / self.counts[a]),
        )

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

    def select(self, candidates: list[Action], features: list[float], row: pd.Series) -> Action:
        del features, row
        if not candidates:
            raise ValueError("Empty candidate set")
        return max(
            candidates,
            key=lambda a: self.rng.betavariate(self.alpha.get(a, self.alpha0), self.beta.get(a, self.beta0)),
        )

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        self.alpha[action] = self.alpha.get(action, self.alpha0) + reward
        self.beta[action] = self.beta.get(action, self.beta0) + (1.0 - reward)


class CatBoostPolicy(BasePolicy):
    can_update_online = False

    def __init__(self, scorer: Callable[[pd.Series, Action], float]):
        self.scorer = scorer

    def select(self, candidates: list[Action], features: list[float], row: pd.Series) -> Action:
        del features
        if not candidates:
            raise ValueError("Empty candidate set")
        return max(candidates, key=lambda a: self.scorer(row, a))


class ContextualBanditPlaceholder(BasePolicy):
    def select(self, candidates: list[Action], features: list[float], row: pd.Series) -> Action:
        del candidates, features, row
        raise NotImplementedError("Contextual bandits are intentionally not implemented yet")


def _parse_candidates(raw: str) -> list[int]:
    if raw is None or raw == "":
        return []
    return [int(x) for x in str(raw).split("\t") if str(x) != ""]


def _parse_features(raw: str) -> list[float]:
    if raw is None or raw == "":
        return []
    return [float(x) for x in str(raw).split("\t") if str(x) != ""]


def preprocess_bandit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse source dataframe:
    - candidates: tab-separated ids -> list[int]
    - features: tab-separated numeric values -> list[float]
    """
    req = {"policy", "reward", "puid", "features", "show", "candidates", "date"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["show"] = out["show"].astype(int)
    out["reward"] = out["reward"].astype(float)
    out["date"] = pd.to_datetime(out["date"])
    out["candidates_list"] = out["candidates"].map(_parse_candidates)
    out["features_list"] = out["features"].map(_parse_features)
    return out


def split_train_test_by_date(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0,1)")
    ordered = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(ordered) * (1.0 - test_ratio))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def select_pretrain_data(train_df: pd.DataFrame, source: Literal["random", "all", "none"]) -> pd.DataFrame:
    if source == "none":
        return train_df.iloc[0:0].copy()
    if source == "all":
        return train_df
    if source == "random":
        return train_df[train_df["policy"] == "random"].copy()
    raise ValueError(f"Unknown pretrain source: {source}")


def evaluate_policy(
    policy: BasePolicy,
    test_df: pd.DataFrame,
    online_update: bool,
    env_reward: Callable[[pd.Series, Action], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns:
    - metrics dataframe with one row
    - history dataframe with columns: step, reward, avg_reward, cumulative_regret, avg_regret
    """
    total_reward = 0.0
    used = 0
    replay_matches = 0
    cumulative_regret = 0.0
    history_rows: list[dict[str, float | int]] = []

    for step, (_, row) in enumerate(test_df.iterrows(), start=1):
        candidates = row["candidates_list"]
        features = row["features_list"]
        action = int(policy.select(candidates, features, row))

        if env_reward is None:
            if action != int(row["show"]):
                continue
            reward = float(row["reward"])
            replay_matches += 1
            regret = 0.0
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

    ctr = total_reward / used if used else 0.0
    match_rate = replay_matches / len(test_df) if len(test_df) else 0.0
    metrics_df = pd.DataFrame(
        [
            {
                "impressions_total": int(len(test_df)),
                "impressions_used": int(used),
                "total_reward": float(total_reward),
                "ctr": float(ctr),
                "replay_match_rate": float(match_rate),
            }
        ]
    )
    history_df = pd.DataFrame(history_rows)
    return metrics_df, history_df


def run_scenarios(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    policy_factories: dict[str, Callable[[], BasePolicy]],
    scenarios: list[ScenarioConfig],
    env_reward: Callable[[pd.Series, Action], float] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run user-defined scenarios.

    Returns dict with two tables:
    - result["metrics"]: pandas DataFrame with one row per (scenario, algo)
    - result["history"]: pandas DataFrame with avg reward/regret curves per (scenario, algo, step)
    """
    metrics_rows: list[pd.DataFrame] = []
    history_rows: list[pd.DataFrame] = []

    for scenario in scenarios:
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if len(pretrain_df) > 0:
                policy.fit(pretrain_df)

            metrics_df, history_df = evaluate_policy(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                env_reward=env_reward,
            )
            metrics_df["scenario"] = scenario.name
            metrics_df["algo"] = algo_name
            metrics_rows.append(metrics_df)

            if len(history_df) > 0:
                history_df["scenario"] = scenario.name
                history_df["algo"] = algo_name
                history_rows.append(history_df)

    out_metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    out_history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    return {"metrics": out_metrics, "history": out_history}


def make_simulated_environment(
    proba_predictor: Callable[[pd.Series, Action], float],
    stochastic: bool = True,
    seed: int = 42,
) -> Callable[[pd.Series, Action], float]:
    rng = random.Random(seed)

    def env_reward(row: pd.Series, action: Action) -> float:
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
