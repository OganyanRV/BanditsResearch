"""Benchmark helpers for logged ad bandit data (polars processing + pandas metrics)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Literal

import pandas as pd
import polars as pl

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

Action = int
NULL_FEATURE_FILL = 0.0


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

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        for a, r, f in pending_updates:
            self.update(a, r, f)

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




class LogisticTSPolicy(BasePolicy):
    """Wrapper over contextualbandits.online.LogisticTS.

    Train-once in this benchmark and no online updates.
    """

    can_update_online = False

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self._model = None
        self._fitted = False
        self._actions: list[int] = []
        self._a2i: dict[int, int] = {}

    def fit(self, train_df: pl.DataFrame) -> None:
        if self._fitted:
            raise RuntimeError("LogisticTSPolicy can only be trained once")
        if train_df.height == 0:
            self._fitted = True
            return

        try:
            from contextualbandits.online import LogisticTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("contextualbandits is required for LogisticTSPolicy") from exc

        actions = sorted({int(r["show"]) for r in train_df.iter_rows(named=True)})
        if not actions:
            self._fitted = True
            return
        self._actions = actions
        self._a2i = {a: i for i, a in enumerate(actions)}

        X: list[list[float]] = []
        a: list[int] = []
        r: list[int] = []
        for row in train_df.iter_rows(named=True):
            action = int(row["show"])
            if action not in self._a2i:
                continue
            X.append(list(row["features_list"]))
            a.append(self._a2i[action])
            r.append(int(float(row["reward"]) > 0.0))

        if not X:
            self._fitted = True
            return

        model = LogisticTS(nchoices=len(self._actions), random_state=self.random_seed)
        model.fit(X, a, r)
        self._model = model
        self._fitted = True

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del row
        if not candidates:
            raise ValueError("Empty candidate set")
        if self._model is None:
            return candidates[0]

        probs = self._model.predict_proba([list(features)])[0]
        best_action = candidates[0]
        best_score = -1e18
        for a in candidates:
            idx = self._a2i.get(int(a))
            score = float(probs[idx]) if idx is not None else 0.0
            if score > best_score:
                best_score = score
                best_action = int(a)
        return best_action


class PartitionedTSPolicy(BasePolicy):
    """Wrapper over contextualbandits.online.PartitionedTS.

    Train-once in this benchmark and no online updates.
    """

    can_update_online = False

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self._model = None
        self._fitted = False
        self._actions: list[int] = []
        self._a2i: dict[int, int] = {}

    def fit(self, train_df: pl.DataFrame) -> None:
        if self._fitted:
            raise RuntimeError("PartitionedTSPolicy can only be trained once")
        if train_df.height == 0:
            self._fitted = True
            return

        try:
            from contextualbandits.online import PartitionedTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("contextualbandits is required for PartitionedTSPolicy") from exc

        actions = sorted({int(r["show"]) for r in train_df.iter_rows(named=True)})
        if not actions:
            self._fitted = True
            return
        self._actions = actions
        self._a2i = {a: i for i, a in enumerate(actions)}

        X: list[list[float]] = []
        a: list[int] = []
        r: list[int] = []
        for row in train_df.iter_rows(named=True):
            action = int(row["show"])
            if action not in self._a2i:
                continue
            X.append(list(row["features_list"]))
            a.append(self._a2i[action])
            r.append(int(float(row["reward"]) > 0.0))

        if not X:
            self._fitted = True
            return

        model = PartitionedTS(nchoices=len(self._actions), random_state=self.random_seed)
        model.fit(X, a, r)
        self._model = model
        self._fitted = True

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del row
        if not candidates:
            raise ValueError("Empty candidate set")
        if self._model is None:
            return candidates[0]

        probs = self._model.predict_proba([list(features)])[0]
        best_action = candidates[0]
        best_score = -1e18
        for a in candidates:
            idx = self._a2i.get(int(a))
            score = float(probs[idx]) if idx is not None else 0.0
            if score > best_score:
                best_score = score
                best_action = int(a)
        return best_action


class CatBoostPolicy(BasePolicy):
    """Gradient boosting policy based on CatBoostClassifier.

    Model is train-once: repeated fit calls are forbidden.
    """

    can_update_online = False

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self._model = None
        self._fitted = False

    @staticmethod
    def _row_to_vector(features: list[float], action: int) -> list[float]:
        return list(features) + [float(action)]

    def fit(self, train_df: pl.DataFrame) -> None:
        if self._fitted:
            raise RuntimeError("CatBoostPolicy can only be trained once")
        if train_df.height == 0:
            self._fitted = True
            return

        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("catboost is required for CatBoostPolicy") from exc

        X: list[list[float]] = []
        y: list[int] = []
        for row in train_df.iter_rows(named=True):
            action = int(row["show"])
            features = row["features_list"]
            X.append(self._row_to_vector(features, action))
            y.append(int(float(row["reward"]) > 0.0))

        if not X:
            self._fitted = True
            return

        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_seed=self.random_seed,
        )
        model.fit(X, y)
        self._model = model
        self._fitted = True

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del row
        if not candidates:
            raise ValueError("Empty candidate set")
        if self._model is None:
            return candidates[0]

        best_action = candidates[0]
        best_score = -1.0
        for a in candidates:
            vec = self._row_to_vector(features, int(a))
            p = float(self._model.predict_proba([vec])[0][1])
            if p > best_score:
                best_score = p
                best_action = int(a)
        return best_action

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del action, reward, features
        return


def _parse_candidates(raw: str) -> list[int]:
    if raw is None or raw == "":
        return []
    return [int(x) for x in str(raw).split("\\t") if str(x) != ""]


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
            pl.col("show").cast(pl.Int64),
            pl.col("reward"),
            pl.col("date").str.to_datetime(strict=False),
            pl.col("candidates").map_elements(_parse_candidates, return_dtype=pl.List(pl.Int64)).alias("candidates_list"),
            pl.col("features").map_elements(_parse_features, return_dtype=pl.List(pl.Float64)).alias("features_list"),
        ]
    )

    return prepared.with_columns(
        (pl.lit(1.0) / pl.col("candidates_list").list.len().cast(pl.Float64)).alias("propensity")
    )




def apply_standard_scaler_to_features(df: pl.DataFrame) -> pl.DataFrame:
    """Scale `features_list` with StandardScaler when sklearn is available."""
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        feat_rows = df.select("features_list").to_series().to_list()
        non_empty_idx = [i for i, row in enumerate(feat_rows) if isinstance(row, list) and len(row) > 0]
        if not non_empty_idx:
            return df

        dim = len(feat_rows[non_empty_idx[0]])
        valid_idx = [i for i in non_empty_idx if len(feat_rows[i]) == dim]
        if not valid_idx:
            return df

        X = np.array([feat_rows[i] for i in valid_idx], dtype=float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        for j, i in enumerate(valid_idx):
            feat_rows[i] = [float(v) for v in Xs[j].tolist()]
        return df.with_columns(pl.Series("features_list", feat_rows))
    except Exception:
        return df
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
        return (sums[action] / n) if n > 0 else global_mean

    return estimate




def build_random_action_ctr_stats(df: pl.DataFrame) -> tuple[dict[int, float], float]:
    random_df = df.filter(pl.col("policy") == "random") if "policy" in df.columns else df
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in random_df.iter_rows(named=True):
        a = int(row["show"])
        r = float(row["reward"])
        sums[a] = sums.get(a, 0.0) + r
        counts[a] = counts.get(a, 0) + 1

    ctr_by_action: dict[int, float] = {}
    for a, n in counts.items():
        ctr_by_action[a] = sums.get(a, 0.0) / n if n > 0 else 0.0
    max_ctr = max(ctr_by_action.values()) if ctr_by_action else 0.0
    return ctr_by_action, max_ctr
def evaluate_policy(
    policy: BasePolicy,
    test_df: pl.DataFrame,
    online_update: bool,
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
    progress_desc: str = "evaluate",
    ctr_by_action: dict[int, float] | None = None,
    global_random_ctr: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_reward = 0.0
    ips_weighted_reward_sum = 0.0
    used = 0
    replay_matches = 0
    cumulative_regret = 0.0
    cumulative_ips_regret = 0.0
    history_rows: list[dict[str, float | int]] = []

    action_ctr = ctr_by_action or {}

    total_steps = max(test_df.height, 1)
    update_chunk = max(1, int(total_steps * 0.10))
    progress_chunk = max(1, int(total_steps * 0.05))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=test_df.height, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=0.5)

    pending_updates: list[tuple[int, float, list[float]]] = []
    next_progress_mark = progress_chunk

    for step, row in enumerate(test_df.iter_rows(named=True), start=1):
        candidates = row["candidates_list"]
        features = row["features_list"]
        action = int(policy.select(candidates, features, row))

        logged_reward = float(row["reward"])
        logged_match = int(action == int(row["show"]))
        propensity = float(row.get("propensity", 0.0) or 0.0)

        ips_reward = (logged_match * logged_reward / propensity) if propensity > 0 else 0.0
        ips_weighted_reward_sum += ips_reward
        candidate_ctrs = [action_ctr.get(int(a), global_random_ctr) for a in candidates] if candidates else [global_random_ctr]
        step_max_ctr = max(candidate_ctrs) if candidate_ctrs else global_random_ctr
        ips_step_regret = step_max_ctr - (logged_reward if logged_match else 0.0)
        cumulative_ips_regret += ips_step_regret

        if env_reward is None:
            if not logged_match:
                if pbar is not None and step >= next_progress_mark:
                    pbar.update(step - pbar.n)
                    next_progress_mark += progress_chunk
                continue
            reward = logged_reward
            replay_matches += 1
            regret = step_max_ctr - reward
        else:
            reward = float(env_reward(row, action))
            replay_matches += int(action == int(row["show"]))
            regret = step_max_ctr - reward

        total_reward += reward
        cumulative_regret += regret
        used += 1

        if online_update and policy.can_update_online:
            pending_updates.append((action, reward, features))
            if len(pending_updates) >= update_chunk:
                policy.update_batch(pending_updates)
                pending_updates.clear()

        history_rows.append(
            {
                "step": step,
                "reward": reward,
                "avg_reward": total_reward / used,
                "ips_reward": ips_reward,
                "ips_avg_reward": ips_weighted_reward_sum / step,
                "cumulative_regret": cumulative_regret,
                "avg_regret": cumulative_regret / used,
                "cumulative_ips_regret": cumulative_ips_regret,
                "avg_ips_regret": cumulative_ips_regret / step,
            }
        )

        if pbar is not None and step >= next_progress_mark:
            pbar.update(step - pbar.n)
            next_progress_mark += progress_chunk

    if pending_updates:
        policy.update_batch(pending_updates)

    if pbar is not None:
        pbar.update(test_df.height - pbar.n)
        pbar.close()

    ctr = total_reward / used if used else 0.0
    ips_ctr = ips_weighted_reward_sum / test_df.height if test_df.height else 0.0
    match_rate = replay_matches / test_df.height if test_df.height else 0.0
    final_avg_regret = (cumulative_regret / used) if used else 0.0
    final_avg_ips_regret = (cumulative_ips_regret / test_df.height) if test_df.height else 0.0
    metrics_df = pd.DataFrame([
        {
            "impressions_total": test_df.height,
            "impressions_used": used,
            "total_reward": total_reward,
            "ctr": ctr,
            "ips_weighted_reward": ips_weighted_reward_sum,
            "ips_ctr": ips_ctr,
            "replay_match_rate": match_rate,
            "global_random_ctr": global_random_ctr,
            "cumulative_regret": cumulative_regret,
            "avg_regret": final_avg_regret,
            "cumulative_ips_regret": cumulative_ips_regret,
            "avg_ips_regret": final_avg_ips_regret,
        }
    ])
    history_df = pd.DataFrame(history_rows)
    return metrics_df, history_df


def run_scenarios(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    policy_factories: dict[str, Callable[[], BasePolicy]],
    scenarios: list[ScenarioConfig],
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    metrics_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []

    for scenario in scenarios:
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)
        ctr_by_action, global_random_ctr = build_random_action_ctr_stats(train_df)

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_df.height > 0:
                policy.fit(pretrain_df)

            metrics_df, history_df = evaluate_policy(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                env_reward=env_reward,
                show_progress=show_progress,
                progress_desc=f"{scenario.name}/{algo_name}",
                ctr_by_action=ctr_by_action,
                global_random_ctr=global_random_ctr,
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


def default_five_ips_scenarios() -> list[ScenarioConfig]:
    """Same 5 scenarios, but intended for IPS evaluation on random-policy test slice."""
    return [
        ScenarioConfig("ips_case_1_random_pretrain_predict_only", "random", False),
        ScenarioConfig("ips_case_2_random_pretrain_online_update", "random", True),
        ScenarioConfig("ips_case_3_all_pretrain_predict_only", "all", False),
        ScenarioConfig("ips_case_4_all_pretrain_online_update", "all", True),
        ScenarioConfig("ips_case_5_no_pretrain_online_update", "none", True),
    ]
