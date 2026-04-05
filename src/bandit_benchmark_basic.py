"""Core benchmark types and simple bandit policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import random
from typing import Literal

import polars as pl

Action = str


def normalize_action(value: object) -> Action:
    """Normalize action ids so datasets can contain ints or strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return str(int(value))
    return str(value)


@dataclass
class ScenarioConfig:
    name: str
    pretrain_source: Literal["random", "all", "none"]
    online_update: bool
    update_frequency: Literal["daily", "step_2p5"] = "daily"

class BasePolicy:
    can_update_online: bool = True

    def __init__(self, can_update_online: bool | None = None):
        if can_update_online is not None:
            self.can_update_online = can_update_online

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del candidates, features, row
        raise NotImplementedError

    def select_batch(
        self,
        candidates_batch: list[list[Action]],
        features_batch: list[list[float]],
        rows_batch: list[dict[str, object]],
    ) -> list[Action]:
        if not (len(candidates_batch) == len(features_batch) == len(rows_batch)):
            raise ValueError("Batch inputs must have equal length")
        return [
            normalize_action(self.select(candidates, features, row))
            for candidates, features, row in zip(candidates_batch, features_batch, rows_batch)
        ]

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        normalized_action = normalize_action(action)
        if not candidates or normalized_action not in candidates:
            return 0.0
        if features is None:
            return 0.0
        return 1.0 if normalize_action(self.select(candidates, features, row or {})) == normalized_action else 0.0

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del action, reward, features

    def update_batch(self, pending_updates: list[tuple[Action, float, list[float]]]) -> None:
        for a, r, f in pending_updates:
            self.update(a, r, f)

    def fit(self, train_df: pl.DataFrame) -> None:
        pending_updates = [
            (normalize_action(r["show"]), float(r["reward"]), r["features_list"])
            for r in train_df.iter_rows(named=True)
        ]
        self.update_batch(pending_updates)

class RandomPolicy(BasePolicy):
    can_update_online = False

    def __init__(self, seed: int = 42, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
        self.rng = random.Random(seed)

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del features, row
        if not candidates:
            raise ValueError("Empty candidate set")
        return normalize_action(self.rng.choice(candidates))

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        del features, row
        if not candidates or normalize_action(action) not in candidates:
            return 0.0
        return 1.0 / len(candidates)

class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 0.1, seed: int = 42, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
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

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        del features, row
        normalized_action = normalize_action(action)
        if not candidates or normalized_action not in candidates:
            return 0.0
        uniform_p = self.epsilon / len(candidates)
        best_value = max(self.values.get(normalize_action(a), 0.0) for a in candidates)
        greedy_actions = [normalize_action(a) for a in candidates if self.values.get(normalize_action(a), 0.0) == best_value]
        exploit_p = (1.0 - self.epsilon) / max(1, len(greedy_actions)) if normalized_action in greedy_actions else 0.0
        return uniform_p + exploit_p

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        n = self.counts.get(action, 0) + 1
        v = self.values.get(action, 0.0)
        self.values[action] = v + (reward - v) / n
        self.counts[action] = n

class UCBPolicy(BasePolicy):
    def __init__(self, exploration: float = 2.0, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
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

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        del features, row
        normalized_action = normalize_action(action)
        if not candidates or normalized_action not in candidates:
            return 0.0
        untried = [normalize_action(a) for a in candidates if self.counts.get(normalize_action(a), 0) == 0]
        if untried:
            return 1.0 if normalized_action == untried[0] else 0.0
        chosen = normalize_action(max(candidates, key=lambda a: self.values.get(normalize_action(a), 0.0) + math.sqrt(self.exploration * math.log(max(self.t, 1)) / max(1, self.counts.get(normalize_action(a), 1)))))
        return 1.0 if normalized_action == chosen else 0.0

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        self.t += 1
        n = self.counts.get(action, 0) + 1
        v = self.values.get(action, 0.0)
        self.values[action] = v + (reward - v) / n
        self.counts[action] = n

class ThompsonSamplingPolicy(BasePolicy):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, seed: int = 42, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
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

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        import numpy as np

        del features, row
        normalized_action = normalize_action(action)
        if not candidates or normalized_action not in candidates:
            return 0.0
        n_mc = 256
        rng = np.random.default_rng(42)
        wins = 0
        target = normalized_action
        for _ in range(n_mc):
            draws = {
                normalize_action(a): float(rng.beta(self.alpha.get(normalize_action(a), self.alpha0), self.beta.get(normalize_action(a), self.beta0)))
                for a in candidates
            }
            best = max(draws.items(), key=lambda kv: kv[1])[0]
            wins += int(best == target)
        return wins / n_mc

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        self.alpha[action] = self.alpha.get(action, self.alpha0) + reward
        self.beta[action] = self.beta.get(action, self.beta0) + (1.0 - reward)
