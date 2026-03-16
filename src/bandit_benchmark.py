"""Benchmark helpers for logged ad bandit data (polars processing + pandas metrics)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Literal

import pandas as pd
import polars as pl

from prepare_datasets import (
    apply_standard_scaler_to_features,
    filter_test_by_train_candidate_coverage,
    preprocess_bandit_dataframe,
    split_train_test_by_date,
)

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

Action = int


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
            int(self.select(candidates, features, row))
            for candidates, features, row in zip(candidates_batch, features_batch, rows_batch)
        ]

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        if not candidates or int(action) not in candidates:
            return 0.0
        if features is None:
            return 0.0
        return 1.0 if int(self.select(candidates, features, row or {})) == int(action) else 0.0

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del action, reward, features

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        for a, r, f in pending_updates:
            self.update(a, r, f)

    def fit(self, train_df: pl.DataFrame) -> None:
        pending_updates = [
            (int(r["show"]), float(r["reward"]), r["features_list"])
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
        return int(self.rng.choice(candidates))

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        del features, row
        if not candidates or int(action) not in candidates:
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
        if not candidates or int(action) not in candidates:
            return 0.0
        uniform_p = self.epsilon / len(candidates)
        best_value = max(self.values.get(int(a), 0.0) for a in candidates)
        greedy_actions = [int(a) for a in candidates if self.values.get(int(a), 0.0) == best_value]
        exploit_p = (1.0 - self.epsilon) / max(1, len(greedy_actions)) if int(action) in greedy_actions else 0.0
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
        if not candidates or int(action) not in candidates:
            return 0.0
        untried = [int(a) for a in candidates if self.counts.get(int(a), 0) == 0]
        if untried:
            return 1.0 if int(action) == untried[0] else 0.0
        chosen = int(max(candidates, key=lambda a: self.values.get(int(a), 0.0) + math.sqrt(self.exploration * math.log(max(self.t, 1)) / max(1, self.counts.get(int(a), 1)))))
        return 1.0 if int(action) == chosen else 0.0

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
        if not candidates or int(action) not in candidates:
            return 0.0
        n_mc = 256
        rng = np.random.default_rng(42)
        wins = 0
        target = int(action)
        for _ in range(n_mc):
            draws = {
                int(a): float(rng.beta(self.alpha.get(int(a), self.alpha0), self.beta.get(int(a), self.beta0)))
                for a in candidates
            }
            best = max(draws.items(), key=lambda kv: kv[1])[0]
            wins += int(best == target)
        return wins / n_mc

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del features
        self.alpha[action] = self.alpha.get(action, self.alpha0) + reward
        self.beta[action] = self.beta.get(action, self.beta0) + (1.0 - reward)


class OnlineLogisticRegression:
    """Online Logistic Regression with diagonal precision q (Laplace-like)."""

    def __init__(self, lambda_: float, alpha: float, n_dim: int, seed: int | None = None):
        import numpy as np

        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self.n_dim = int(n_dim)

        self.m = np.zeros(self.n_dim, dtype=np.float64)
        self.q = np.ones(self.n_dim, dtype=np.float64) * self.lambda_

        self.rng = np.random.default_rng(seed)
        self.w = self.get_weights()

    def loss(self, w, X, y) -> float:
        import numpy as np

        prior = 0.5 * (self.q * (w - self.m)).dot(w - self.m)
        z = y * (X @ w)
        ll = np.logaddexp(0.0, -z).sum()
        return float(prior + ll)

    def grad(self, w, X, y):
        from scipy.special import expit

        g = self.q * (w - self.m)
        z = y * (X @ w)
        coeff = -y * expit(-z)
        g += (coeff[:, None] * X).sum(axis=0)
        return g

    def get_weights(self):
        import numpy as np

        std = self.alpha / np.sqrt(np.maximum(self.q, 1e-12))
        return self.rng.normal(loc=self.m, scale=std, size=self.n_dim)

    def fit(self, X, y, maxiter: int = 20) -> None:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import expit

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != self.n_dim:
            raise ValueError(f"X must be (n,{self.n_dim}), got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be (n,) aligned with X")

        res = minimize(
            fun=self.loss,
            x0=self.m,
            args=(X, y),
            jac=self.grad,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter), "disp": False},
        )
        self.w = res.x.astype(np.float64, copy=False)
        self.m = self.w.copy()

        p = expit(X @ self.m)
        v = p * (1.0 - p)
        self.q = self.q + (v[:, None] * (X * X)).sum(axis=0)

    def predict_proba(self, X, mode: str = "sample"):
        import numpy as np
        from scipy.special import expit

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if mode == "sample":
            w = self.get_weights()
        elif mode == "expected":
            w = self.m
        else:
            raise ValueError("mode not recognized: use 'sample' or 'expected'")

        p = expit(X @ w)
        return np.vstack([1.0 - p, p]).T




class ActionTreeThompsonModel:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 300,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        c_min: int = 5,
        random_state: int = 42,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.c_min = int(c_min)
        self.random_state = int(random_state)

        self.tree = None
        self.leaf_stats: dict[int, dict[str, float | int]] = {}
        self.global_alpha: float | None = None
        self.global_beta: float | None = None
        import numpy as np
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X, y):
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        self.tree = DecisionTreeClassifier(
            criterion="log_loss",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.tree.fit(X, y)

        leaf_ids = self.tree.apply(X)
        total_clicks = int(y.sum())
        total_n = int(len(y))
        self.global_alpha = self.alpha0 + total_clicks
        self.global_beta = self.beta0 + (total_n - total_clicks)

        self.leaf_stats = {}
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            n = int(mask.sum())
            c = int(y[mask].sum())
            self.leaf_stats[int(leaf)] = {
                "n": n,
                "clicks": c,
                "alpha": self.alpha0 + c,
                "beta": self.beta0 + (n - c),
            }
        return self

    def _ensure_fitted(self):
        if self.tree is None:
            raise RuntimeError("Model is not fitted yet.")

    def _to_2d(self, X):
        import numpy as np

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _leaf_ids(self, X):
        self._ensure_fitted()
        X = self._to_2d(X)
        return self.tree.apply(X)

    def _effective_stats_for_leaf(self, leaf_id: int) -> dict[str, float | int | bool]:
        leaf_id = int(leaf_id)
        stats = self.leaf_stats.get(leaf_id)

        if stats is None:
            return {
                "n": 0,
                "clicks": 0,
                "alpha": self.global_alpha,
                "beta": self.global_beta,
                "used_global_fallback": True,
            }

        if int(stats["clicks"]) < self.c_min:
            return {
                "n": stats["n"],
                "clicks": stats["clicks"],
                "alpha": self.global_alpha,
                "beta": self.global_beta,
                "used_global_fallback": True,
            }

        return {
            "n": stats["n"],
            "clicks": stats["clicks"],
            "alpha": stats["alpha"],
            "beta": stats["beta"],
            "used_global_fallback": False,
        }

    def get_leaf_stats(self, X):
        leaf_ids = self._leaf_ids(X)
        return [self._effective_stats_for_leaf(int(leaf)) for leaf in leaf_ids]

    def sample_proba(self, X, n_samples: int = 1, random_state: int | None = None):
        import numpy as np

        rng = np.random.default_rng(random_state) if random_state is not None else self._rng
        stats = self.get_leaf_stats(X)
        alpha = np.array([float(s["alpha"]) for s in stats], dtype=float)
        beta = np.array([float(s["beta"]) for s in stats], dtype=float)

        if n_samples == 1:
            return rng.beta(alpha, beta)

        return rng.beta(
            np.broadcast_to(alpha, (n_samples, len(alpha))),
            np.broadcast_to(beta, (n_samples, len(beta))),
        )

    def update_batch(self, X, y):
        import numpy as np

        self._ensure_fitted()
        X = self._to_2d(X)
        y = np.asarray(y).astype(int)

        leaf_ids = self.tree.apply(X)
        for leaf, reward in zip(leaf_ids, y):
            leaf = int(leaf)
            if leaf not in self.leaf_stats:
                self.leaf_stats[leaf] = {
                    "n": 0,
                    "clicks": 0,
                    "alpha": self.alpha0,
                    "beta": self.beta0,
                }

            self.leaf_stats[leaf]["n"] = int(self.leaf_stats[leaf]["n"]) + 1
            self.leaf_stats[leaf]["clicks"] = int(self.leaf_stats[leaf]["clicks"]) + int(reward)
            self.leaf_stats[leaf]["alpha"] = float(self.leaf_stats[leaf]["alpha"]) + int(reward)
            self.leaf_stats[leaf]["beta"] = float(self.leaf_stats[leaf]["beta"]) + int(1 - reward)

        self.global_alpha = float(self.global_alpha) + int(y.sum())
        self.global_beta = float(self.global_beta) + int(len(y) - y.sum())


class TreeThompsonSamplingPolicy(BasePolicy):
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 300,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        c_min: int = 5,
        random_state: int = 42,
        can_update_online: bool | None = True,
    ):
        super().__init__(can_update_online=can_update_online)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.c_min = int(c_min)
        self.random_state = int(random_state)

        self.action_models: dict[int, ActionTreeThompsonModel] = {}
        # per-action storage of (features, reward, action)
        self.action_history: dict[int, list[tuple[list[float], float, int]]] = {}

    def _build_model(self) -> ActionTreeThompsonModel:
        return ActionTreeThompsonModel(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            alpha0=self.alpha0,
            beta0=self.beta0,
            c_min=self.c_min,
            random_state=self.random_state,
        )

    def fit(self, train_df: pl.DataFrame) -> None:
        import numpy as np

        rows = list(train_df.iter_rows(named=True))
        self.action_models = {}
        self.action_history = {}

        for r in rows:
            a = int(r["show"])
            feat = [float(v) for v in r["features_list"]]
            rew = float(r["reward"])
            self.action_history.setdefault(a, []).append((feat, rew, a))

        for a, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            model = self._build_model().fit(X, y)
            self.action_models[a] = model

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        import numpy as np

        del row
        if not candidates:
            raise ValueError("Empty candidate set")

        x = np.asarray(features, dtype=float)
        best_a = int(candidates[0])
        best_score = -1.0

        rng = np.random.default_rng(self.random_state)
        for a in candidates:
            aa = int(a)
            model = self.action_models.get(aa)
            if model is None or model.tree is None:
                score = float(rng.beta(self.alpha0, self.beta0))
            else:
                score = float(model.sample_proba(x, n_samples=1)[0])
            if score > best_score:
                best_score = score
                best_a = aa
        return best_a

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        import numpy as np

        del row
        if features is None or not candidates or int(action) not in candidates:
            return 0.0
        x = np.asarray(features, dtype=float)
        n_mc = 128
        wins = 0
        target = int(action)
        rng = np.random.default_rng(self.random_state)
        for _ in range(n_mc):
            best_a = int(candidates[0])
            best_score = -1.0
            for a in candidates:
                aa = int(a)
                model = self.action_models.get(aa)
                if model is None or model.tree is None:
                    score = float(rng.beta(self.alpha0, self.beta0))
                else:
                    score = float(model.sample_proba(x, n_samples=1)[0])
                if score > best_score:
                    best_score = score
                    best_a = aa
            wins += int(best_a == target)
        return wins / n_mc

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        import numpy as np

        if not pending_updates:
            return

        for a, r, f in pending_updates:
            aa = int(a)
            ff = [float(v) for v in f]
            rr = float(r)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        grouped: dict[int, list[tuple[list[float], float, int]]] = {}
        for a, r, f in pending_updates:
            grouped.setdefault(int(a), []).append(([float(v) for v in f], float(r), int(a)))

        for a, items in grouped.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if a in self.action_models and self.action_models[a].tree is not None:
                self.action_models[a].update_batch(X, y)
            else:
                # new action: fit tree on all stored samples for this action
                all_items = self.action_history.get(a, items)
                Xa = np.asarray([it[0] for it in all_items], dtype=float)
                ya = np.asarray([1 if it[1] > 0 else 0 for it in all_items], dtype=int)
                if len(Xa) == 0:
                    continue
                self.action_models[a] = self._build_model().fit(Xa, ya)


class TreeThompsonSamplingPolicyUpdateV1(TreeThompsonSamplingPolicy):
    """Incremental tree updates via ActionTreeThompsonModel.update_batch."""


class TreeThompsonSamplingPolicyDummyRefit(TreeThompsonSamplingPolicy):
    """Refits per-action trees on full stored history at each update."""

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        import numpy as np

        if not pending_updates:
            return

        for a, r, f in pending_updates:
            aa = int(a)
            ff = [float(v) for v in f]
            rr = float(r)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        for a, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            self.action_models[a] = self._build_model().fit(X, y)


class LaplaceThompsonViaBayesianLogRegPolicy(BasePolicy):
    """Per-action Bayesian online logistic TS with Laplace-style diagonal precision."""

    can_update_online: bool = True

    def __init__(
        self,
        lambda_: float = 1.0,
        alpha: float = 1.0,
        maxiter_update: int = 5,
        maxiter_batch: int = 20,
        maxiter_fit: int = 50,
        seed: int | None = None,
        can_update_online: bool | None = None,
    ) -> None:
        super().__init__(can_update_online=can_update_online)
        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self.maxiter_update = int(maxiter_update)
        self.maxiter_batch = int(maxiter_batch)
        self.maxiter_fit = int(maxiter_fit)
        self.seed = seed

        self._d: int | None = None
        self._models: dict[int, OnlineLogisticRegression] = {}

    def _get_model(self, a: int) -> OnlineLogisticRegression:
        m = self._models.get(a)
        if m is None:
            if self._d is None:
                raise ValueError("Feature dimension is unknown; call update/select with features first")
            arm_seed = None if self.seed is None else (self.seed + 1000003 * a)
            m = OnlineLogisticRegression(self.lambda_, self.alpha, self._d, seed=arm_seed)
            self._models[a] = m
        return m

    def _ensure_dim(self, features: list[float]) -> int:
        d = len(features)
        if self._d is None:
            self._d = d
        elif self._d != d:
            raise ValueError(f"Feature dimension changed: expected {self._d}, got {d}")
        return d

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        import numpy as np

        if features is None:
            return
        self._ensure_dim(features)
        a = int(action)
        model = self._get_model(a)

        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        y = np.asarray([1 if float(reward) > 0 else -1], dtype=np.int64)
        model.fit(x, y, maxiter=self.maxiter_update)

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        import numpy as np

        if not pending_updates:
            return

        self._ensure_dim(pending_updates[0][2])
        by_arm: dict[int, tuple[list, list[int]]] = {}
        for a, r, f in pending_updates:
            arm = int(a)
            x = np.asarray(f, dtype=np.float64)
            y = 1 if float(r) > 0 else -1
            if arm not in by_arm:
                by_arm[arm] = ([], [])
            by_arm[arm][0].append(x)
            by_arm[arm][1].append(y)

        for arm, (X_list, y_list) in by_arm.items():
            model = self._get_model(arm)
            X = np.vstack(X_list)
            y = np.asarray(y_list, dtype=np.int64)
            model.fit(X, y, maxiter=self.maxiter_batch)

    def fit(self, train_df: pl.DataFrame) -> None:
        import numpy as np

        pending_updates = [
            (int(r["show"]), float(r["reward"]), list(r["features_list"]))
            for r in train_df.iter_rows(named=True)
        ]
        if not pending_updates:
            return

        self._ensure_dim(pending_updates[0][2])
        by_arm: dict[int, tuple[list, list[int]]] = {}
        for a, r, f in pending_updates:
            arm = int(a)
            x = np.asarray(f, dtype=np.float64)
            y = 1 if float(r) > 0 else -1
            if arm not in by_arm:
                by_arm[arm] = ([], [])
            by_arm[arm][0].append(x)
            by_arm[arm][1].append(y)

        for arm, (X_list, y_list) in by_arm.items():
            model = self._get_model(arm)
            X = np.vstack(X_list)
            y = np.asarray(y_list, dtype=np.int64)
            model.fit(X, y, maxiter=self.maxiter_fit)

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        import numpy as np

        del row
        if not candidates:
            raise ValueError("candidates is empty")
        self._ensure_dim(features)

        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        best_a = int(candidates[0])
        best_score = -np.inf

        for a_raw in candidates:
            a = int(a_raw)
            model = self._get_model(a)
            p = float(model.predict_proba(x, mode="sample")[0, 1])
            if p > best_score:
                best_score = p
                best_a = a

        return best_a

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        import numpy as np

        del row
        if features is None or not candidates or int(action) not in candidates:
            return 0.0
        self._ensure_dim(features)
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        n_mc = 128
        wins = 0
        target = int(action)
        for _ in range(n_mc):
            best_a = int(candidates[0])
            best_score = -np.inf
            for a_raw in candidates:
                a = int(a_raw)
                model = self._get_model(a)
                p = float(model.predict_proba(x, mode="sample")[0, 1])
                if p > best_score:
                    best_score = p
                    best_a = a
            wins += int(best_a == target)
        return wins / n_mc


class _NeuralActionRewardEncoder:
    """Simple MLP encoder + action logits head trained with BCE on logged action."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: list[int],
        rep_dim: int,
        lr: float,
        seed: int | None,
    ):
        import torch
        import torch.nn as nn

        if seed is not None:
            torch.manual_seed(seed)

        dims = [input_dim] + hidden_dims
        trunk_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            trunk_layers.append(nn.ReLU())

        if hidden_dims:
            trunk_out = hidden_dims[-1]
        else:
            trunk_out = input_dim

        self.trunk = nn.Sequential(*trunk_layers)
        self.rep_layer = nn.Linear(trunk_out, rep_dim)
        self.head = nn.Linear(rep_dim, num_actions)

        self.optimizer = torch.optim.RMSprop(
            list(self.trunk.parameters()) + list(self.rep_layer.parameters()) + list(self.head.parameters()),
            lr=lr,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_encoder(
        self,
        X,
        action_idx,
        reward,
        epochs: int,
        batch_size: int,
        val_fraction: float = 0.2,
        early_stopping_patience: int = 5,
        min_delta: float = 1e-4,
    ) -> None:
        import torch

        X_t = torch.as_tensor(X, dtype=torch.float32)
        a_t = torch.as_tensor(action_idx, dtype=torch.long)
        r_t = torch.as_tensor(reward, dtype=torch.float32)
        n = X_t.shape[0]
        if n == 0:
            return

        all_idx = torch.randperm(n)
        val_size = 0
        if n > 1 and val_fraction > 0:
            val_size = max(1, min(n - 1, int(round(n * float(val_fraction)))))

        val_idx = all_idx[:val_size]
        train_idx = all_idx[val_size:] if val_size > 0 else all_idx
        if train_idx.numel() == 0:
            train_idx = all_idx
            val_idx = all_idx[:0]
            val_size = 0

        best_val_loss = float("inf")
        best_state: dict[str, dict[str, torch.Tensor]] | None = None
        patience_left = max(1, int(early_stopping_patience))

        for _ in range(max(1, int(epochs))):
            perm = train_idx[torch.randperm(train_idx.numel())]
            for st in range(0, perm.numel(), max(1, int(batch_size))):
                idx = perm[st : st + max(1, int(batch_size))]
                xb = X_t[idx]
                ab = a_t[idx]
                rb = r_t[idx]

                z = self.trunk(xb)
                rep = self.rep_layer(z)
                logits = self.head(rep)
                chosen_logits = logits.gather(1, ab.view(-1, 1)).squeeze(1)
                loss = self.loss_fn(chosen_logits, rb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if val_size > 0:
                with torch.no_grad():
                    vb = X_t[val_idx]
                    vab = a_t[val_idx]
                    vrb = r_t[val_idx]
                    vz = self.trunk(vb)
                    vrep = self.rep_layer(vz)
                    vlogits = self.head(vrep)
                    vchosen = vlogits.gather(1, vab.view(-1, 1)).squeeze(1)
                    val_loss = float(self.loss_fn(vchosen, vrb).item())

                if val_loss < (best_val_loss - float(min_delta)):
                    best_val_loss = val_loss
                    best_state = {
                        "trunk": {k: v.detach().cpu().clone() for k, v in self.trunk.state_dict().items()},
                        "rep_layer": {k: v.detach().cpu().clone() for k, v in self.rep_layer.state_dict().items()},
                        "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()},
                    }
                    patience_left = max(1, int(early_stopping_patience))
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            self.trunk.load_state_dict(best_state["trunk"])
            self.rep_layer.load_state_dict(best_state["rep_layer"])
            self.head.load_state_dict(best_state["head"])

    def transform(self, X):
        import torch

        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32)
            z = self.trunk(x)
            rep = self.rep_layer(z)
        return rep.cpu().numpy()


class NeuralLaplaceThompsonViaBayesianLogRegPolicy(BasePolicy):
    """Laplace TS over neural representations; NN trains only in fit()."""

    can_update_online: bool = True

    def __init__(
        self,
        lambda_: float = 1.0,
        alpha: float = 1.0,
        maxiter_update: int = 5,
        maxiter_batch: int = 20,
        maxiter_fit: int = 50,
        hidden_dims: list[int] | None = None,
        encoder: _NeuralActionRewardEncoder | None = None,
        rep_dim: int = 32,
        nn_lr: float = 1e-3,
        nn_epochs: int = 10,
        nn_batch_size: int = 256,
        encoder_train_data_mode: Literal["all", "random_half", "time_half"] = "all",
        seed: int | None = None,
        can_update_online: bool | None = None,
    ) -> None:
        super().__init__(can_update_online=can_update_online)
        self.hidden_dims = hidden_dims or [64, 32]
        self.rep_dim = int(rep_dim)
        self.nn_lr = float(nn_lr)
        self._provided_encoder = encoder
        self.nn_epochs = int(nn_epochs)
        self.nn_batch_size = int(nn_batch_size)
        self.encoder_train_data_mode = encoder_train_data_mode
        self.seed = seed

        self._encoder: _NeuralActionRewardEncoder | None = None
        self._encoder_trained = False
        self._action_to_idx: dict[int, int] = {}

        self._base = LaplaceThompsonViaBayesianLogRegPolicy(
            lambda_=lambda_,
            alpha=alpha,
            maxiter_update=maxiter_update,
            maxiter_batch=maxiter_batch,
            maxiter_fit=maxiter_fit,
            seed=seed,
            can_update_online=can_update_online,
        )

    def _transform_features(self, features: list[float]) -> list[float]:
        import numpy as np

        if not self._encoder_trained or self._encoder is None:
            return list(features)
        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
        rep = self._encoder.transform(arr)[0]
        return [float(v) for v in rep.tolist()]

    def fit(self, train_df: pl.DataFrame) -> None:
        import numpy as np

        rows = list(train_df.iter_rows(named=True))
        if not rows:
            return

        actions = sorted({int(r["show"]) for r in rows})
        self._action_to_idx = {a: i for i, a in enumerate(actions)}

        mode = self.encoder_train_data_mode
        if mode not in {"all", "random_half", "time_half"}:
            raise ValueError(f"Unknown encoder_train_data_mode: {mode}")

        encoder_rows = rows
        reg_rows = rows
        n_rows = len(rows)

        if mode == "random_half" and n_rows > 1:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n_rows)
            split = max(1, n_rows // 2)
            enc_idx = perm[:split]
            reg_idx = perm[split:]
            encoder_rows = [rows[int(i)] for i in enc_idx.tolist()]
            reg_rows = [rows[int(i)] for i in reg_idx.tolist()] if len(reg_idx) > 0 else encoder_rows
        elif mode == "time_half" and n_rows > 1:
            ordered_rows = sorted(rows, key=lambda r: r.get("date"))
            split = max(1, n_rows // 2)
            encoder_rows = ordered_rows[:split]
            reg_rows = ordered_rows[split:] if split < n_rows else encoder_rows

        X_enc = np.asarray([list(r["features_list"]) for r in encoder_rows], dtype=np.float32)
        a_idx_enc = np.asarray([self._action_to_idx[int(r["show"])] for r in encoder_rows], dtype=np.int64)
        y_enc = np.asarray([1.0 if float(r["reward"]) > 0 else 0.0 for r in encoder_rows], dtype=np.float32)

        if self._provided_encoder is not None:
            self._encoder = self._provided_encoder
        else:
            self._encoder = _NeuralActionRewardEncoder(
                input_dim=X_enc.shape[1],
                num_actions=len(actions),
                hidden_dims=self.hidden_dims,
                rep_dim=self.rep_dim,
                lr=self.nn_lr,
                seed=self.seed,
            )
        self._encoder.train_encoder(
            X_enc,
            a_idx_enc,
            y_enc,
            epochs=self.nn_epochs,
            batch_size=self.nn_batch_size,
        )
        self._encoder_trained = True

        X_reg = np.asarray([list(r["features_list"]) for r in reg_rows], dtype=np.float32)
        Z_reg = self._encoder.transform(X_reg)
        transformed_updates = [
            (int(r["show"]), float(r["reward"]), [float(v) for v in Z_reg[i].tolist()])
            for i, r in enumerate(reg_rows)
        ]

        by_arm: dict[int, tuple[list, list[int]]] = {}
        for a, rew, feat in transformed_updates:
            arm = int(a)
            x = np.asarray(feat, dtype=np.float64)
            yy = 1 if float(rew) > 0 else -1
            if arm not in by_arm:
                by_arm[arm] = ([], [])
            by_arm[arm][0].append(x)
            by_arm[arm][1].append(yy)

        self._base._ensure_dim(transformed_updates[0][2])
        for arm, (X_list, y_list) in by_arm.items():
            model = self._base._get_model(arm)
            X_arm = np.vstack(X_list)
            y_arm = np.asarray(y_list, dtype=np.int64)
            model.fit(X_arm, y_arm, maxiter=self._base.maxiter_fit)

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        if features is None:
            return
        self._base.update(action, reward, self._transform_features(features))

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        transformed = [(a, r, self._transform_features(f)) for a, r, f in pending_updates]
        self._base.update_batch(transformed)

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        return self._base.select(candidates, self._transform_features(features), row)

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        if features is None or not candidates or int(action) not in candidates:
            return 0.0
        transformed = self._transform_features(features)
        return self._base.get_action_proba(candidates, action, transformed, row)

class ContextualBanditPlaceholder(BasePolicy):
    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        del candidates, features, row
        raise NotImplementedError("Contextual bandits are intentionally not implemented yet")




class _ContextualTSLibPolicyBase(BasePolicy):
    def __init__(self, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
        self._model = None
        self._fitted = False
        self._actions: list[int] = []
        self._a2i: dict[int, int] = {}

    def _build_action_index(self, rows: list[dict[str, object]]) -> None:
        actions = sorted({int(r["show"]) for r in rows})
        self._actions = actions
        self._a2i = {a: i for i, a in enumerate(actions)}

    def _select_from_score_vector(self, candidates: list[Action], scores: list[float]) -> Action:
        if not candidates:
            raise ValueError("Empty candidate set")
        best_action = candidates[0]
        best_score = -1e18
        for a in candidates:
            idx = self._a2i.get(int(a))
            score = float(scores[idx]) if idx is not None else 0.0
            if score > best_score:
                best_score = score
                best_action = int(a)
        return best_action


class LogisticTSLibPolicy(_ContextualTSLibPolicyBase):
    """Wrapper over contextualbandits.online.LogisticTS."""

    can_update_online = False

    def __init__(self, random_seed: int = 42, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
        self.random_seed = random_seed
        self.a: list[int] = []
        self.r: list[int] = []
        self.f: list[list[float]] = []

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        import numpy as np
        try:
            from contextualbandits.online import LogisticTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("contextualbandits is required for LogisticTSLibPolicy") from exc

        new_actions = {int(a) for a, _, _ in pending_updates}
        if not new_actions:
            raise ValueError("pending_updates contains no actions")

        for a in new_actions:
            if a not in self._a2i:
                self._a2i[a] = len(self._actions)
                self._actions.append(a)

        for a, r, f in pending_updates:
            self.a.append(self._a2i[a])
            self.r.append(int(float(r) > 0.0))
            self.f.append(f)

        self._model = LogisticTS(nchoices=len(self._actions), random_state=self.random_seed)
        self._model.fit(np.array(self.f)[:, :50], np.array(self.a), np.array(self.r))

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        import numpy as np

        del row
        if not candidates:
            raise ValueError("Empty candidate set")
        if self._model is None:
            return int(np.random.choice(candidates))

        ids = [self._a2i[candidate] for candidate in candidates if self._a2i.get(candidate) is not None]
        if len(ids) == 0:
            return int(np.random.choice(candidates))
        probs = self._model.predict(np.array(features[:50]), output_all_scores=True)
        idx_max = probs["scores"][0][ids].argmax()
        best_action = self._actions[ids[idx_max]]
        return int(best_action)

    def select_batch(
        self,
        candidates_batch: list[list[Action]],
        features_batch: list[list[float]],
        rows_batch: list[dict[str, object]],
    ) -> list[Action]:
        import numpy as np

        if not (len(candidates_batch) == len(features_batch) == len(rows_batch)):
            raise ValueError("Batch inputs must have equal length")
        if len(candidates_batch) == 0:
            return []

        del rows_batch

        if self._model is None:
            out_random: list[Action] = []
            for cands in candidates_batch:
                if not cands:
                    raise ValueError("Empty candidate set in batch")
                out_random.append(int(np.random.choice(cands)))
            return out_random

        X = np.asarray([f[:50] for f in features_batch], dtype=np.float64)
        probs = self._model.predict(X, output_all_scores=True)
        scores = probs["scores"]

        out: list[Action] = []
        for i, candidates in enumerate(candidates_batch):
            if not candidates:
                raise ValueError("Empty candidate set in batch")
            ids = [self._a2i[candidate] for candidate in candidates if self._a2i.get(candidate) is not None]
            if not ids:
                out.append(int(np.random.choice(candidates)))
                continue

            row_scores = scores[i]
            best_local = int(np.argmax(row_scores[ids]))
            out.append(int(self._actions[ids[best_local]]))

        return out


class PartitionedTSLibPolicy(_ContextualTSLibPolicyBase):
    """Wrapper over contextualbandits.online.PartitionedTS."""

    can_update_online = True

    def __init__(self, random_seed: int = 42, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
        self.random_seed = random_seed
        self.a: list[int] = []
        self.r: list[int] = []
        self.f: list[list[float]] = []

    def update_batch(self, pending_updates: list[tuple[int, float, list[float]]]) -> None:
        import numpy as np
        try:
            from contextualbandits.online import PartitionedTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("contextualbandits is required for PartitionedTSLibPolicy") from exc

        new_actions = {int(a) for a, _, _ in pending_updates}
        if not new_actions:
            raise ValueError("pending_updates contains no actions")

        for a in new_actions:
            if a not in self._a2i:
                self._a2i[a] = len(self._actions)
                self._actions.append(a)

        for a, r, f in pending_updates:
            self.a.append(self._a2i[a])
            self.r.append(int(float(r) > 0.0))
            self.f.append(f)

        self._model = PartitionedTS(nchoices=len(self._actions), random_state=self.random_seed)
        self._model.fit(np.array(self.f)[:, :50], np.array(self.a), np.array(self.r))

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        import numpy as np

        del row
        if not candidates:
            raise ValueError("Empty candidate set")
        if self._model is None:
            return int(np.random.choice(candidates))

        ids = [self._a2i[candidate] for candidate in candidates if self._a2i.get(candidate) is not None]
        if len(ids) == 0:
            return int(np.random.choice(candidates))
        probs = self._model.predict(np.array(features[:50]), output_all_scores=True)
        idx_max = probs["scores"][0][ids].argmax()
        best_action = self._actions[ids[idx_max]]

        return int(best_action)

    def select_batch(
        self,
        candidates_batch: list[list[Action]],
        features_batch: list[list[float]],
        rows_batch: list[dict[str, object]],
    ) -> list[Action]:
        import numpy as np

        if not (len(candidates_batch) == len(features_batch) == len(rows_batch)):
            raise ValueError("Batch inputs must have equal length")

        n = len(candidates_batch)
        if n == 0:
            return []

        del rows_batch

        if self._model is None:
            out_random: list[Action] = []
            for cands in candidates_batch:
                if not cands:
                    raise ValueError("Empty candidate set in batch")
                out_random.append(int(np.random.choice(cands)))
            return out_random

        X = np.asarray([f[:50] for f in features_batch], dtype=np.float64)
        probs = self._model.predict(X, output_all_scores=True)
        scores = probs["scores"]

        out: list[Action] = []
        for i, candidates in enumerate(candidates_batch):
            if not candidates:
                raise ValueError("Empty candidate set in batch")
            ids = [self._a2i[candidate] for candidate in candidates if self._a2i.get(candidate) is not None]
            if not ids:
                out.append(int(np.random.choice(candidates)))
                continue

            row_scores = scores[i]
            best_local = int(np.argmax(row_scores[ids]))
            best_action = self._actions[ids[best_local]]
            out.append(int(best_action))

        return out


# Backward-compatible aliases
TreeThompsonSamplingPolicyV1 = TreeThompsonSamplingPolicyUpdateV1
LogisticTSPolicy = LogisticTSLibPolicy
PartitionedTSPolicy = PartitionedTSLibPolicy


class CatBoostPolicy(BasePolicy):
    """Gradient boosting policy based on CatBoostClassifier.

    Model is train-once: repeated fit calls are forbidden.
    """

    can_update_online = False

    def __init__(self, random_seed: int = 42, can_update_online: bool | None = None):
        super().__init__(can_update_online=can_update_online)
        self.random_seed = random_seed
        self._model = None
        self._fitted = False
        self._actions: list[int] = []
        self._a2i: dict[int, int] = {}

    def _row_to_vector(self, features: list[float], action: int) -> list[float]:
        vec = list(features)
        one_hot = [0.0] * len(self._actions)
        idx = self._a2i.get(int(action))
        if idx is not None:
            one_hot[idx] = 1.0
        return vec + one_hot

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

        actions = sorted({int(r["show"]) for r in train_df.iter_rows(named=True)})
        self._actions = actions
        self._a2i = {a: i for i, a in enumerate(actions)}

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

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        del row
        if features is None or not candidates or int(action) not in candidates:
            return 0.0
        if self._model is None:
            return 1.0 if int(action) == int(candidates[0]) else 0.0
        scores = {}
        for a in candidates:
            vec = self._row_to_vector(features, int(a))
            scores[int(a)] = float(self._model.predict_proba([vec])[0][1])
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return 1.0 if int(action) == best else 0.0

    def update(self, action: Action, reward: float, features: list[float] | None = None) -> None:
        del action, reward, features
        return


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
    update_frequency: Literal["daily", "step_2p5"] = "daily",
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
    progress_desc: str = "evaluate",
    ctr_by_action: dict[int, float] | None = None,
    max_random_ctr: float = 0.0,
    initial_seen_actions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_reward = 0.0
    ips_weighted_reward_sum = 0.0
    snips_weight_sum = 0.0
    ips_sensitive_reward_sum = 0.0
    ips_sensitive_regret_sum = 0.0
    sensitive_rows = 0
    used = 0
    replay_matches = 0
    cumulative_regret = 0.0  # replay regret accumulator (used only for regret metrics)
    cumulative_ips_regret = 0.0  # IPS regret accumulator (used only for regret metrics)
    history_rows: list[dict[str, float | int]] = []
    action_stats_rows: list[dict[str, int]] = []
    selected_action_rows: list[dict[str, object]] = []
    action_sensitive_stats: dict[int, dict[str, float | int | bool]] = {}

    action_ctr = ctr_by_action or {}
    # Regret-only baseline fallback for unseen actions within candidate sets.

    total_steps = max(test_df.height, 1)
    progress_chunk = max(1, int(total_steps * 0.05))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=test_df.height, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=0.5)

    pending_updates: list[tuple[int, float, list[float]]] = []
    next_progress_mark = progress_chunk
    update_chunk = max(1, int(total_steps * 0.025))
    next_step_update_mark = update_chunk

    sensitive_seen = 0
    sensitive_ips_reward_cum = 0.0
    sensitive_ips_regret_cum = 0.0
    seen_actions_total: set[int] = set(initial_seen_actions or set())
    current_day_actions: set[int] = set()

    test_rows = list(test_df.iter_rows(named=True))
    first_row_date = test_rows[0].get("date") if test_rows else None
    prev_date = first_row_date

    if show_progress:
        msg = f"{progress_desc}: train_unique_actions={len(seen_actions_total)}"
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    batch_size = 512

    for batch_start in range(0, len(test_rows), batch_size):
        rows_batch = test_rows[batch_start : batch_start + batch_size]
        candidates_batch = [row["candidates_list"] for row in rows_batch]
        features_batch = [row["features_list"] for row in rows_batch]
        actions_batch = policy.select_batch(candidates_batch, features_batch, rows_batch)
        if len(actions_batch) != len(rows_batch):
            raise ValueError("select_batch must return one action per input row")

        for offset, row in enumerate(rows_batch):
            step = batch_start + offset + 1
            candidates = candidates_batch[offset]
            features = features_batch[offset]
            current_date = row.get("date")
            if prev_date is not None and current_date is not None and current_date > prev_date:
                if update_frequency == "daily" and online_update and policy.can_update_online and pending_updates:
                    policy.update_batch(pending_updates)
                    pending_updates.clear()

                new_actions_in_day = current_day_actions - seen_actions_total
                seen_actions_total.update(current_day_actions)
                action_stats_rows.append(
                    {
                        "step": step,
                        "date": prev_date,
                        "unique_actions_total": len(seen_actions_total),
                        "unique_actions_new_in_day": len(new_actions_in_day),
                    }
                )
                current_day_actions.clear()

            prev_date = current_date if current_date is not None else prev_date

            action = int(actions_batch[offset])
            current_day_actions.add(action)

            selected_action_rows.append(
                {
                    "step": step,
                    "date": current_date if current_date is not None else prev_date,
                    "action": action,
                }
            )

            logged_reward = float(row["reward"])
            logged_match = int(action == int(row["show"]))
            propensity = float(row.get("propensity", 0.0) or 0.0)

            ips_weight = (logged_match / propensity) if propensity > 0 else 0.0
            ips_reward = ips_weight * logged_reward
            ips_weighted_reward_sum += ips_reward
            snips_weight_sum += ips_weight
            # Regret-only per-step baseline: max expected CTR among currently available actions.
            candidate_ctrs = [action_ctr.get(int(a), max_random_ctr) for a in candidates] if candidates else [max_random_ctr]
            step_max_ctr = max(candidate_ctrs) if candidate_ctrs else max_random_ctr
            ips_step_regret = step_max_ctr - (logged_reward if logged_match else 0.0)
            cumulative_ips_regret += ips_step_regret

            if len(candidates) > 1:
                sensitive_rows += 1
                ips_sensitive_reward_sum += ips_reward
                ips_sensitive_regret_sum += ips_step_regret
                sensitive_seen += 1
                sensitive_ips_reward_cum += ips_reward
                sensitive_ips_regret_cum += ips_step_regret

                action_stat = action_sensitive_stats.setdefault(
                    action,
                    {
                        "action": action,
                        "in_train": bool(action in (initial_seen_actions or set())),
                        "sensitive_impressions": 0,
                        "cumulative_sensitive_ips_reward": 0.0,
                    },
                )
                action_stat["sensitive_impressions"] = int(action_stat["sensitive_impressions"]) + 1
                action_stat["cumulative_sensitive_ips_reward"] = float(action_stat["cumulative_sensitive_ips_reward"]) + float(ips_reward)

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
                if update_frequency == "step_2p5" and step >= next_step_update_mark and pending_updates:
                    policy.update_batch(pending_updates)
                    pending_updates.clear()
                    while next_step_update_mark <= step:
                        next_step_update_mark += update_chunk

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
                    "sensitive_impressions_so_far": sensitive_seen,
                    "ips_ctr_sensitive_so_far": (sensitive_ips_reward_cum / sensitive_seen) if sensitive_seen else 0.0,
                    "avg_ips_regret_sens_so_far": (sensitive_ips_regret_cum / sensitive_seen) if sensitive_seen else 0.0,
                }
            )

            if pbar is not None and step >= next_progress_mark:
                pbar.update(step - pbar.n)
                next_progress_mark += progress_chunk


    if current_day_actions:
        new_actions_in_day = current_day_actions - seen_actions_total
        seen_actions_total.update(current_day_actions)
        final_step = test_df.height
        action_stats_rows.append(
            {
                "step": final_step,
                "date": prev_date,
                "unique_actions_total": len(seen_actions_total),
                "unique_actions_new_in_day": len(new_actions_in_day),
            }
        )

    if pbar is not None:
        pbar.update(test_df.height - pbar.n)
        pbar.close()

    if online_update and policy.can_update_online and pending_updates:
        policy.update_batch(pending_updates)
        pending_updates.clear()

    ctr = total_reward / used if used else 0.0
    ips_ctr = ips_weighted_reward_sum / test_df.height if test_df.height else 0.0
    snips_ctr = (ips_weighted_reward_sum / snips_weight_sum) if snips_weight_sum > 0 else 0.0
    impressions_extrapolated = snips_weight_sum
    match_rate = replay_matches / test_df.height if test_df.height else 0.0
    final_avg_regret = (cumulative_regret / used) if used else 0.0
    final_avg_ips_regret = (cumulative_ips_regret / test_df.height) if test_df.height else 0.0
    ips_ctr_sensitive = (ips_sensitive_reward_sum / sensitive_rows) if sensitive_rows else 0.0
    ips_regret_sens = (ips_sensitive_regret_sum / sensitive_rows) if sensitive_rows else 0.0
    metrics_df = pd.DataFrame([
        {
            "impressions_total": test_df.height,
            "impressions_used": used,
            "total_reward": total_reward,
            "ctr": ctr,
            "ips_weighted_reward": ips_weighted_reward_sum,
            "ips_ctr": ips_ctr,
            "snips_ctr": snips_ctr,
            "impressions_extrapolated": impressions_extrapolated,
            "replay_match_rate": match_rate,
            "cumulative_regret": cumulative_regret,
            "avg_regret": final_avg_regret,
            "cumulative_ips_regret": cumulative_ips_regret,
            "avg_ips_regret": final_avg_ips_regret,
            "sensitive_impressions": sensitive_rows,
            "ips_ctr_sensitive": ips_ctr_sensitive,
            "ips_regret_sens": ips_regret_sens,
        }
    ])
    history_df = pd.DataFrame(history_rows)
    action_stats_df = pd.DataFrame(action_stats_rows)

    selected_df = pd.DataFrame(selected_action_rows)
    if not selected_df.empty:
        action_daily_stats_df = selected_df.groupby(["date", "action"], as_index=False).agg(
            impressions_selected=("action", "size")
        )
    else:
        action_daily_stats_df = pd.DataFrame(columns=["date", "action", "impressions_selected"])

    if action_sensitive_stats:
        action_sensitive_df = pd.DataFrame(list(action_sensitive_stats.values()))
        action_sensitive_df["sensitive_ips_ctr"] = action_sensitive_df.apply(
            lambda r: (float(r["cumulative_sensitive_ips_reward"]) / int(r["sensitive_impressions"])) if int(r["sensitive_impressions"]) > 0 else 0.0,
            axis=1,
        )
        action_sensitive_df = action_sensitive_df.sort_values("sensitive_ips_ctr", ascending=False).reset_index(drop=True)
    else:
        action_sensitive_df = pd.DataFrame(columns=["action", "in_train", "sensitive_impressions", "cumulative_sensitive_ips_reward", "sensitive_ips_ctr"])

    return metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df


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
    action_stats_parts: list[pd.DataFrame] = []
    action_daily_stats_parts: list[pd.DataFrame] = []
    action_sensitive_parts: list[pd.DataFrame] = []
    trained_models: dict[str, dict[str, BasePolicy]] = {}

    for scenario in scenarios:
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)
        trained_models[scenario.name] = {}
        # Regret-only statistics are estimated from combined train+test logs.
        ctr_source = pl.concat([train_df.select(["policy", "show", "reward"]), test_df.select(["policy", "show", "reward"])], how="vertical")
        ctr_by_action, max_random_ctr = build_random_action_ctr_stats(ctr_source)

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_df.height > 0:
                policy.fit(pretrain_df)

            initial_seen_actions = {int(r["show"]) for r in pretrain_df.iter_rows(named=True)}

            metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df = evaluate_policy(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                update_frequency=scenario.update_frequency,
                env_reward=env_reward,
                show_progress=show_progress,
                progress_desc=f"{scenario.name}/{algo_name}",
                ctr_by_action=ctr_by_action,
                max_random_ctr=max_random_ctr,
                initial_seen_actions=initial_seen_actions,
            )
            metrics_df["scenario"] = scenario.name
            metrics_df["algo"] = algo_name
            metrics_parts.append(metrics_df)
            trained_models[scenario.name][algo_name] = policy

            if not history_df.empty:
                history_df["scenario"] = scenario.name
                history_df["algo"] = algo_name
                history_parts.append(history_df)

            if not action_stats_df.empty:
                action_stats_df["scenario"] = scenario.name
                action_stats_df["algo"] = algo_name
                action_stats_parts.append(action_stats_df)

            if not action_daily_stats_df.empty:
                action_daily_stats_df["scenario"] = scenario.name
                action_daily_stats_df["algo"] = algo_name
                action_daily_stats_parts.append(action_daily_stats_df)

            if not action_sensitive_df.empty:
                action_sensitive_df["scenario"] = scenario.name
                action_sensitive_df["algo"] = algo_name
                action_sensitive_parts.append(action_sensitive_df)

    out_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    out_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    out_action_stats = pd.concat(action_stats_parts, ignore_index=True) if action_stats_parts else pd.DataFrame()
    out_action_daily_stats = pd.concat(action_daily_stats_parts, ignore_index=True) if action_daily_stats_parts else pd.DataFrame()
    out_action_sensitive_stats = pd.concat(action_sensitive_parts, ignore_index=True) if action_sensitive_parts else pd.DataFrame()
    return {
        "metrics": out_metrics,
        "history": out_history,
        "action_stats": out_action_stats,
        "action_daily_stats": out_action_daily_stats,
        "action_sensitive_stats": out_action_sensitive_stats,
        "trained_models": trained_models,
    }




def evaluate_policy_ips(
    policy: BasePolicy,
    test_df: pl.DataFrame,
    online_update: bool,
    update_frequency: Literal["daily", "step_2p5"] = "daily",
    show_progress: bool = True,
    progress_desc: str = "evaluate_ips",
    initial_seen_actions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ips_weighted_reward_sum = 0.0
    snips_weight_sum = 0.0
    ips_sensitive_reward_sum = 0.0
    sensitive_rows = 0

    history_rows: list[dict[str, float | int]] = []
    action_stats_rows: list[dict[str, int]] = []
    selected_action_rows: list[dict[str, object]] = []
    action_sensitive_stats: dict[int, dict[str, float | int | bool]] = {}

    total_steps = max(test_df.height, 1)
    progress_chunk = max(1, int(total_steps * 0.05))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=test_df.height, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=0.5)

    pending_updates: list[tuple[int, float, list[float]]] = []
    next_progress_mark = progress_chunk
    update_chunk = max(1, int(total_steps * 0.025))
    next_step_update_mark = update_chunk

    sensitive_seen = 0
    sensitive_ips_reward_cum = 0.0
    seen_actions_total: set[int] = set(initial_seen_actions or set())
    current_day_actions: set[int] = set()

    test_rows = list(test_df.iter_rows(named=True))
    first_row_date = test_rows[0].get("date") if test_rows else None
    prev_date = first_row_date

    if show_progress:
        msg = f"{progress_desc}: train_unique_actions={len(seen_actions_total)}"
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    for step, row in enumerate(test_rows, start=1):
        candidates = row["candidates_list"]
        features = row["features_list"]
        logged_action = int(row["show"])
        logged_reward = float(row["reward"])
        propensity = float(row.get("propensity", 0.0) or 0.0)

        current_date = row.get("date")
        if prev_date is not None and current_date is not None and current_date > prev_date:
            if update_frequency == "daily" and online_update and policy.can_update_online and pending_updates:
                policy.update_batch(pending_updates)
                pending_updates.clear()

            new_actions_in_day = current_day_actions - seen_actions_total
            seen_actions_total.update(current_day_actions)
            action_stats_rows.append(
                {
                    "step": step,
                    "date": prev_date,
                    "unique_actions_total": len(seen_actions_total),
                    "unique_actions_new_in_day": len(new_actions_in_day),
                }
            )
            current_day_actions.clear()

        prev_date = current_date if current_date is not None else prev_date
        current_day_actions.add(logged_action)

        selected_action_rows.append(
            {
                "step": step,
                "date": current_date if current_date is not None else prev_date,
                "action": logged_action,
            }
        )

        target_proba = float(policy.get_action_proba(candidates, logged_action, features, row))
        ips_weight = (target_proba / propensity) if propensity > 0 else 0.0
        ips_reward = ips_weight * logged_reward
        ips_weighted_reward_sum += ips_reward
        snips_weight_sum += ips_weight

        if len(candidates) > 1:
            sensitive_rows += 1
            ips_sensitive_reward_sum += ips_reward
            sensitive_seen += 1
            sensitive_ips_reward_cum += ips_reward

            action_stat = action_sensitive_stats.setdefault(
                logged_action,
                {
                    "action": logged_action,
                    "in_train": bool(logged_action in (initial_seen_actions or set())),
                    "sensitive_impressions": 0,
                    "cumulative_sensitive_ips_reward": 0.0,
                },
            )
            action_stat["sensitive_impressions"] = int(action_stat["sensitive_impressions"]) + 1
            action_stat["cumulative_sensitive_ips_reward"] = float(action_stat["cumulative_sensitive_ips_reward"]) + float(ips_reward)

        if online_update and policy.can_update_online:
            pending_updates.append((logged_action, logged_reward, features))
            if update_frequency == "step_2p5" and step >= next_step_update_mark and pending_updates:
                policy.update_batch(pending_updates)
                pending_updates.clear()
                while next_step_update_mark <= step:
                    next_step_update_mark += update_chunk

        history_rows.append(
            {
                "step": step,
                "ips_reward": ips_reward,
                "ips_avg_reward": ips_weighted_reward_sum / step,
                "impressions_extrapolated_so_far": snips_weight_sum,
                "sensitive_impressions_so_far": sensitive_seen,
                "ips_ctr_sensitive_so_far": (sensitive_ips_reward_cum / sensitive_seen) if sensitive_seen else 0.0,
            }
        )

        if pbar is not None and step >= next_progress_mark:
            pbar.update(step - pbar.n)
            next_progress_mark += progress_chunk

    if current_day_actions:
        new_actions_in_day = current_day_actions - seen_actions_total
        seen_actions_total.update(current_day_actions)
        final_step = test_df.height
        action_stats_rows.append(
            {
                "step": final_step,
                "date": prev_date,
                "unique_actions_total": len(seen_actions_total),
                "unique_actions_new_in_day": len(new_actions_in_day),
            }
        )

    if pbar is not None:
        pbar.update(test_df.height - pbar.n)
        pbar.close()

    if online_update and policy.can_update_online and pending_updates:
        policy.update_batch(pending_updates)
        pending_updates.clear()

    ips_ctr = ips_weighted_reward_sum / test_df.height if test_df.height else 0.0
    snips_ctr = (ips_weighted_reward_sum / snips_weight_sum) if snips_weight_sum > 0 else 0.0
    impressions_extrapolated = snips_weight_sum
    ips_ctr_sensitive = (ips_sensitive_reward_sum / sensitive_rows) if sensitive_rows else 0.0

    metrics_df = pd.DataFrame([
        {
            "impressions_total": test_df.height,
            "ips_weighted_reward": ips_weighted_reward_sum,
            "ips_ctr": ips_ctr,
            "snips_ctr": snips_ctr,
            "impressions_extrapolated": impressions_extrapolated,
            "sensitive_impressions": sensitive_rows,
            "ips_ctr_sensitive": ips_ctr_sensitive,
        }
    ])

    history_df = pd.DataFrame(history_rows)
    action_stats_df = pd.DataFrame(action_stats_rows)

    selected_df = pd.DataFrame(selected_action_rows)
    if not selected_df.empty:
        action_daily_stats_df = selected_df.groupby(["date", "action"], as_index=False).agg(
            impressions_selected=("action", "size")
        )
    else:
        action_daily_stats_df = pd.DataFrame(columns=["date", "action", "impressions_selected"])

    if action_sensitive_stats:
        action_sensitive_df = pd.DataFrame(list(action_sensitive_stats.values()))
        action_sensitive_df["sensitive_ips_ctr"] = action_sensitive_df.apply(
            lambda r: (float(r["cumulative_sensitive_ips_reward"]) / int(r["sensitive_impressions"])) if int(r["sensitive_impressions"]) > 0 else 0.0,
            axis=1,
        )
        action_sensitive_df = action_sensitive_df.sort_values("sensitive_ips_ctr", ascending=False).reset_index(drop=True)
    else:
        action_sensitive_df = pd.DataFrame(columns=["action", "in_train", "sensitive_impressions", "cumulative_sensitive_ips_reward", "sensitive_ips_ctr"])

    return metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df


def run_scenarios_ips(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    policy_factories: dict[str, Callable[[], BasePolicy]],
    scenarios: list[ScenarioConfig],
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    metrics_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    action_stats_parts: list[pd.DataFrame] = []
    action_daily_stats_parts: list[pd.DataFrame] = []
    action_sensitive_parts: list[pd.DataFrame] = []
    trained_models: dict[str, dict[str, BasePolicy]] = {}

    for scenario in scenarios:
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)
        trained_models[scenario.name] = {}

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_df.height > 0:
                policy.fit(pretrain_df)

            initial_seen_actions = {int(r["show"]) for r in pretrain_df.iter_rows(named=True)}

            metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df = evaluate_policy_ips(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                update_frequency=scenario.update_frequency,
                show_progress=show_progress,
                progress_desc=f"{scenario.name}/{algo_name}",
                initial_seen_actions=initial_seen_actions,
            )
            metrics_df["scenario"] = scenario.name
            metrics_df["algo"] = algo_name
            metrics_parts.append(metrics_df)
            trained_models[scenario.name][algo_name] = policy

            if not history_df.empty:
                history_df["scenario"] = scenario.name
                history_df["algo"] = algo_name
                history_parts.append(history_df)

            if not action_stats_df.empty:
                action_stats_df["scenario"] = scenario.name
                action_stats_df["algo"] = algo_name
                action_stats_parts.append(action_stats_df)

            if not action_daily_stats_df.empty:
                action_daily_stats_df["scenario"] = scenario.name
                action_daily_stats_df["algo"] = algo_name
                action_daily_stats_parts.append(action_daily_stats_df)

            if not action_sensitive_df.empty:
                action_sensitive_df["scenario"] = scenario.name
                action_sensitive_df["algo"] = algo_name
                action_sensitive_parts.append(action_sensitive_df)

    out_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    out_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    out_action_stats = pd.concat(action_stats_parts, ignore_index=True) if action_stats_parts else pd.DataFrame()
    out_action_daily_stats = pd.concat(action_daily_stats_parts, ignore_index=True) if action_daily_stats_parts else pd.DataFrame()
    out_action_sensitive_stats = pd.concat(action_sensitive_parts, ignore_index=True) if action_sensitive_parts else pd.DataFrame()

    return {
        "metrics": out_metrics,
        "history": out_history,
        "action_stats": out_action_stats,
        "action_daily_stats": out_action_daily_stats,
        "action_sensitive_stats": out_action_sensitive_stats,
        "trained_models": trained_models,
    }

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
        ScenarioConfig("case_2_random_pretrain_online_update_daily", "random", True, "daily"),
        ScenarioConfig("case_2b_random_pretrain_online_update_step_2p5", "random", True, "step_2p5"),
        ScenarioConfig("case_3_all_pretrain_predict_only", "all", False),
        ScenarioConfig("case_4_all_pretrain_online_update_daily", "all", True, "daily"),
        ScenarioConfig("case_4b_all_pretrain_online_update_step_2p5", "all", True, "step_2p5"),
        ScenarioConfig("case_5_no_pretrain_online_update_daily", "none", True, "daily"),
        ScenarioConfig("case_5b_no_pretrain_online_update_step_2p5", "none", True, "step_2p5"),
    ]


def default_scenario() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("case_2_random_pretrain_online_update_daily", "random", True, "daily"),
    ]


def two_ways_default_scenario() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("case_2_random_pretrain_online_update_daily", "random", True, "daily"),
        ScenarioConfig("case_2b_random_pretrain_online_update_step_2p5", "random", True, "step_2p5"),
    ]


def core_scenarios() -> list[ScenarioConfig]:
    """Backward-compatible alias for previous default (two ways)."""
    return two_ways_default_scenario()
