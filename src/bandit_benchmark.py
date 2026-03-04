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
    max_random_ctr: float = 0.0,
    initial_seen_actions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_reward = 0.0
    ips_weighted_reward_sum = 0.0
    used = 0
    replay_matches = 0
    cumulative_regret = 0.0  # replay regret accumulator (used only for regret metrics)
    cumulative_ips_regret = 0.0  # IPS regret accumulator (used only for regret metrics)
    history_rows: list[dict[str, float | int]] = []
    action_stats_rows: list[dict[str, int]] = []
    selected_action_rows: list[dict[str, object]] = []

    action_ctr = ctr_by_action or {}
    # Regret-only baseline fallback for unseen actions within candidate sets.

    total_steps = max(test_df.height, 1)
    progress_chunk = max(1, int(total_steps * 0.05))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=test_df.height, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=0.5)

    pending_updates: list[tuple[int, float, list[float]]] = []
    next_progress_mark = progress_chunk
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
                if online_update and policy.can_update_online and pending_updates:
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

            ips_reward = (logged_match * logged_reward / propensity) if propensity > 0 else 0.0
            ips_weighted_reward_sum += ips_reward
            # Regret-only per-step baseline: max expected CTR among currently available actions.
            candidate_ctrs = [action_ctr.get(int(a), max_random_ctr) for a in candidates] if candidates else [max_random_ctr]
            step_max_ctr = max(candidate_ctrs) if candidate_ctrs else max_random_ctr
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
            "cumulative_regret": cumulative_regret,
            "avg_regret": final_avg_regret,
            "cumulative_ips_regret": cumulative_ips_regret,
            "avg_ips_regret": final_avg_ips_regret,
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

    return metrics_df, history_df, action_stats_df, action_daily_stats_df


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

            metrics_df, history_df, action_stats_df, action_daily_stats_df = evaluate_policy(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
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

    out_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    out_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    out_action_stats = pd.concat(action_stats_parts, ignore_index=True) if action_stats_parts else pd.DataFrame()
    out_action_daily_stats = pd.concat(action_daily_stats_parts, ignore_index=True) if action_daily_stats_parts else pd.DataFrame()
    return {
        "metrics": out_metrics,
        "history": out_history,
        "action_stats": out_action_stats,
        "action_daily_stats": out_action_daily_stats,
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
        ScenarioConfig("case_2_random_pretrain_online_update", "random", True),
        ScenarioConfig("case_3_all_pretrain_predict_only", "all", False),
        ScenarioConfig("case_4_all_pretrain_online_update", "all", True),
        ScenarioConfig("case_5_no_pretrain_online_update", "none", True),
    ]


def core_scenarios() -> list[ScenarioConfig]:
    return [ScenarioConfig("case_2_random_pretrain_online_update", "random", True)]
