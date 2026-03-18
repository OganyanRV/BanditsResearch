"""Tree-based Thompson sampling models and policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import polars as pl

from bandit_benchmark_basic import Action, BasePolicy


SplitCriterion = Literal["bernoulli_log_likelihood", "beta_marginal_likelihood"]
SampleWeightMode = Literal["unit", "propensity", "inverse_propensity"]


@dataclass
class _TreeNode:
    raw_count: int
    raw_clicks: float
    weighted_count: float
    weighted_clicks: float
    alpha: float
    beta: float
    depth: int
    leaf_id: int | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class ActionTreeThompsonModel:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int | None = None,
        alpha0: float | None = 1.0,
        beta0: float | None = 1.0,
        c_min: int = 5,
        random_state: int = 42,
        split_criterion: SplitCriterion = "bernoulli_log_likelihood",
        prior_mean: float | None = None,
        prior_strength: float | None = None,
    ):
        import numpy as np

        self.max_depth = int(max_depth)
        self.min_samples_leaf = None if min_samples_leaf is None else int(min_samples_leaf)
        self.c_min = int(c_min)
        self.random_state = int(random_state)
        self.split_criterion = split_criterion

        self.alpha0, self.beta0 = self._resolve_prior(alpha0, beta0, prior_mean, prior_strength)

        self.tree: _TreeNode | None = None
        self.leaf_stats: dict[int, dict[str, float | int]] = {}
        self.global_alpha: float | None = None
        self.global_beta: float | None = None
        self.resolved_min_samples_leaf: int = 1

        self._rng = np.random.default_rng(self.random_state)
        self._next_leaf_id = 0
        self._X_hist = np.empty((0, 0), dtype=float)
        self._y_hist = np.empty((0,), dtype=float)
        self._w_hist = np.empty((0,), dtype=float)

    @staticmethod
    def _resolve_prior(
        alpha0: float | None,
        beta0: float | None,
        prior_mean: float | None,
        prior_strength: float | None,
    ) -> tuple[float, float]:
        if alpha0 is not None and beta0 is not None:
            return float(alpha0), float(beta0)
        if prior_mean is not None and prior_strength is not None:
            mean = min(max(float(prior_mean), 1e-6), 1.0 - 1e-6)
            strength = max(float(prior_strength), 1e-6)
            return strength * mean, strength * (1.0 - mean)
        return 1.0, 1.0

    def _ensure_fitted(self):
        if self.global_alpha is None or self.global_beta is None:
            raise RuntimeError("Model is not fitted yet.")

    def _to_2d(self, X):
        import numpy as np

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    @staticmethod
    def _to_1d(y):
        import numpy as np

        y = np.asarray(y, dtype=float).reshape(-1)
        return (y > 0).astype(float)

    @staticmethod
    def _to_weights(sample_weight, n: int):
        import numpy as np

        if sample_weight is None:
            return np.ones(n, dtype=float)
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.shape[0] != n:
            raise ValueError("sample_weight must be aligned with y")
        return weights

    def _resolve_min_samples_leaf(self, y, sample_weight) -> int:
        if self.min_samples_leaf is not None:
            return max(1, int(self.min_samples_leaf))
        if self.c_min <= 0:
            return 1
        ctr = float(y.mean()) if len(y) else 0.0
        ctr = max(ctr, 1e-6)
        return max(1, int(math.ceil(self.c_min / ctr)))

    def _leaf_dict(self, node: _TreeNode) -> dict[str, float | int]:
        return {
            "n": int(node.raw_count),
            "clicks": float(node.raw_clicks),
            "weighted_n": float(node.weighted_count),
            "weighted_clicks": float(node.weighted_clicks),
            "alpha": float(node.alpha),
            "beta": float(node.beta),
        }

    def _bernoulli_log_likelihood(self, weighted_clicks: float, weighted_fails: float) -> float:
        total = weighted_clicks + weighted_fails
        if total <= 0:
            return float("-inf")
        p = weighted_clicks / total
        out = 0.0
        if weighted_clicks > 0 and p > 0:
            out += weighted_clicks * math.log(p)
        if weighted_fails > 0 and p < 1:
            out += weighted_fails * math.log(1.0 - p)
        return out

    def _beta_marginal_log_likelihood(self, weighted_clicks: float, weighted_fails: float) -> float:
        a = self.alpha0
        b = self.beta0
        return (
            math.lgamma(a + weighted_clicks)
            + math.lgamma(b + weighted_fails)
            - math.lgamma(a + b + weighted_clicks + weighted_fails)
            - math.lgamma(a)
            - math.lgamma(b)
            + math.lgamma(a + b)
        )

    def _score(self, weighted_clicks: float, weighted_fails: float) -> float:
        if self.split_criterion == "bernoulli_log_likelihood":
            return self._bernoulli_log_likelihood(weighted_clicks, weighted_fails)
        if self.split_criterion == "beta_marginal_likelihood":
            return self._beta_marginal_log_likelihood(weighted_clicks, weighted_fails)
        raise ValueError(f"Unknown split_criterion: {self.split_criterion}")

    def _make_leaf(self, raw_count: int, raw_clicks: float, weighted_count: float, weighted_clicks: float, depth: int) -> _TreeNode:
        node = _TreeNode(
            raw_count=int(raw_count),
            raw_clicks=float(raw_clicks),
            weighted_count=float(weighted_count),
            weighted_clicks=float(weighted_clicks),
            alpha=self.alpha0 + float(weighted_clicks),
            beta=self.beta0 + float(max(weighted_count - weighted_clicks, 0.0)),
            depth=int(depth),
            leaf_id=self._next_leaf_id,
        )
        self.leaf_stats[self._next_leaf_id] = self._leaf_dict(node)
        self._next_leaf_id += 1
        return node

    def _build_node(self, X, y, w, idx, depth: int) -> _TreeNode | None:
        import numpy as np

        raw_count = int(len(idx))
        raw_clicks = float(y[idx].sum())
        weighted_count = float(w[idx].sum())
        weighted_clicks = float((w[idx] * y[idx]).sum())

        if raw_count == 0:
            return None
        if raw_clicks < self.c_min:
            return None

        parent_score = self._score(weighted_clicks, weighted_count - weighted_clicks)
        if (
            depth >= self.max_depth
            or raw_count < 2 * self.resolved_min_samples_leaf
            or raw_clicks < 2 * self.c_min
        ):
            return self._make_leaf(raw_count, raw_clicks, weighted_count, weighted_clicks, depth)

        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        best_left_idx = None
        best_right_idx = None

        for feature_idx in range(X.shape[1]):
            ordered_local = idx[np.argsort(X[idx, feature_idx], kind="mergesort")]
            values = X[ordered_local, feature_idx]
            y_sorted = y[ordered_local]
            w_sorted = w[ordered_local]

            raw_clicks_cum = np.cumsum(y_sorted)
            weighted_count_cum = np.cumsum(w_sorted)
            weighted_clicks_cum = np.cumsum(w_sorted * y_sorted)

            for split_pos in range(self.resolved_min_samples_leaf - 1, raw_count - self.resolved_min_samples_leaf):
                if values[split_pos] == values[split_pos + 1]:
                    continue

                left_raw_count = split_pos + 1
                right_raw_count = raw_count - left_raw_count
                left_raw_clicks = float(raw_clicks_cum[split_pos])
                right_raw_clicks = raw_clicks - left_raw_clicks
                if left_raw_count < self.resolved_min_samples_leaf or right_raw_count < self.resolved_min_samples_leaf:
                    continue
                if left_raw_clicks < self.c_min or right_raw_clicks < self.c_min:
                    continue

                left_weighted_count = float(weighted_count_cum[split_pos])
                left_weighted_clicks = float(weighted_clicks_cum[split_pos])
                right_weighted_count = weighted_count - left_weighted_count
                right_weighted_clicks = weighted_clicks - left_weighted_clicks

                split_score = self._score(left_weighted_clicks, left_weighted_count - left_weighted_clicks) + self._score(
                    right_weighted_clicks,
                    right_weighted_count - right_weighted_clicks,
                )
                gain = split_score - parent_score
                if gain <= best_gain:
                    continue

                threshold = float((values[split_pos] + values[split_pos + 1]) / 2.0)
                left_mask = X[idx, feature_idx] <= threshold
                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]
                if len(left_idx) < self.resolved_min_samples_leaf or len(right_idx) < self.resolved_min_samples_leaf:
                    continue

                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                best_left_idx = left_idx
                best_right_idx = right_idx

        if best_feature is None or best_left_idx is None or best_right_idx is None:
            return self._make_leaf(raw_count, raw_clicks, weighted_count, weighted_clicks, depth)

        left_node = self._build_node(X, y, w, best_left_idx, depth + 1)
        right_node = self._build_node(X, y, w, best_right_idx, depth + 1)
        if left_node is None or right_node is None:
            return self._make_leaf(raw_count, raw_clicks, weighted_count, weighted_clicks, depth)

        return _TreeNode(
            raw_count=raw_count,
            raw_clicks=raw_clicks,
            weighted_count=weighted_count,
            weighted_clicks=weighted_clicks,
            alpha=self.alpha0 + weighted_clicks,
            beta=self.beta0 + max(weighted_count - weighted_clicks, 0.0),
            depth=depth,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )

    def fit(self, X, y, sample_weight=None):
        import numpy as np

        X = self._to_2d(X)
        y = self._to_1d(y)
        w = self._to_weights(sample_weight, len(y))

        self.resolved_min_samples_leaf = self._resolve_min_samples_leaf(y, w)
        self.global_alpha = self.alpha0 + float((w * y).sum())
        self.global_beta = self.beta0 + float((w * (1.0 - y)).sum())
        self.leaf_stats = {}
        self._next_leaf_id = 0
        self._X_hist = X.copy()
        self._y_hist = y.copy()
        self._w_hist = w.copy()

        idx = np.arange(len(y), dtype=int)
        self.tree = self._build_node(X, y, w, idx, depth=0)
        return self

    def _traverse_row(self, x) -> _TreeNode | None:
        node = self.tree
        while node is not None and not node.is_leaf:
            if node.feature_index is None or node.threshold is None:
                break
            if float(x[node.feature_index]) <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def get_leaf_stats(self, X):
        self._ensure_fitted()
        X = self._to_2d(X)
        out = []
        for row in X:
            node = self._traverse_row(row)
            if node is None:
                out.append(
                    {
                        "n": 0,
                        "clicks": 0.0,
                        "weighted_n": 0.0,
                        "weighted_clicks": 0.0,
                        "alpha": self.global_alpha,
                        "beta": self.global_beta,
                        "used_global_fallback": True,
                    }
                )
                continue
            out.append({**self._leaf_dict(node), "used_global_fallback": False})
        return out

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

    def should_rebuild(self, X, y, sample_weight=None) -> bool:
        del X, y, sample_weight
        return False

    def rebuild(self) -> None:
        if self._X_hist.size == 0:
            return
        self.fit(self._X_hist, self._y_hist, sample_weight=self._w_hist)

    def update_batch(self, X, y, sample_weight=None):
        import numpy as np

        self._ensure_fitted()
        X = self._to_2d(X)
        y = self._to_1d(y)
        w = self._to_weights(sample_weight, len(y))

        if self._X_hist.size == 0:
            self._X_hist = X.copy()
        else:
            self._X_hist = np.vstack([self._X_hist, X])
        self._y_hist = np.concatenate([self._y_hist, y])
        self._w_hist = np.concatenate([self._w_hist, w])

        self.global_alpha = float(self.global_alpha) + float((w * y).sum())
        self.global_beta = float(self.global_beta) + float((w * (1.0 - y)).sum())

        if self.should_rebuild(X, y, sample_weight=w):
            self.rebuild()
            return

        if self.tree is None:
            return

        for row, reward, weight in zip(X, y, w):
            node = self._traverse_row(row)
            if node is None:
                continue
            node.raw_count += 1
            node.raw_clicks += float(reward)
            node.weighted_count += float(weight)
            node.weighted_clicks += float(weight * reward)
            node.alpha = self.alpha0 + node.weighted_clicks
            node.beta = self.beta0 + max(node.weighted_count - node.weighted_clicks, 0.0)
            if node.leaf_id is not None:
                self.leaf_stats[node.leaf_id] = self._leaf_dict(node)


class TreeThompsonSamplingPolicy(BasePolicy):
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int | None = None,
        alpha0: float | None = 1.0,
        beta0: float | None = 1.0,
        c_min: int = 5,
        random_state: int = 42,
        split_criterion: SplitCriterion = "bernoulli_log_likelihood",
        prior_mean: float | None = None,
        prior_strength: float | None = None,
        sample_weight_mode: SampleWeightMode = "unit",
        can_update_online: bool | None = True,
    ):
        super().__init__(can_update_online=can_update_online)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = None if min_samples_leaf is None else int(min_samples_leaf)
        self.c_min = int(c_min)
        self.random_state = int(random_state)
        self.split_criterion = split_criterion
        self.sample_weight_mode = sample_weight_mode
        self.alpha0, self.beta0 = ActionTreeThompsonModel._resolve_prior(alpha0, beta0, prior_mean, prior_strength)

        self.action_models: dict[int, ActionTreeThompsonModel] = {}
        self.action_history: dict[int, list[tuple[list[float], float, int, float]]] = {}

    def _build_model(self) -> ActionTreeThompsonModel:
        return ActionTreeThompsonModel(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            alpha0=self.alpha0,
            beta0=self.beta0,
            c_min=self.c_min,
            random_state=self.random_state,
            split_criterion=self.split_criterion,
        )

    def _sample_weight_from_propensity(self, propensity: float | None) -> float:
        propensity_value = float(propensity or 0.0)
        if self.sample_weight_mode == "unit":
            return 1.0
        if self.sample_weight_mode == "propensity":
            return propensity_value if propensity_value > 0 else 0.0
        if self.sample_weight_mode == "inverse_propensity":
            return (1.0 / propensity_value) if propensity_value > 0 else 0.0
        raise ValueError(f"Unknown sample_weight_mode: {self.sample_weight_mode}")

    def _parse_update(self, update) -> tuple[int, float, list[float], float]:
        if len(update) == 3:
            a, r, f = update
            return int(a), float(r), [float(v) for v in f], 1.0
        if len(update) == 4:
            a, r, f, w = update
            return int(a), float(r), [float(v) for v in f], float(w)
        raise ValueError("Updates must be (action, reward, features) or (action, reward, features, sample_weight)")

    def fit(self, train_df: pl.DataFrame) -> None:
        import numpy as np

        rows = list(train_df.iter_rows(named=True))
        self.action_models = {}
        self.action_history = {}

        for r in rows:
            action = int(r["show"])
            feat = [float(v) for v in r["features_list"]]
            reward = float(r["reward"])
            sample_weight = self._sample_weight_from_propensity(r.get("propensity", 1.0))
            self.action_history.setdefault(action, []).append((feat, reward, action, sample_weight))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=float)
            w = np.asarray([it[3] for it in items], dtype=float)
            if len(X) == 0:
                continue
            model = self._build_model().fit(X, y, sample_weight=w)
            self.action_models[action] = model

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

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        grouped: dict[int, list[tuple[list[float], float, int, float]]] = {}
        for update in pending_updates:
            action, reward, features, sample_weight = self._parse_update(update)
            self.action_history.setdefault(action, []).append((features, reward, action, sample_weight))
            grouped.setdefault(action, []).append((features, reward, action, sample_weight))

        for action, items in grouped.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=float)
            w = np.asarray([it[3] for it in items], dtype=float)
            if action in self.action_models and self.action_models[action].tree is not None:
                self.action_models[action].update_batch(X, y, sample_weight=w)
            else:
                all_items = self.action_history.get(action, items)
                Xa = np.asarray([it[0] for it in all_items], dtype=float)
                ya = np.asarray([1 if it[1] > 0 else 0 for it in all_items], dtype=float)
                wa = np.asarray([it[3] for it in all_items], dtype=float)
                if len(Xa) == 0:
                    continue
                self.action_models[action] = self._build_model().fit(Xa, ya, sample_weight=wa)


class TreeThompsonSamplingPolicyUpdateV1(TreeThompsonSamplingPolicy):
    """Incremental tree updates via ActionTreeThompsonModel.update_batch."""


class TreeThompsonSamplingPolicyDummyRefit(TreeThompsonSamplingPolicy):
    """Refits per-action trees on full stored history at each update."""

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for update in pending_updates:
            action, reward, features, sample_weight = self._parse_update(update)
            self.action_history.setdefault(action, []).append((features, reward, action, sample_weight))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=float)
            w = np.asarray([it[3] for it in items], dtype=float)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model().fit(X, y, sample_weight=w)
