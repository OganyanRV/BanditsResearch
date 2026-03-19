"""Tree-based Thompson sampling models and policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import polars as pl

from bandit_benchmark_basic import Action, BasePolicy


class ActionTreeThompsonModel:
    """Original sklearn-based tree Thompson model."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 300,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        c_min: int = 5,
        random_state: int = 42,
    ):
        import numpy as np

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
    """Original sklearn-based tree Thompson policy."""

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

        for row in rows:
            action = int(row["show"])
            features = [float(v) for v in row["features_list"]]
            reward = float(row["reward"])
            self.action_history.setdefault(action, []).append((features, reward, action))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model().fit(X, y)

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        import numpy as np

        del row
        if not candidates:
            raise ValueError("Empty candidate set")

        x = np.asarray(features, dtype=float)
        best_action = int(candidates[0])
        best_score = -1.0
        rng = np.random.default_rng(self.random_state)

        for candidate in candidates:
            action = int(candidate)
            model = self.action_models.get(action)
            if model is None or model.tree is None:
                score = float(rng.beta(self.alpha0, self.beta0))
            else:
                score = float(model.sample_proba(x, n_samples=1)[0])
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

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
        rng = np.random.default_rng(self.random_state)
        target = int(action)
        wins = 0
        n_mc = 128

        for _ in range(n_mc):
            best_action = int(candidates[0])
            best_score = -1.0
            for candidate in candidates:
                aa = int(candidate)
                model = self.action_models.get(aa)
                if model is None or model.tree is None:
                    score = float(rng.beta(self.alpha0, self.beta0))
                else:
                    score = float(model.sample_proba(x, n_samples=1)[0])
                if score > best_score:
                    best_score = score
                    best_action = aa
            wins += int(best_action == target)
        return wins / n_mc

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for action, reward, features in pending_updates:
            aa = int(action)
            ff = [float(v) for v in features]
            rr = float(reward)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        grouped: dict[int, list[tuple[list[float], float, int]]] = {}
        for action, reward, features in pending_updates:
            grouped.setdefault(int(action), []).append(([float(v) for v in features], float(reward), int(action)))

        for action, items in grouped.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if action in self.action_models and self.action_models[action].tree is not None:
                self.action_models[action].update_batch(X, y)
            else:
                all_items = self.action_history.get(action, items)
                Xa = np.asarray([it[0] for it in all_items], dtype=float)
                ya = np.asarray([1 if it[1] > 0 else 0 for it in all_items], dtype=int)
                if len(Xa) == 0:
                    continue
                self.action_models[action] = self._build_model().fit(Xa, ya)


class TreeThompsonSamplingPolicyUpdateV1(TreeThompsonSamplingPolicy):
    """Incremental sklearn-tree updates via ActionTreeThompsonModel.update_batch."""


class TreeThompsonSamplingPolicyDummyRefit(TreeThompsonSamplingPolicy):
    """Sklearn-tree variant that refits on full action history after each update."""

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for action, reward, features in pending_updates:
            aa = int(action)
            ff = [float(v) for v in features]
            rr = float(reward)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model().fit(X, y)


CustomSplitCriterion = Literal["bernoulli_log_likelihood", "beta_marginal_likelihood"]
@dataclass
class _CustomTreeNode:
    count: int
    clicks: float
    alpha: float
    beta: float
    depth: int
    leaf_id: int | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: _CustomTreeNode | None = None
    right: _CustomTreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def mean_score_class_1(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.5
        return self.alpha / denom

    @property
    def mean_score_class_0(self) -> float:
        return 1.0 - self.mean_score_class_1


class CustomActionTreeThompsonModel:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int | None = None,
        alpha0: float | None = 1.0,
        beta0: float | None = 1.0,
        c_min: int = 5,
        random_state: int = 42,
        split_criterion: CustomSplitCriterion = "bernoulli_log_likelihood",
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

        self.tree: _CustomTreeNode | None = None
        self.leaf_stats: dict[int, dict[str, float | int]] = {}
        self.global_alpha: float | None = None
        self.global_beta: float | None = None
        self.resolved_min_samples_leaf: int = 1

        self._rng = np.random.default_rng(self.random_state)
        self._next_leaf_id = 0
        self._X_hist = np.empty((0, 0), dtype=float)
        self._y_hist = np.empty((0,), dtype=float)

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

    def _resolve_min_samples_leaf(self, y) -> int:
        if self.min_samples_leaf is not None:
            return max(1, int(self.min_samples_leaf))
        if self.c_min <= 0:
            return 1
        ctr = float(y.mean()) if len(y) else 0.0
        ctr = max(ctr, 1e-6)
        return max(1, int(math.ceil(self.c_min / ctr)))

    def _leaf_dict(self, node: _CustomTreeNode) -> dict[str, float | int]:
        return {
            "n": int(node.count),
            "clicks": float(node.clicks),
            "alpha": float(node.alpha),
            "beta": float(node.beta),
            "mean_score_class_0": float(node.mean_score_class_0),
            "mean_score_class_1": float(node.mean_score_class_1),
        }

    def _bernoulli_log_likelihood(self, clicks: float, fails: float) -> float:
        total = clicks + fails
        if total <= 0:
            return float("-inf")
        p = clicks / total
        out = 0.0
        if clicks > 0 and p > 0:
            out += clicks * math.log(p)
        if fails > 0 and p < 1:
            out += fails * math.log(1.0 - p)
        return out

    def _beta_marginal_log_likelihood(self, clicks: float, fails: float) -> float:
        a = self.alpha0
        b = self.beta0
        return (
            math.lgamma(a + clicks)
            + math.lgamma(b + fails)
            - math.lgamma(a + b + clicks + fails)
            - math.lgamma(a)
            - math.lgamma(b)
            + math.lgamma(a + b)
        )

    def _score(self, clicks: float, fails: float) -> float:
        if self.split_criterion == "bernoulli_log_likelihood":
            return self._bernoulli_log_likelihood(clicks, fails)
        if self.split_criterion == "beta_marginal_likelihood":
            return self._beta_marginal_log_likelihood(clicks, fails)
        raise ValueError(f"Unknown split_criterion: {self.split_criterion}")

    def _make_leaf(self, count: int, clicks: float, depth: int) -> _CustomTreeNode:
        node = _CustomTreeNode(
            count=int(count),
            clicks=float(clicks),
            alpha=self.alpha0 + float(clicks),
            beta=self.beta0 + float(max(count - clicks, 0.0)),
            depth=int(depth),
            leaf_id=self._next_leaf_id,
        )
        self.leaf_stats[self._next_leaf_id] = self._leaf_dict(node)
        self._next_leaf_id += 1
        return node

    def _build_node(self, X, y, idx, depth: int) -> _CustomTreeNode | None:
        import numpy as np

        count = int(len(idx))
        clicks = float(y[idx].sum())

        if count == 0 or clicks < self.c_min:
            return None

        parent_score = self._score(clicks, count - clicks)
        if depth >= self.max_depth or count < 2 * self.resolved_min_samples_leaf or clicks < 2 * self.c_min:
            return self._make_leaf(count, clicks, depth)

        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        best_left_idx = None
        best_right_idx = None

        for feature_idx in range(X.shape[1]):
            ordered_local = idx[np.argsort(X[idx, feature_idx], kind="mergesort")]
            values = X[ordered_local, feature_idx]
            y_sorted = y[ordered_local]
            clicks_cum = np.cumsum(y_sorted)
            count_cum = np.cumsum(np.ones_like(y_sorted, dtype=float))

            for split_pos in range(self.resolved_min_samples_leaf - 1, count - self.resolved_min_samples_leaf):
                if values[split_pos] == values[split_pos + 1]:
                    continue

                left_raw_count = split_pos + 1
                right_raw_count = count - left_raw_count
                left_raw_clicks = float(clicks_cum[split_pos])
                right_raw_clicks = clicks - left_raw_clicks
                if left_raw_count < self.resolved_min_samples_leaf or right_raw_count < self.resolved_min_samples_leaf:
                    continue
                if left_raw_clicks < self.c_min or right_raw_clicks < self.c_min:
                    continue

                left_count = float(count_cum[split_pos])
                left_clicks = float(clicks_cum[split_pos])
                right_count = float(count) - left_count
                right_clicks = clicks - left_clicks

                split_score = self._score(left_clicks, left_count - left_clicks) + self._score(
                    right_clicks,
                    right_count - right_clicks,
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
            return self._make_leaf(count, clicks, depth)

        left_node = self._build_node(X, y, best_left_idx, depth + 1)
        right_node = self._build_node(X, y, best_right_idx, depth + 1)
        if left_node is None or right_node is None:
            return self._make_leaf(count, clicks, depth)

        return _CustomTreeNode(
            count=count,
            clicks=clicks,
            alpha=self.alpha0 + clicks,
            beta=self.beta0 + max(count - clicks, 0.0),
            depth=depth,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )

    def fit(self, X, y):
        import numpy as np

        X = self._to_2d(X)
        y = self._to_1d(y)

        self.resolved_min_samples_leaf = self._resolve_min_samples_leaf(y)
        self.global_alpha = self.alpha0 + float(y.sum())
        self.global_beta = self.beta0 + float(len(y) - y.sum())
        self.leaf_stats = {}
        self._next_leaf_id = 0
        self._X_hist = X.copy()
        self._y_hist = y.copy()

        idx = np.arange(len(y), dtype=int)
        self.tree = self._build_node(X, y, idx, depth=0)
        return self

    def _traverse_row(self, x) -> _CustomTreeNode | None:
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
                        "alpha": self.global_alpha,
                        "beta": self.global_beta,
                        "mean_score_class_0": float(self.global_beta / (self.global_alpha + self.global_beta)),
                        "mean_score_class_1": float(self.global_alpha / (self.global_alpha + self.global_beta)),
                        "used_global_fallback": True,
                    }
                )
            else:
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

    def should_rebuild(self, X, y) -> bool:
        del X, y
        return False

    def rebuild(self) -> None:
        if self._X_hist.size == 0:
            return
        self.fit(self._X_hist, self._y_hist)

    def update_batch(self, X, y):
        import numpy as np

        self._ensure_fitted()
        X = self._to_2d(X)
        y = self._to_1d(y)

        if self._X_hist.size == 0:
            self._X_hist = X.copy()
        else:
            self._X_hist = np.vstack([self._X_hist, X])
        self._y_hist = np.concatenate([self._y_hist, y])

        self.global_alpha = float(self.global_alpha) + float(y.sum())
        self.global_beta = float(self.global_beta) + float(len(y) - y.sum())

        if self.should_rebuild(X, y):
            self.rebuild()
            return

        if self.tree is None:
            return

        for row, reward in zip(X, y):
            node = self._traverse_row(row)
            if node is None:
                continue
            node.count += 1
            node.clicks += float(reward)
            node.alpha = self.alpha0 + node.clicks
            node.beta = self.beta0 + max(node.count - node.clicks, 0.0)
            if node.leaf_id is not None:
                self.leaf_stats[node.leaf_id] = self._leaf_dict(node)

    def visualize_tree(self) -> str:
        return render_custom_tree_text(self)


def _format_custom_tree_node(node: _CustomTreeNode, indent: str = "") -> list[str]:
    mean0 = node.mean_score_class_0
    mean1 = node.mean_score_class_1
    summary = (
        f"n={node.count}, clicks={node.clicks:.3f}, alpha={node.alpha:.3f}, beta={node.beta:.3f}, "
        f"mean_p0={mean0:.3f}, mean_p1={mean1:.3f}"
    )
    if node.is_leaf:
        leaf_label = f"leaf_id={node.leaf_id}" if node.leaf_id is not None else "leaf"
        return [f"{indent}{leaf_label}: {summary}"]

    split = f"x[{node.feature_index}] <= {node.threshold:.6f}"
    lines = [f"{indent}{split}: {summary}"]
    if node.left is not None:
        lines.append(f"{indent}├─ yes")
        lines.extend(_format_custom_tree_node(node.left, indent + "│  "))
    if node.right is not None:
        lines.append(f"{indent}└─ no")
        lines.extend(_format_custom_tree_node(node.right, indent + "   "))
    return lines


def render_custom_tree_text(model: CustomActionTreeThompsonModel) -> str:
    model._ensure_fitted()
    if model.tree is None:
        return "<empty custom tree>"
    return "\n".join(_format_custom_tree_node(model.tree))


class CustomTreeThompsonSamplingPolicy(TreeThompsonSamplingPolicy):
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int | None = None,
        alpha0: float | None = 1.0,
        beta0: float | None = 1.0,
        c_min: int = 5,
        random_state: int = 42,
        split_criterion: CustomSplitCriterion = "bernoulli_log_likelihood",
        prior_mean: float | None = None,
        prior_strength: float | None = None,
        can_update_online: bool | None = True,
    ):
        BasePolicy.__init__(self, can_update_online=can_update_online)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = None if min_samples_leaf is None else int(min_samples_leaf)
        self.c_min = int(c_min)
        self.random_state = int(random_state)
        self.split_criterion = split_criterion
        self.alpha0, self.beta0 = CustomActionTreeThompsonModel._resolve_prior(alpha0, beta0, prior_mean, prior_strength)
        self.action_models: dict[int, CustomActionTreeThompsonModel] = {}
        self.action_history: dict[int, list[tuple[list[float], float, int]]] = {}

    def _build_model(self) -> CustomActionTreeThompsonModel:
        return CustomActionTreeThompsonModel(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            alpha0=self.alpha0,
            beta0=self.beta0,
            c_min=self.c_min,
            random_state=self.random_state,
            split_criterion=self.split_criterion,
        )

    def _parse_update(self, update) -> tuple[int, float, list[float]]:
        if len(update) != 3:
            raise ValueError("Updates must be (action, reward, features)")
        action, reward, features = update
        return int(action), float(reward), [float(v) for v in features]

    def fit(self, train_df: pl.DataFrame) -> None:
        import numpy as np

        rows = list(train_df.iter_rows(named=True))
        self.action_models = {}
        self.action_history = {}

        for row in rows:
            action = int(row["show"])
            features = [float(v) for v in row["features_list"]]
            reward = float(row["reward"])
            self.action_history.setdefault(action, []).append((features, reward, action))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=float)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model().fit(X, y)

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        return TreeThompsonSamplingPolicy.select(self, candidates, features, row)

    def get_action_proba(
        self,
        candidates: list[Action],
        action: Action,
        features: list[float] | None = None,
        row: dict[str, object] | None = None,
    ) -> float:
        return TreeThompsonSamplingPolicy.get_action_proba(self, candidates, action, features, row)

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        grouped: dict[int, list[tuple[list[float], float, int]]] = {}
        for update in pending_updates:
            action, reward, features = self._parse_update(update)
            self.action_history.setdefault(action, []).append((features, reward, action))
            grouped.setdefault(action, []).append((features, reward, action))

        for action, items in grouped.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=float)
            if action in self.action_models and self.action_models[action].tree is not None:
                self.action_models[action].update_batch(X, y)
            else:
                all_items = self.action_history.get(action, items)
                Xa = np.asarray([it[0] for it in all_items], dtype=float)
                ya = np.asarray([1 if it[1] > 0 else 0 for it in all_items], dtype=float)
                if len(Xa) == 0:
                    continue
                self.action_models[action] = self._build_model().fit(Xa, ya)


class CustomTreeThompsonSamplingPolicyUpdateV1(CustomTreeThompsonSamplingPolicy):
    """Incremental custom-tree updates via CustomActionTreeThompsonModel.update_batch."""


class CustomTreeThompsonSamplingPolicyDummyRefit(CustomTreeThompsonSamplingPolicy):
    """Custom-tree variant that refits on full action history after each update."""

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for update in pending_updates:
            action, reward, features = self._parse_update(update)
            self.action_history.setdefault(action, []).append((features, reward, action))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=float)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model().fit(X, y)
