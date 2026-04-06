"""Tree-based Thompson sampling models and policies."""

from __future__ import annotations

import math

import polars as pl

from bandit_benchmark_basic import Action, BasePolicy, normalize_action


class ActionTreeThompsonModel:
    """Original sklearn-based tree Thompson model."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int | None = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        c_min: int = 5,
        use_c_min_fallback: bool = True,
        random_state: int = 42,
    ):
        import numpy as np

        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.c_min = int(c_min)
        self.use_c_min_fallback = bool(use_c_min_fallback)
        self.random_state = int(random_state)

        self.tree = None
        self.leaf_stats: dict[int, dict[str, float | int]] = {}
        self.global_alpha: float | None = None
        self.global_beta: float | None = None
        self.is_tree_trained = False
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
        self.is_tree_trained = bool(len(np.unique(leaf_ids)) >= 2)
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

        if self.use_c_min_fallback and int(stats["clicks"]) < self.c_min:
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
        use_c_min_fallback: bool = True,
        random_state: int = 42,
        can_update_online: bool | None = True,
    ):
        super().__init__(can_update_online=can_update_online)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = None if min_samples_leaf is None else int(min_samples_leaf)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.c_min = int(c_min)
        self.use_c_min_fallback = bool(use_c_min_fallback)
        self.random_state = int(random_state)

        self.action_models: dict[int, ActionTreeThompsonModel] = {}
        self.action_history: dict[int, list[tuple[list[float], float, int]]] = {}

    def _resolve_min_samples_leaf(self, y) -> int:
        if self.min_samples_leaf is not None:
            return max(1, int(self.min_samples_leaf))
        if self.c_min <= 0:
            return 1
        ctr = float(y.mean()) if len(y) else 0.0
        ctr = max(ctr, 1e-6)
        return max(1, int(math.ceil(self.c_min / ctr)))

    def _build_model(self, resolved_min_samples_leaf: int) -> ActionTreeThompsonModel:
        return ActionTreeThompsonModel(
            max_depth=self.max_depth,
            min_samples_leaf=resolved_min_samples_leaf,
            alpha0=self.alpha0,
            beta0=self.beta0,
            c_min=self.c_min,
            use_c_min_fallback=self.use_c_min_fallback,
            random_state=self.random_state,
        )

    def fit(self, train_df: pl.DataFrame) -> None:
        import numpy as np

        rows = list(train_df.iter_rows(named=True))
        self.action_models = {}
        self.action_history = {}

        for row in rows:
            action = normalize_action(row["show"])
            features = [float(v) for v in row["features_list"]]
            reward = float(row["reward"])
            self.action_history.setdefault(action, []).append((features, reward, action))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model(self._resolve_min_samples_leaf(y)).fit(X, y)

    def select(self, candidates: list[Action], features: list[float], row: dict[str, object]) -> Action:
        import numpy as np

        del row
        if not candidates:
            raise ValueError("Empty candidate set")

        x = np.asarray(features, dtype=float)
        best_action = normalize_action(candidates[0])
        best_score = -1.0
        rng = np.random.default_rng(self.random_state)

        for candidate in candidates:
            action = normalize_action(candidate)
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
        normalized_action = normalize_action(action)
        normalized_candidates = {normalize_action(a) for a in candidates}
        if features is None or not candidates or normalized_action not in normalized_candidates:
            return 0.0

        x = np.asarray(features, dtype=float)
        rng = np.random.default_rng(self.random_state)
        target = normalized_action
        wins = 0
        n_mc = 128

        for _ in range(n_mc):
            best_action = normalize_action(candidates[0])
            best_score = -1.0
            for candidate in candidates:
                aa = normalize_action(candidate)
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
            aa = normalize_action(action)
            ff = [float(v) for v in features]
            rr = float(reward)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        grouped: dict[Action, list[tuple[list[float], float, Action]]] = {}
        for action, reward, features in pending_updates:
            aa = normalize_action(action)
            grouped.setdefault(aa, []).append(([float(v) for v in features], float(reward), aa))

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
                self.action_models[action] = self._build_model(self._resolve_min_samples_leaf(ya)).fit(Xa, ya)


class TreeThompsonSamplingPolicyUpdateV1(TreeThompsonSamplingPolicy):
    """Incremental sklearn-tree updates via ActionTreeThompsonModel.update_batch."""


class TreeThompsonSamplingPolicyUpdateV2(TreeThompsonSamplingPolicy):
    """Incremental sklearn-tree updates that retry fitting until the action-tree becomes split/trained."""

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for action, reward, features in pending_updates:
            aa = normalize_action(action)
            ff = [float(v) for v in features]
            rr = float(reward)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        grouped: dict[Action, list[tuple[list[float], float, Action]]] = {}
        for action, reward, features in pending_updates:
            aa = normalize_action(action)
            grouped.setdefault(aa, []).append(([float(v) for v in features], float(reward), aa))

        for action, items in grouped.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            model = self.action_models.get(action)

            if model is not None and model.tree is not None and model.is_tree_trained:
                model.update_batch(X, y)
                continue

            all_items = self.action_history.get(action, items)
            Xa = np.asarray([it[0] for it in all_items], dtype=float)
            ya = np.asarray([1 if it[1] > 0 else 0 for it in all_items], dtype=int)
            if len(Xa) == 0:
                continue
            self.action_models[action] = self._build_model(self._resolve_min_samples_leaf(ya)).fit(Xa, ya)


class TreeThompsonSamplingPolicyUpdateV3(TreeThompsonSamplingPolicy):
    """Incremental sklearn-tree updates with full refits every N update_batch calls."""

    def __init__(self, *args, retrain_every_n_updates: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrain_every_n_updates = max(1, int(retrain_every_n_updates))
        self._update_batch_calls = 0

    def _refit_all_models(self) -> None:
        import numpy as np

        self.action_models = {}
        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model(self._resolve_min_samples_leaf(y)).fit(X, y)

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for action, reward, features in pending_updates:
            aa = normalize_action(action)
            ff = [float(v) for v in features]
            rr = float(reward)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        self._update_batch_calls += 1
        if self._update_batch_calls % self.retrain_every_n_updates == 0:
            self._refit_all_models()
            return

        grouped: dict[Action, list[tuple[list[float], float, Action]]] = {}
        for action, reward, features in pending_updates:
            aa = normalize_action(action)
            grouped.setdefault(aa, []).append(([float(v) for v in features], float(reward), aa))

        for action, items in grouped.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)

            model = self.action_models.get(action)

            if model is not None and model.tree is not None and model.is_tree_trained:
                model.update_batch(X, y)
                continue

            all_items = self.action_history.get(action, items)
            Xa = np.asarray([it[0] for it in all_items], dtype=float)
            ya = np.asarray([1 if it[1] > 0 else 0 for it in all_items], dtype=int)
            if len(Xa) == 0:
                continue
            self.action_models[action] = self._build_model(self._resolve_min_samples_leaf(ya)).fit(Xa, ya)


class TreeThompsonSamplingPolicyDummyRefit(TreeThompsonSamplingPolicy):
    """Sklearn-tree variant that refits on full action history after each update."""

    def update_batch(self, pending_updates) -> None:
        import numpy as np

        if not pending_updates:
            return

        for action, reward, features in pending_updates:
            aa = normalize_action(action)
            ff = [float(v) for v in features]
            rr = float(reward)
            self.action_history.setdefault(aa, []).append((ff, rr, aa))

        for action, items in self.action_history.items():
            X = np.asarray([it[0] for it in items], dtype=float)
            y = np.asarray([1 if it[1] > 0 else 0 for it in items], dtype=int)
            if len(X) == 0:
                continue
            self.action_models[action] = self._build_model(self._resolve_min_samples_leaf(y)).fit(X, y)
