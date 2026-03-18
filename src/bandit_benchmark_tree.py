"""Tree-based Thompson sampling models and policies."""

from __future__ import annotations

import polars as pl

from bandit_benchmark_basic import Action, BasePolicy


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
