"""External-library backed benchmark policies."""

from __future__ import annotations

import polars as pl

from bandit_benchmark_basic import Action, BasePolicy, normalize_action


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
        actions = sorted({normalize_action(r["show"]) for r in rows})
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

class CatBoostPolicy(BasePolicy):
    """Gradient boosting policy based on CatBoostClassifier.

    Model is train-once: repeated fit calls are forbidden.
    """

    can_update_online = False

    def __init__(
        self,
        random_seed: int = 42,
        iterations: int = 200,
        depth: int = 6,
        learning_rate: float = 0.05,
        can_update_online: bool | None = None,
    ):
        super().__init__(can_update_online=can_update_online)
        self.random_seed = random_seed
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
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

        actions = sorted({normalize_action(r["show"]) for r in train_df.iter_rows(named=True)})
        self._actions = actions
        self._a2i = {a: i for i, a in enumerate(actions)}

        X: list[list[float]] = []
        y: list[int] = []
        for row in train_df.iter_rows(named=True):
            action = normalize_action(row["show"])
            features = row["features_list"]
            X.append(self._row_to_vector(features, action))
            y.append(int(float(row["reward"]) > 0.0))

        if not X:
            self._fitted = True
            return

        model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
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


# Backward-compatible aliases
LogisticTSPolicy = LogisticTSLibPolicy
PartitionedTSPolicy = PartitionedTSLibPolicy
