"""Linear/logistic and neural Thompson sampling policies."""

from __future__ import annotations

from typing import Literal

import polars as pl

from bandit_benchmark_basic import Action, BasePolicy, normalize_action


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
        a = normalize_action(action)
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
            (normalize_action(r["show"]), float(r["reward"]), list(r["features_list"]))
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
        best_a = normalize_action(candidates[0])
        best_score = -np.inf

        for a_raw in candidates:
            a = normalize_action(a_raw)
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
        normalized_action = normalize_action(action)
        normalized_candidates = {normalize_action(a) for a in candidates}
        if features is None or not candidates or normalized_action not in normalized_candidates:
            return 0.0
        self._ensure_dim(features)
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        n_mc = 128
        wins = 0
        target = normalized_action
        for _ in range(n_mc):
            best_a = normalize_action(candidates[0])
            best_score = -np.inf
            for a_raw in candidates:
                a = normalize_action(a_raw)
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

        actions = sorted({normalize_action(r["show"]) for r in rows})
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
        a_idx_enc = np.asarray([self._action_to_idx[normalize_action(r["show"])] for r in encoder_rows], dtype=np.int64)
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
            (normalize_action(r["show"]), float(r["reward"]), [float(v) for v in Z_reg[i].tolist()])
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
        normalized_action = normalize_action(action)
        normalized_candidates = {normalize_action(a) for a in candidates}
        if features is None or not candidates or normalized_action not in normalized_candidates:
            return 0.0
        transformed = self._transform_features(features)
        return self._base.get_action_proba(candidates, action, transformed, row)
