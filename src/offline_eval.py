"""Minimal offline policy evaluation utilities for logged bandit data.

Expected columns in a pandas DataFrame:
- reward: 0/1
- show: logged action id
- candidates: iterable of action ids available at request
- propensity: probability logging policy selected `show`
- q_hat: optional reward model prediction for logged action (for DR)

Policy interface:
- policy_prob(row, action) -> probability target policy picks `action` for row
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


PolicyProbFn = Callable[[pd.Series, int], float]


@dataclass
class OPEstimates:
    ips: float
    snips: float
    dr: float | None


def _safe_propensity(p: pd.Series, clip_min: float) -> np.ndarray:
    return np.clip(p.to_numpy(dtype=float), clip_min, 1.0)


def estimate_ope(
    df: pd.DataFrame,
    policy_prob: PolicyProbFn,
    clip_min: float = 1e-3,
    use_dr: bool = False,
) -> OPEstimates:
    """Estimate policy value with IPS/SNIPS and optional DR.

    Notes:
    - Assumes support overlap: target policy only puts mass on actions possible under logging.
    - For DR, dataframe must contain `q_hat` for logged action and a column
      `q_hat_by_action` (dict-like action->predicted reward for available actions).
    """

    req = {"reward", "show", "propensity", "candidates"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    p = _safe_propensity(df["propensity"], clip_min=clip_min)
    pi = np.array([policy_prob(row, int(row["show"])) for _, row in df.iterrows()], dtype=float)
    w = pi / p
    r = df["reward"].to_numpy(dtype=float)

    ips = float(np.mean(w * r))
    denom = np.sum(w)
    snips = float(np.sum(w * r) / denom) if denom > 0 else 0.0

    dr_val: float | None = None
    if use_dr:
        if "q_hat" not in df.columns or "q_hat_by_action" not in df.columns:
            raise ValueError("DR requires columns q_hat and q_hat_by_action")

        q_logged = df["q_hat"].to_numpy(dtype=float)
        correction = w * (r - q_logged)

        direct_terms = []
        for _, row in df.iterrows():
            q_map = row["q_hat_by_action"]
            val = 0.0
            for a in row["candidates"]:
                pa = policy_prob(row, int(a))
                val += pa * float(q_map.get(int(a), 0.0))
            direct_terms.append(val)

        dr_val = float(np.mean(np.array(direct_terms) + correction))

    return OPEstimates(ips=ips, snips=snips, dr=dr_val)


def uniform_policy_prob(row: pd.Series, action: int) -> float:
    """Uniform policy over the logged candidate set."""
    cands = list(row["candidates"])
    if action not in cands or len(cands) == 0:
        return 0.0
    return 1.0 / len(cands)
