"""Benchmark helpers for logged ad bandit data.

This module focuses on experiment mechanics requested for ad-offer selection:
- train/test split
- comparison for epsilon-greedy, UCB, Thompson Sampling, CatBoost policy
- contextual bandit placeholder (not implemented yet)
- 5 experiment scenarios (pretrain source x online updates)
- optional simulated environment evaluation (if you can estimate rewards for all actions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
import random
from typing import Callable, Iterable, Literal


Action = int
Features = dict[str, object]


@dataclass
class BanditEvent:
    policy: str
    reward: int
    puid: str
    features: Features
    show: Action
    candidates: list[Action]
    date: str
    propensity: float | None = None
    rewards_by_action: dict[Action, int] | None = None

    def event_date(self) -> date:
        return date.fromisoformat(self.date)


@dataclass
class EvalMetrics:
    steps: int
    impressions_used: int
    total_reward: float
    ctr: float
    replay_match_rate: float


@dataclass
class ScenarioResult:
    scenario_name: str
    metrics_by_algo: dict[str, EvalMetrics]


class BasePolicy:
    can_update_online: bool = True

    def select(self, event: BanditEvent) -> Action:
        raise NotImplementedError

    def update(self, action: Action, reward: float) -> None:
        del action, reward

    def fit(self, events: Iterable[BanditEvent]) -> None:
        for ev in events:
            self.update(ev.show, ev.reward)


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts: dict[Action, int] = {}
        self.values: dict[Action, float] = {}

    def select(self, event: BanditEvent) -> Action:
        cands = event.candidates
        if not cands:
            raise ValueError("Empty candidate set")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(cands)

        def score(a: Action) -> float:
            return self.values.get(a, 0.0)

        return max(cands, key=score)

    def update(self, action: Action, reward: float) -> None:
        n = self.counts.get(action, 0) + 1
        v = self.values.get(action, 0.0)
        self.values[action] = v + (reward - v) / n
        self.counts[action] = n


class UCBPolicy(BasePolicy):
    def __init__(self, exploration: float = 2.0):
        self.exploration = exploration
        self.t = 0
        self.counts: dict[Action, int] = {}
        self.values: dict[Action, float] = {}

    def select(self, event: BanditEvent) -> Action:
        cands = event.candidates
        if not cands:
            raise ValueError("Empty candidate set")
        for a in cands:
            if self.counts.get(a, 0) == 0:
                return a

        log_t = math.log(max(self.t, 1))

        def ucb(a: Action) -> float:
            n = self.counts[a]
            mean = self.values[a]
            bonus = math.sqrt(self.exploration * log_t / n)
            return mean + bonus

        return max(cands, key=ucb)

    def update(self, action: Action, reward: float) -> None:
        self.t += 1
        n = self.counts.get(action, 0) + 1
        v = self.values.get(action, 0.0)
        self.values[action] = v + (reward - v) / n
        self.counts[action] = n


class ThompsonSamplingPolicy(BasePolicy):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, seed: int = 42):
        self.alpha0 = alpha
        self.beta0 = beta
        self.rng = random.Random(seed)
        self.alpha: dict[Action, float] = {}
        self.beta: dict[Action, float] = {}

    def select(self, event: BanditEvent) -> Action:
        cands = event.candidates
        if not cands:
            raise ValueError("Empty candidate set")

        def draw(a: Action) -> float:
            return self.rng.betavariate(self.alpha.get(a, self.alpha0), self.beta.get(a, self.beta0))

        return max(cands, key=draw)

    def update(self, action: Action, reward: float) -> None:
        self.alpha[action] = self.alpha.get(action, self.alpha0) + reward
        self.beta[action] = self.beta.get(action, self.beta0) + (1.0 - reward)


class CatBoostPolicy(BasePolicy):
    """Wrapper around external scorer (e.g., CatBoost model).

    scorer(event, action) -> predicted click probability.
    """

    can_update_online = False

    def __init__(self, scorer: Callable[[BanditEvent, Action], float]):
        self.scorer = scorer

    def select(self, event: BanditEvent) -> Action:
        if not event.candidates:
            raise ValueError("Empty candidate set")
        return max(event.candidates, key=lambda a: self.scorer(event, a))


class ContextualBanditPlaceholder(BasePolicy):
    def select(self, event: BanditEvent) -> Action:
        raise NotImplementedError("Contextual bandits are intentionally not implemented yet")


def split_train_test_by_date(
    events: list[BanditEvent],
    test_ratio: float = 0.2,
) -> tuple[list[BanditEvent], list[BanditEvent]]:
    """Time-based split by event date."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0,1)")
    events_sorted = sorted(events, key=lambda e: e.event_date())
    split_idx = int(len(events_sorted) * (1.0 - test_ratio))
    return events_sorted[:split_idx], events_sorted[split_idx:]


def select_pretrain_data(events: list[BanditEvent], source: Literal["random", "all", "none"]) -> list[BanditEvent]:
    if source == "none":
        return []
    if source == "all":
        return events
    if source == "random":
        return [ev for ev in events if ev.policy == "random"]
    raise ValueError(f"Unknown pretrain source: {source}")


def evaluate_policy(
    policy: BasePolicy,
    test_events: list[BanditEvent],
    online_update: bool,
    env_reward: Callable[[BanditEvent, Action], float] | None = None,
) -> EvalMetrics:
    """Evaluate one policy on test events.

    If env_reward is None, replay-style evaluation is used:
    - reward is observable only when selected action equals logged `show`.

    If env_reward is provided, reward for any chosen action is available.
    """

    total_reward = 0.0
    used = 0
    replay_matches = 0

    for ev in test_events:
        action = policy.select(ev)

        if env_reward is None:
            if action != ev.show:
                continue
            reward = float(ev.reward)
            replay_matches += 1
        else:
            reward = float(env_reward(ev, action))
            if action == ev.show:
                replay_matches += 1

        total_reward += reward
        used += 1

        if online_update and policy.can_update_online:
            policy.update(action, reward)

    ctr = total_reward / used if used else 0.0
    match_rate = replay_matches / len(test_events) if test_events else 0.0
    return EvalMetrics(
        steps=len(test_events),
        impressions_used=used,
        total_reward=total_reward,
        ctr=ctr,
        replay_match_rate=match_rate,
    )


def run_five_scenarios(
    train_events: list[BanditEvent],
    test_events: list[BanditEvent],
    policy_factories: dict[str, Callable[[], BasePolicy]],
    env_reward: Callable[[BanditEvent, Action], float] | None = None,
) -> list[ScenarioResult]:
    """Run required cases:
    1) pretrain random + predict-only
    2) pretrain random + online updates (except CatBoost)
    3) pretrain all + predict-only
    4) pretrain all + online updates (except CatBoost)
    5) no train + online updates (except CatBoost)
    """

    scenarios: list[tuple[str, Literal["random", "all", "none"], bool]] = [
        ("case_1_random_pretrain_predict_only", "random", False),
        ("case_2_random_pretrain_online_update", "random", True),
        ("case_3_all_pretrain_predict_only", "all", False),
        ("case_4_all_pretrain_online_update", "all", True),
        ("case_5_no_pretrain_online_update", "none", True),
    ]

    outputs: list[ScenarioResult] = []

    for scenario_name, pretrain_source, online_update in scenarios:
        metrics_by_algo: dict[str, EvalMetrics] = {}
        pretrain_subset = select_pretrain_data(train_events, pretrain_source)

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_subset:
                policy.fit(pretrain_subset)
            metrics = evaluate_policy(
                policy=policy,
                test_events=test_events,
                online_update=online_update,
                env_reward=env_reward,
            )
            metrics_by_algo[algo_name] = metrics

        outputs.append(ScenarioResult(scenario_name=scenario_name, metrics_by_algo=metrics_by_algo))

    return outputs


def make_simulated_environment(
    proba_predictor: Callable[[BanditEvent, Action], float],
    stochastic: bool = True,
    seed: int = 42,
) -> Callable[[BanditEvent, Action], float]:
    """Build environment callback from a learned reward model.

    proba_predictor should return P(click | x, action).
    """
    rng = random.Random(seed)

    def env_reward(ev: BanditEvent, action: Action) -> float:
        p = max(0.0, min(1.0, float(proba_predictor(ev, action))))
        if not stochastic:
            return p
        return 1.0 if rng.random() < p else 0.0

    return env_reward
