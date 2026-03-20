"""Benchmark helpers for logged ad bandit data (polars processing + pandas metrics)."""

from __future__ import annotations

from prepare_datasets import (
    apply_standard_scaler_to_features,
    filter_test_by_train_candidate_coverage,
    preprocess_bandit_dataframe,
    split_train_test_by_date,
)

from bandit_benchmark_basic import (
    Action,
    BasePolicy,
    EpsilonGreedyPolicy,
    RandomPolicy,
    ScenarioConfig,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from bandit_benchmark_eval import (
    build_expected_reward_estimator,
    build_random_action_ctr_stats,
    core_scenarios,
    default_five_scenarios,
    default_scenario,
    evaluate_policy,
    evaluate_policy_ips,
    make_simulated_environment,
    run_scenarios,
    run_scenarios_ips,
    select_pretrain_data,
    two_ways_default_scenario,
)
from bandit_benchmark_external import (
    CatBoostPolicy,
    ContextualBanditPlaceholder,
    LogisticTSLibPolicy,
    LogisticTSPolicy,
    PartitionedTSLibPolicy,
    PartitionedTSPolicy,
)
from bandit_benchmark_linear import (
    LaplaceThompsonViaBayesianLogRegPolicy,
    NeuralLaplaceThompsonViaBayesianLogRegPolicy,
    OnlineLogisticRegression,
    _NeuralActionRewardEncoder,
)
from bandit_benchmark_tree import (
    ActionTreeThompsonModel,
    TreeThompsonSamplingPolicy,
    TreeThompsonSamplingPolicyDummyRefit,
    TreeThompsonSamplingPolicyUpdateV1,
    TreeThompsonSamplingPolicyUpdateV2,
    TreeThompsonSamplingPolicyUpdateV3,
)
from bandit_benchmark_custom_tree import (
    CustomActionTreeThompsonModel,
    CustomTreeThompsonSamplingPolicy,
    CustomTreeThompsonSamplingPolicyDummyRefit,
    CustomTreeThompsonSamplingPolicyUpdateV1,
    render_custom_tree_text,
)

TreeThompsonSamplingPolicyV1 = TreeThompsonSamplingPolicyUpdateV1
