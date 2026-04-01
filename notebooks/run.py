from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_benchmark import load_dataset
from bandit_benchmark import (
    CatBoostPolicy,
    LogisticTSLibPolicy,
    LaplaceThompsonViaBayesianLogRegPolicy,
    NeuralLaplaceThompsonViaBayesianLogRegPolicy,
    PartitionedTSLibPolicy,
    RandomPolicy,
    EpsilonGreedyPolicy,
    UCBPolicy,
    ThompsonSamplingPolicy,
    TreeThompsonSamplingPolicy,
    TreeThompsonSamplingPolicyDummyRefit,
    TreeThompsonSamplingPolicyUpdateV1,
    CustomTreeThompsonSamplingPolicyUpdateV1,
    _NeuralActionRewardEncoder,
    build_expected_reward_estimator,
    default_scenario,
    default_five_scenarios,
    make_simulated_environment,
    preprocess_bandit_dataframe,
    run_scenarios,
    run_scenarios_ips,
    split_train_test_by_date,
)
from prepare_datasets import stage1_make_splits, stage2_scale_features
# Укажите путь к данным (.tsv или .parquet)
DATA_PATH = ROOT / 'data' / 'banner_small.tsv'
DATASET_NAME = DATA_PATH.stem
ARTIFACTS_DIR = ROOT / 'artifacts' / DATASET_NAME
DATASETS_DIR = ARTIFACTS_DIR / 'datasets'
PREPARED_TRAIN_PATH = DATASETS_DIR / 'train_variant_1_scaled.parquet'
PREPARED_TEST_PATH = DATASETS_DIR / 'test_variant_1_scaled.parquet'
USE_PREPARED_SPLITS = True
TRAIN_DAYS = 20
EPSILON = 0.1
SEED = 42
SIMULATE = False
STOCHASTIC_SIM = True
FULL_SCENARIOS = False
NEURAL_HIDDEN_DIMS = [128, 68]
USE_SAMPLED_DATASET = False
SAMPLE_FRACTION = 0.2  # доля строк для train/test при демо-прогоне
USE_CUSTOM_ENCODER = True
ENCODER_REP_DIM = 32
ENCODER_NN_LR = 1e-3
ENCODER_TRAIN_DATA_MODE = 'random_half'  # all | random_half | time_half


BOOTSTRAP_ENABLED = True
BOOTSTRAP_ITERATIONS = 100
BOOTSTRAP_CI = 0.95


TREE_UPDATE_C_MIN_GRID = [1, 5]
TREE_UPDATE_MIN_SAMPLES_LEAF_GRID = [150, 300]
TREE_UPDATE_MAX_DEPTH_GRID = [3, 4]
if USE_PREPARED_SPLITS and PREPARED_TRAIN_PATH.exists() and PREPARED_TEST_PATH.exists():
    train_df = pl.read_parquet(PREPARED_TRAIN_PATH)
    test_df = pl.read_parquet(PREPARED_TEST_PATH)
    print('loaded prepared splits')
else:
    # Stage 1: load -> split -> filter -> save
    train_s1, test_s1 = stage1_make_splits(str(DATA_PATH), str(DATASETS_DIR), TRAIN_DAYS, SEED)
    # Stage 2: scale features and save prepared datasets
    train_final, test_final = stage2_scale_features(train_s1, test_s1, str(DATASETS_DIR))
    train_df = pl.read_parquet(train_final)
    test_df = pl.read_parquet(test_final)

print('train:', train_df.height, 'test(random only):', test_df.height)

policy_factories = {
    "test1": lambda: CustomTreeThompsonSamplingPolicyUpdateV1(
        random_state=SEED,
        c_min=20,
        min_samples_leaf=3000,
        max_depth=3,
        split_criterion = "beta_marginal_likelihood"
    ),
    "test2": lambda: CustomTreeThompsonSamplingPolicyUpdateV1(
        random_state=SEED,
        c_min=20,
        min_samples_leaf=3000,
        max_depth=3,
        split_criterion = "bernoulli_log_likelihood"
    ),
    
    'thompson_sampling': lambda: ThompsonSamplingPolicy(seed=SEED)
}

scenarios = default_five_scenarios() if FULL_SCENARIOS else default_scenario()

if USE_SAMPLED_DATASET:
    if not (0.0 < SAMPLE_FRACTION <= 1.0):
        raise ValueError('SAMPLE_FRACTION must be in (0, 1]')
    train_eval_df = train_df.sample(fraction=SAMPLE_FRACTION, shuffle=True, seed=SEED).sort('date')
    test_eval_df = test_df.sample(fraction=SAMPLE_FRACTION, shuffle=True, seed=SEED).sort('date')
else:
    train_eval_df = train_df
    test_eval_df = test_df

print('eval train rows:', train_eval_df.height, 'of', train_df.height)
print('eval test rows:', test_eval_df.height, 'of', test_df.height)

env_reward = None
if SIMULATE:
    expected_fn = build_expected_reward_estimator(train_df)
    env_reward = make_simulated_environment(
        proba_predictor=expected_fn,
        stochastic=STOCHASTIC_SIM,
        seed=SEED,
    )

result = run_scenarios(
    train_df=train_eval_df,
    test_df=test_eval_df,
    policy_factories=policy_factories,
    scenarios=scenarios,
    env_reward=env_reward,
    show_progress=True,
)

metrics_df = result['metrics']
history_df = result['history']
action_stats_df = result['action_stats']
action_daily_stats_df = result['action_daily_stats']
action_sensitive_stats_df = result['action_sensitive_stats']

display(metrics_df.sort_values('ips_ctr', ascending=False).reset_index(drop=True))

ips_snips_cols = ['scenario', 'algo', 'impressions_total', 'impressions_extrapolated', 'ips_ctr', 'snips_ctr']
if set(ips_snips_cols).issubset(metrics_df.columns):
    print('IPS/SNIPS metrics (run_scenarios):')
    display(metrics_df[ips_snips_cols].sort_values(['scenario', 'algo']).reset_index(drop=True))

sensitive_cols = ['scenario', 'algo', 'sensitive_impressions', 'ips_ctr_sensitive', 'ips_regret_sens']
if set(sensitive_cols).issubset(metrics_df.columns):
    print('sensitive IPS metrics (rows with len(candidates) > 1):')
    display(metrics_df[sensitive_cols].sort_values(['ips_ctr_sensitive', 'scenario', 'algo'], ascending=[False, True, True]).reset_index(drop=True))

print('history rows:', len(history_df))
if SIMULATE:
    display(action_stats_df)
    display(action_daily_stats_df)
    display(action_sensitive_stats_df)
