"""Scenario evaluation and benchmarking helpers."""

from __future__ import annotations

import random
from typing import Callable, Literal

import pandas as pd
import polars as pl

from bandit_benchmark_basic import Action, BasePolicy, ScenarioConfig

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def select_pretrain_data(train_df: pl.DataFrame, source: Literal["random", "all", "none"]) -> pl.DataFrame:
    if source == "none":
        return train_df.clear()
    if source == "all":
        return train_df
    if source == "random":
        return train_df.filter(pl.col("policy") == "random")
    raise ValueError(f"Unknown pretrain source: {source}")

def build_expected_reward_estimator(train_df: pl.DataFrame) -> Callable[[dict[str, object], Action], float]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in train_df.iter_rows(named=True):
        a = int(row["show"])
        r = float(row["reward"])
        sums[a] = sums.get(a, 0.0) + r
        counts[a] = counts.get(a, 0) + 1

    total_sum = sum(sums.values())
    total_n = sum(counts.values())
    global_mean = (total_sum / total_n) if total_n > 0 else 0.0

    def estimate(_row: dict[str, object], action: Action) -> float:
        n = counts.get(action, 0)
        return (sums[action] / n) if n > 0 else global_mean

    return estimate

def build_random_action_ctr_stats(df: pl.DataFrame) -> tuple[dict[int, float], float]:
    random_df = df.filter(pl.col("policy") == "random") if "policy" in df.columns else df
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in random_df.iter_rows(named=True):
        a = int(row["show"])
        r = float(row["reward"])
        sums[a] = sums.get(a, 0.0) + r
        counts[a] = counts.get(a, 0) + 1

    ctr_by_action: dict[int, float] = {}
    for a, n in counts.items():
        ctr_by_action[a] = sums.get(a, 0.0) / n if n > 0 else 0.0
    max_ctr = max(ctr_by_action.values()) if ctr_by_action else 0.0
    return ctr_by_action, max_ctr

def evaluate_policy(
    policy: BasePolicy,
    test_df: pl.DataFrame,
    online_update: bool,
    update_frequency: Literal["daily", "step_2p5"] = "daily",
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
    progress_desc: str = "evaluate",
    ctr_by_action: dict[int, float] | None = None,
    max_random_ctr: float = 0.0,
    initial_seen_actions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_reward = 0.0
    ips_weighted_reward_sum = 0.0
    snips_weight_sum = 0.0
    ips_sensitive_reward_sum = 0.0
    ips_sensitive_regret_sum = 0.0
    sensitive_rows = 0
    used = 0
    replay_matches = 0
    cumulative_regret = 0.0  # replay regret accumulator (used only for regret metrics)
    cumulative_ips_regret = 0.0  # IPS regret accumulator (used only for regret metrics)
    history_rows: list[dict[str, float | int]] = []
    action_stats_rows: list[dict[str, int]] = []
    selected_action_rows: list[dict[str, object]] = []
    action_sensitive_stats: dict[int, dict[str, float | int | bool]] = {}

    action_ctr = ctr_by_action or {}
    # Regret-only baseline fallback for unseen actions within candidate sets.

    total_steps = max(test_df.height, 1)
    progress_chunk = max(1, int(total_steps * 0.05))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=test_df.height, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=0.5)

    pending_updates: list[tuple[int, float, list[float]]] = []
    next_progress_mark = progress_chunk
    update_chunk = max(1, int(total_steps * 0.025))
    next_step_update_mark = update_chunk

    sensitive_seen = 0
    sensitive_ips_reward_cum = 0.0
    sensitive_ips_regret_cum = 0.0
    seen_actions_total: set[int] = set(initial_seen_actions or set())
    current_day_actions: set[int] = set()

    test_rows = list(test_df.iter_rows(named=True))
    first_row_date = test_rows[0].get("date") if test_rows else None
    prev_date = first_row_date

    if show_progress:
        msg = f"{progress_desc}: train_unique_actions={len(seen_actions_total)}"
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    batch_size = 512

    for batch_start in range(0, len(test_rows), batch_size):
        rows_batch = test_rows[batch_start : batch_start + batch_size]
        candidates_batch = [row["candidates_list"] for row in rows_batch]
        features_batch = [row["features_list"] for row in rows_batch]
        actions_batch = policy.select_batch(candidates_batch, features_batch, rows_batch)
        if len(actions_batch) != len(rows_batch):
            raise ValueError("select_batch must return one action per input row")

        for offset, row in enumerate(rows_batch):
            step = batch_start + offset + 1
            candidates = candidates_batch[offset]
            features = features_batch[offset]
            current_date = row.get("date")
            if prev_date is not None and current_date is not None and current_date > prev_date:
                if update_frequency == "daily" and online_update and policy.can_update_online and pending_updates:
                    policy.update_batch(pending_updates)
                    pending_updates.clear()

                new_actions_in_day = current_day_actions - seen_actions_total
                seen_actions_total.update(current_day_actions)
                action_stats_rows.append(
                    {
                        "step": step,
                        "date": prev_date,
                        "unique_actions_total": len(seen_actions_total),
                        "unique_actions_new_in_day": len(new_actions_in_day),
                    }
                )
                current_day_actions.clear()

            prev_date = current_date if current_date is not None else prev_date

            action = int(actions_batch[offset])
            current_day_actions.add(action)

            selected_action_rows.append(
                {
                    "step": step,
                    "date": current_date if current_date is not None else prev_date,
                    "action": action,
                }
            )

            logged_reward = float(row["reward"])
            logged_match = int(action == int(row["show"]))
            propensity = float(row.get("propensity", 0.0) or 0.0)

            ips_weight = (logged_match / propensity) if propensity > 0 else 0.0
            ips_reward = ips_weight * logged_reward
            ips_weighted_reward_sum += ips_reward
            snips_weight_sum += ips_weight
            # Regret-only per-step baseline: max expected CTR among currently available actions.
            candidate_ctrs = [action_ctr.get(int(a), max_random_ctr) for a in candidates] if candidates else [max_random_ctr]
            step_max_ctr = max(candidate_ctrs) if candidate_ctrs else max_random_ctr
            ips_step_regret = step_max_ctr - (logged_reward if logged_match else 0.0)
            cumulative_ips_regret += ips_step_regret

            if len(candidates) > 1:
                sensitive_rows += 1
                ips_sensitive_reward_sum += ips_reward
                ips_sensitive_regret_sum += ips_step_regret
                sensitive_seen += 1
                sensitive_ips_reward_cum += ips_reward
                sensitive_ips_regret_cum += ips_step_regret

                action_stat = action_sensitive_stats.setdefault(
                    action,
                    {
                        "action": action,
                        "in_train": bool(action in (initial_seen_actions or set())),
                        "sensitive_impressions": 0,
                        "cumulative_sensitive_ips_reward": 0.0,
                    },
                )
                action_stat["sensitive_impressions"] = int(action_stat["sensitive_impressions"]) + 1
                action_stat["cumulative_sensitive_ips_reward"] = float(action_stat["cumulative_sensitive_ips_reward"]) + float(ips_reward)

            if env_reward is None:
                if not logged_match:
                    if pbar is not None and step >= next_progress_mark:
                        pbar.update(step - pbar.n)
                        next_progress_mark += progress_chunk
                    continue
                reward = logged_reward
                replay_matches += 1
                regret = step_max_ctr - reward
            else:
                reward = float(env_reward(row, action))
                replay_matches += int(action == int(row["show"]))
                regret = step_max_ctr - reward

            total_reward += reward
            cumulative_regret += regret
            used += 1

            if online_update and policy.can_update_online:
                pending_updates.append((action, reward, features))
                if update_frequency == "step_2p5" and step >= next_step_update_mark and pending_updates:
                    policy.update_batch(pending_updates)
                    pending_updates.clear()
                    while next_step_update_mark <= step:
                        next_step_update_mark += update_chunk

            history_rows.append(
                {
                    "step": step,
                    "reward": reward,
                    "avg_reward": total_reward / used,
                    "ips_reward": ips_reward,
                    "ips_avg_reward": ips_weighted_reward_sum / step,
                    "cumulative_regret": cumulative_regret,
                    "avg_regret": cumulative_regret / used,
                    "cumulative_ips_regret": cumulative_ips_regret,
                    "avg_ips_regret": cumulative_ips_regret / step,
                    "sensitive_impressions_so_far": sensitive_seen,
                    "ips_ctr_sensitive_so_far": (sensitive_ips_reward_cum / sensitive_seen) if sensitive_seen else 0.0,
                    "avg_ips_regret_sens_so_far": (sensitive_ips_regret_cum / sensitive_seen) if sensitive_seen else 0.0,
                }
            )

            if pbar is not None and step >= next_progress_mark:
                pbar.update(step - pbar.n)
                next_progress_mark += progress_chunk


    if current_day_actions:
        new_actions_in_day = current_day_actions - seen_actions_total
        seen_actions_total.update(current_day_actions)
        final_step = test_df.height
        action_stats_rows.append(
            {
                "step": final_step,
                "date": prev_date,
                "unique_actions_total": len(seen_actions_total),
                "unique_actions_new_in_day": len(new_actions_in_day),
            }
        )

    if pbar is not None:
        pbar.update(test_df.height - pbar.n)
        pbar.close()

    if online_update and policy.can_update_online and pending_updates:
        policy.update_batch(pending_updates)
        pending_updates.clear()

    ctr = total_reward / used if used else 0.0
    ips_ctr = ips_weighted_reward_sum / test_df.height if test_df.height else 0.0
    snips_ctr = (ips_weighted_reward_sum / snips_weight_sum) if snips_weight_sum > 0 else 0.0
    impressions_extrapolated = snips_weight_sum
    match_rate = replay_matches / test_df.height if test_df.height else 0.0
    final_avg_regret = (cumulative_regret / used) if used else 0.0
    final_avg_ips_regret = (cumulative_ips_regret / test_df.height) if test_df.height else 0.0
    ips_ctr_sensitive = (ips_sensitive_reward_sum / sensitive_rows) if sensitive_rows else 0.0
    ips_regret_sens = (ips_sensitive_regret_sum / sensitive_rows) if sensitive_rows else 0.0
    metrics_df = pd.DataFrame([
        {
            "impressions_total": test_df.height,
            "impressions_used": used,
            "total_reward": total_reward,
            "ctr": ctr,
            "ips_weighted_reward": ips_weighted_reward_sum,
            "ips_ctr": ips_ctr,
            "snips_ctr": snips_ctr,
            "impressions_extrapolated": impressions_extrapolated,
            "replay_match_rate": match_rate,
            "cumulative_regret": cumulative_regret,
            "avg_regret": final_avg_regret,
            "cumulative_ips_regret": cumulative_ips_regret,
            "avg_ips_regret": final_avg_ips_regret,
            "sensitive_impressions": sensitive_rows,
            "ips_ctr_sensitive": ips_ctr_sensitive,
            "ips_regret_sens": ips_regret_sens,
        }
    ])
    history_df = pd.DataFrame(history_rows)
    action_stats_df = pd.DataFrame(action_stats_rows)

    selected_df = pd.DataFrame(selected_action_rows)
    if not selected_df.empty:
        action_daily_stats_df = selected_df.groupby(["date", "action"], as_index=False).agg(
            impressions_selected=("action", "size")
        )
    else:
        action_daily_stats_df = pd.DataFrame(columns=["date", "action", "impressions_selected"])

    if action_sensitive_stats:
        action_sensitive_df = pd.DataFrame(list(action_sensitive_stats.values()))
        action_sensitive_df["sensitive_ips_ctr"] = action_sensitive_df.apply(
            lambda r: (float(r["cumulative_sensitive_ips_reward"]) / int(r["sensitive_impressions"])) if int(r["sensitive_impressions"]) > 0 else 0.0,
            axis=1,
        )
        action_sensitive_df = action_sensitive_df.sort_values("sensitive_ips_ctr", ascending=False).reset_index(drop=True)
    else:
        action_sensitive_df = pd.DataFrame(columns=["action", "in_train", "sensitive_impressions", "cumulative_sensitive_ips_reward", "sensitive_ips_ctr"])

    return metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df

def run_scenarios(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    policy_factories: dict[str, Callable[[], BasePolicy]],
    scenarios: list[ScenarioConfig],
    env_reward: Callable[[dict[str, object], Action], float] | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    metrics_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    action_stats_parts: list[pd.DataFrame] = []
    action_daily_stats_parts: list[pd.DataFrame] = []
    action_sensitive_parts: list[pd.DataFrame] = []
    trained_models: dict[str, dict[str, BasePolicy]] = {}

    for scenario in scenarios:
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)
        trained_models[scenario.name] = {}
        # Regret-only statistics are estimated from combined train+test logs.
        ctr_source = pl.concat([train_df.select(["policy", "show", "reward"]), test_df.select(["policy", "show", "reward"])], how="vertical")
        ctr_by_action, max_random_ctr = build_random_action_ctr_stats(ctr_source)

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_df.height > 0:
                policy.fit(pretrain_df)

            initial_seen_actions = {int(r["show"]) for r in pretrain_df.iter_rows(named=True)}

            metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df = evaluate_policy(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                update_frequency=scenario.update_frequency,
                env_reward=env_reward,
                show_progress=show_progress,
                progress_desc=f"{scenario.name}/{algo_name}",
                ctr_by_action=ctr_by_action,
                max_random_ctr=max_random_ctr,
                initial_seen_actions=initial_seen_actions,
            )
            metrics_df["scenario"] = scenario.name
            metrics_df["algo"] = algo_name
            metrics_parts.append(metrics_df)
            trained_models[scenario.name][algo_name] = policy

            if not history_df.empty:
                history_df["scenario"] = scenario.name
                history_df["algo"] = algo_name
                history_parts.append(history_df)

            if not action_stats_df.empty:
                action_stats_df["scenario"] = scenario.name
                action_stats_df["algo"] = algo_name
                action_stats_parts.append(action_stats_df)

            if not action_daily_stats_df.empty:
                action_daily_stats_df["scenario"] = scenario.name
                action_daily_stats_df["algo"] = algo_name
                action_daily_stats_parts.append(action_daily_stats_df)

            if not action_sensitive_df.empty:
                action_sensitive_df["scenario"] = scenario.name
                action_sensitive_df["algo"] = algo_name
                action_sensitive_parts.append(action_sensitive_df)

    out_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    out_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    out_action_stats = pd.concat(action_stats_parts, ignore_index=True) if action_stats_parts else pd.DataFrame()
    out_action_daily_stats = pd.concat(action_daily_stats_parts, ignore_index=True) if action_daily_stats_parts else pd.DataFrame()
    out_action_sensitive_stats = pd.concat(action_sensitive_parts, ignore_index=True) if action_sensitive_parts else pd.DataFrame()
    return {
        "metrics": out_metrics,
        "history": out_history,
        "action_stats": out_action_stats,
        "action_daily_stats": out_action_daily_stats,
        "action_sensitive_stats": out_action_sensitive_stats,
        "trained_models": trained_models,
    }

def evaluate_policy_ips(
    policy: BasePolicy,
    test_df: pl.DataFrame,
    online_update: bool,
    update_frequency: Literal["daily", "step_2p5"] = "daily",
    show_progress: bool = True,
    progress_desc: str = "evaluate_ips",
    initial_seen_actions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ips_weighted_reward_sum = 0.0
    snips_weight_sum = 0.0
    ips_sensitive_reward_sum = 0.0
    sensitive_rows = 0

    history_rows: list[dict[str, float | int]] = []
    action_stats_rows: list[dict[str, int]] = []
    selected_action_rows: list[dict[str, object]] = []
    action_sensitive_stats: dict[int, dict[str, float | int | bool]] = {}

    total_steps = max(test_df.height, 1)
    progress_chunk = max(1, int(total_steps * 0.05))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=test_df.height, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=0.5)

    pending_updates: list[tuple[int, float, list[float]]] = []
    next_progress_mark = progress_chunk
    update_chunk = max(1, int(total_steps * 0.025))
    next_step_update_mark = update_chunk

    sensitive_seen = 0
    sensitive_ips_reward_cum = 0.0
    seen_actions_total: set[int] = set(initial_seen_actions or set())
    current_day_actions: set[int] = set()

    test_rows = list(test_df.iter_rows(named=True))
    first_row_date = test_rows[0].get("date") if test_rows else None
    prev_date = first_row_date

    if show_progress:
        msg = f"{progress_desc}: train_unique_actions={len(seen_actions_total)}"
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    for step, row in enumerate(test_rows, start=1):
        candidates = row["candidates_list"]
        features = row["features_list"]
        logged_action = int(row["show"])
        logged_reward = float(row["reward"])
        propensity = float(row.get("propensity", 0.0) or 0.0)

        current_date = row.get("date")
        if prev_date is not None and current_date is not None and current_date > prev_date:
            if update_frequency == "daily" and online_update and policy.can_update_online and pending_updates:
                policy.update_batch(pending_updates)
                pending_updates.clear()

            new_actions_in_day = current_day_actions - seen_actions_total
            seen_actions_total.update(current_day_actions)
            action_stats_rows.append(
                {
                    "step": step,
                    "date": prev_date,
                    "unique_actions_total": len(seen_actions_total),
                    "unique_actions_new_in_day": len(new_actions_in_day),
                }
            )
            current_day_actions.clear()

        prev_date = current_date if current_date is not None else prev_date
        current_day_actions.add(logged_action)

        selected_action_rows.append(
            {
                "step": step,
                "date": current_date if current_date is not None else prev_date,
                "action": logged_action,
            }
        )

        target_proba = float(policy.get_action_proba(candidates, logged_action, features, row))
        ips_weight = (target_proba / propensity) if propensity > 0 else 0.0
        ips_reward = ips_weight * logged_reward
        ips_weighted_reward_sum += ips_reward
        snips_weight_sum += ips_weight

        if len(candidates) > 1:
            sensitive_rows += 1
            ips_sensitive_reward_sum += ips_reward
            sensitive_seen += 1
            sensitive_ips_reward_cum += ips_reward

            action_stat = action_sensitive_stats.setdefault(
                logged_action,
                {
                    "action": logged_action,
                    "in_train": bool(logged_action in (initial_seen_actions or set())),
                    "sensitive_impressions": 0,
                    "cumulative_sensitive_ips_reward": 0.0,
                },
            )
            action_stat["sensitive_impressions"] = int(action_stat["sensitive_impressions"]) + 1
            action_stat["cumulative_sensitive_ips_reward"] = float(action_stat["cumulative_sensitive_ips_reward"]) + float(ips_reward)

        if online_update and policy.can_update_online:
            pending_updates.append((logged_action, logged_reward, features))
            if update_frequency == "step_2p5" and step >= next_step_update_mark and pending_updates:
                policy.update_batch(pending_updates)
                pending_updates.clear()
                while next_step_update_mark <= step:
                    next_step_update_mark += update_chunk

        history_rows.append(
            {
                "step": step,
                "ips_reward": ips_reward,
                "ips_avg_reward": ips_weighted_reward_sum / step,
                "impressions_extrapolated_so_far": snips_weight_sum,
                "sensitive_impressions_so_far": sensitive_seen,
                "ips_ctr_sensitive_so_far": (sensitive_ips_reward_cum / sensitive_seen) if sensitive_seen else 0.0,
            }
        )

        if pbar is not None and step >= next_progress_mark:
            pbar.update(step - pbar.n)
            next_progress_mark += progress_chunk

    if current_day_actions:
        new_actions_in_day = current_day_actions - seen_actions_total
        seen_actions_total.update(current_day_actions)
        final_step = test_df.height
        action_stats_rows.append(
            {
                "step": final_step,
                "date": prev_date,
                "unique_actions_total": len(seen_actions_total),
                "unique_actions_new_in_day": len(new_actions_in_day),
            }
        )

    if pbar is not None:
        pbar.update(test_df.height - pbar.n)
        pbar.close()

    if online_update and policy.can_update_online and pending_updates:
        policy.update_batch(pending_updates)
        pending_updates.clear()

    ips_ctr = ips_weighted_reward_sum / test_df.height if test_df.height else 0.0
    snips_ctr = (ips_weighted_reward_sum / snips_weight_sum) if snips_weight_sum > 0 else 0.0
    impressions_extrapolated = snips_weight_sum
    ips_ctr_sensitive = (ips_sensitive_reward_sum / sensitive_rows) if sensitive_rows else 0.0

    metrics_df = pd.DataFrame([
        {
            "impressions_total": test_df.height,
            "ips_weighted_reward": ips_weighted_reward_sum,
            "ips_ctr": ips_ctr,
            "snips_ctr": snips_ctr,
            "impressions_extrapolated": impressions_extrapolated,
            "sensitive_impressions": sensitive_rows,
            "ips_ctr_sensitive": ips_ctr_sensitive,
        }
    ])

    history_df = pd.DataFrame(history_rows)
    action_stats_df = pd.DataFrame(action_stats_rows)

    selected_df = pd.DataFrame(selected_action_rows)
    if not selected_df.empty:
        action_daily_stats_df = selected_df.groupby(["date", "action"], as_index=False).agg(
            impressions_selected=("action", "size")
        )
    else:
        action_daily_stats_df = pd.DataFrame(columns=["date", "action", "impressions_selected"])

    if action_sensitive_stats:
        action_sensitive_df = pd.DataFrame(list(action_sensitive_stats.values()))
        action_sensitive_df["sensitive_ips_ctr"] = action_sensitive_df.apply(
            lambda r: (float(r["cumulative_sensitive_ips_reward"]) / int(r["sensitive_impressions"])) if int(r["sensitive_impressions"]) > 0 else 0.0,
            axis=1,
        )
        action_sensitive_df = action_sensitive_df.sort_values("sensitive_ips_ctr", ascending=False).reset_index(drop=True)
    else:
        action_sensitive_df = pd.DataFrame(columns=["action", "in_train", "sensitive_impressions", "cumulative_sensitive_ips_reward", "sensitive_ips_ctr"])

    return metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df

def run_scenarios_ips(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    policy_factories: dict[str, Callable[[], BasePolicy]],
    scenarios: list[ScenarioConfig],
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    metrics_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    action_stats_parts: list[pd.DataFrame] = []
    action_daily_stats_parts: list[pd.DataFrame] = []
    action_sensitive_parts: list[pd.DataFrame] = []
    trained_models: dict[str, dict[str, BasePolicy]] = {}

    for scenario in scenarios:
        pretrain_df = select_pretrain_data(train_df, scenario.pretrain_source)
        trained_models[scenario.name] = {}

        for algo_name, make_policy in policy_factories.items():
            policy = make_policy()
            if pretrain_df.height > 0:
                policy.fit(pretrain_df)

            initial_seen_actions = {int(r["show"]) for r in pretrain_df.iter_rows(named=True)}

            metrics_df, history_df, action_stats_df, action_daily_stats_df, action_sensitive_df = evaluate_policy_ips(
                policy=policy,
                test_df=test_df,
                online_update=scenario.online_update,
                update_frequency=scenario.update_frequency,
                show_progress=show_progress,
                progress_desc=f"{scenario.name}/{algo_name}",
                initial_seen_actions=initial_seen_actions,
            )
            metrics_df["scenario"] = scenario.name
            metrics_df["algo"] = algo_name
            metrics_parts.append(metrics_df)
            trained_models[scenario.name][algo_name] = policy

            if not history_df.empty:
                history_df["scenario"] = scenario.name
                history_df["algo"] = algo_name
                history_parts.append(history_df)

            if not action_stats_df.empty:
                action_stats_df["scenario"] = scenario.name
                action_stats_df["algo"] = algo_name
                action_stats_parts.append(action_stats_df)

            if not action_daily_stats_df.empty:
                action_daily_stats_df["scenario"] = scenario.name
                action_daily_stats_df["algo"] = algo_name
                action_daily_stats_parts.append(action_daily_stats_df)

            if not action_sensitive_df.empty:
                action_sensitive_df["scenario"] = scenario.name
                action_sensitive_df["algo"] = algo_name
                action_sensitive_parts.append(action_sensitive_df)

    out_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    out_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    out_action_stats = pd.concat(action_stats_parts, ignore_index=True) if action_stats_parts else pd.DataFrame()
    out_action_daily_stats = pd.concat(action_daily_stats_parts, ignore_index=True) if action_daily_stats_parts else pd.DataFrame()
    out_action_sensitive_stats = pd.concat(action_sensitive_parts, ignore_index=True) if action_sensitive_parts else pd.DataFrame()

    return {
        "metrics": out_metrics,
        "history": out_history,
        "action_stats": out_action_stats,
        "action_daily_stats": out_action_daily_stats,
        "action_sensitive_stats": out_action_sensitive_stats,
        "trained_models": trained_models,
    }

def make_simulated_environment(
    proba_predictor: Callable[[dict[str, object], Action], float],
    stochastic: bool = True,
    seed: int = 42,
) -> Callable[[dict[str, object], Action], float]:
    rng = random.Random(seed)

    def env_reward(row: dict[str, object], action: Action) -> float:
        p = max(0.0, min(1.0, float(proba_predictor(row, action))))
        if not stochastic:
            return p
        return 1.0 if rng.random() < p else 0.0

    return env_reward

def default_five_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("case_1_random_pretrain_predict_only", "random", False),
        ScenarioConfig("case_2_random_pretrain_online_update_daily", "random", True, "daily"),
        ScenarioConfig("case_2b_random_pretrain_online_update_step_2p5", "random", True, "step_2p5"),
        ScenarioConfig("case_3_all_pretrain_predict_only", "all", False),
        ScenarioConfig("case_4_all_pretrain_online_update_daily", "all", True, "daily"),
        ScenarioConfig("case_4b_all_pretrain_online_update_step_2p5", "all", True, "step_2p5"),
        ScenarioConfig("case_5_no_pretrain_online_update_daily", "none", True, "daily"),
        ScenarioConfig("case_5b_no_pretrain_online_update_step_2p5", "none", True, "step_2p5"),
    ]

def default_scenario() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("case_2_random_pretrain_online_update_daily", "random", True, "daily"),
    ]

def two_ways_default_scenario() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("case_2_random_pretrain_online_update_daily", "random", True, "daily"),
        ScenarioConfig("case_2b_random_pretrain_online_update_step_2p5", "random", True, "step_2p5"),
    ]

def core_scenarios() -> list[ScenarioConfig]:
    """Backward-compatible alias for previous default (two ways)."""
    return two_ways_default_scenario()
