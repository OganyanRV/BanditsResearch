# Bandit/CatBoost experiment plan for logged ad data

Ниже — ровно та схема, которую вы описали, в терминах кода.

## 1) Разбиение train/test
Используем только time-based split:
- train: более ранние даты
- test: более поздние даты

Функция: `split_train_test_by_date(events, test_ratio=0.2)`.

## 2) Что сравниваем
- `epsilon_greedy`
- `ucb`
- `thompson_sampling`
- `catboost_policy` (только инференс на тесте, без онлайн-дообучения)
- `contextual_bandit` — placeholder (пока не реализован)

## 3) 5 сценариев оценки
Реализованы как `run_five_scenarios(...)`:
1. `case_1_random_pretrain_predict_only`
2. `case_2_random_pretrain_online_update`
3. `case_3_all_pretrain_predict_only`
4. `case_4_all_pretrain_online_update`
5. `case_5_no_pretrain_online_update`

Важно: CatBoost policy не обновляется онлайн в сценариях 2/4/5.

## 4) Как считается метрика
Базово применяется replay-оценка на логах:
- если выбранное действие совпало с `show`, используем логированный `reward`
- иначе награда не наблюдаема и этот шаг пропускается

Возвращаются метрики:
- `ctr`
- `total_reward`
- `impressions_used`
- `replay_match_rate`

## 5) "Если бы знали все метки"
Сделано через симулированную среду:
1. обучаете модель среды (например CatBoost) как `P(click|x, action)`
2. передаёте её в `make_simulated_environment(...)`
3. запускаете `run_five_scenarios(...)` с `env_reward`

Тогда для любого выбранного действия есть награда (stochastic Bernoulli или expected reward).

## 6) Минимальные поля события
`BanditEvent`:
- `policy`, `reward`, `puid`, `features`, `show`, `candidates`, `date`
- опционально `propensity`, `rewards_by_action`

`propensity` пока не используется в replay-части этого модуля (можно добавить IPS/SNIPS слоем сверху).
