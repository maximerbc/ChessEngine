"""Measured workload arithmetic. Missing observations remain unknown, never zero."""
import math


def forecast(measurement, rate, storage_gb, storage_rate, config):
    required = ['tournament_games_per_second', 'label_positions_per_second',
                'supervised_seconds_per_epoch', 'cpu_evaluation_games_per_second',
                'compilation_seconds', 'measured_on_nvidia', 'hardware', 'workload_sha256']
    if any(k not in measurement for k in required):
        raise ValueError('Complete-game, labeling, training and CPU evaluation benchmarks required')
    for k in required[:5]:
        if not isinstance(measurement[k], (int,float)) or not math.isfinite(measurement[k]) or measurement[k] <= 0:
            raise ValueError('Measured throughput and overhead must be positive and finite')
    if rate < 0 or storage_gb < 0 or storage_rate < 0:
        raise ValueError('Invalid quoted costs')
    training = config['population'] * config['generations'] * len(config['seeds'])
    pilot_games = config['population'] * config['pilot_generations'] * 48
    # Main checkpoint development: initializer plus every mean, both references.
    development = len(config['development_checkpoints']) * 40 * 2 * 2 * len(config['seeds'])
    qualification_final = 200 * 2 * 2 * 2 * len(config['seeds']) * 2  # two new models
    pilot_checks = 40 * 2 * 2
    supervised_development = 40 * 2 * 2 * len(config['seeds'])
    eval_games = development + qualification_final + pilot_checks + supervised_development
    label_seconds = 1000000 / measurement['label_positions_per_second']
    fit_seconds = 40 * len(config['seeds']) * measurement['supervised_seconds_per_epoch']
    eval_seconds = eval_games / measurement['cpu_evaluation_games_per_second']
    rows = []
    for games in [24, 48, 96]:
        main_games = training * games
        main_seconds = main_games / measurement['tournament_games_per_second']
        pilot_seconds = pilot_games / measurement['tournament_games_per_second']
        seconds = sum([main_seconds, pilot_seconds, label_seconds, fit_seconds, eval_seconds,
                       measurement['compilation_seconds']])
        hours = seconds / 3600
        # Sequential conservative bill; separate CPU/GPU quotes may lower this after provider selection.
        storage = storage_gb * storage_rate * (hours / 24 / 30)
        subtotal = hours * rate + storage
        rows.append(dict(games_per_candidate=games, main_games=main_games, pilot_games=pilot_games,
            evaluation_games=eval_games, hours=hours, days=hours / 24,
            phase_hours=dict(main=main_seconds / 3600, pilot=pilot_seconds / 3600,
                labeling=label_seconds / 3600, supervised=fit_seconds / 3600, evaluation=eval_seconds / 3600),
            quoted_compute_cost=subtotal - storage, storage_cost=storage,
            cost_with_20_percent_contingency=subtotal * 1.2))
    return dict(status='measured-nvidia' if measurement['measured_on_nvidia'] else 'CPU-only diagnostic; not a GPU quote',
        hardware=measurement['hardware'], currency='quote currency; conversion and taxes must be added',
        training_budget='unset pending user decision', planned_default_main_games=368640,
        scenarios=rows, measurement=measurement,
        limitation='Provider availability, interruption overhead, transfers and taxes require a live quote. No resources provisioned.')
