import os
import pandas as pd
import statistics as stat
import socceraction.spadl as spadl
import utils

from tqdm import tqdm
from typing import List, Tuple


def compare(setback: pd.Series, a_actions: pd.DataFrame, actions: pd.DataFrame, player_game: pd.Series) -> pd.DataFrame:  # TODO: use player_games to determine minutes played before after and fix time on ball per minute
    a_actions_before, a_actions_after = get_before_after(setback, a_actions)
    a_setback_action = a_actions[(a_actions['period_id'] == setback.period_id) &
                                 (a_actions['time_seconds'] == setback.time_seconds)].iloc[0]

    vaep_before = vaep_aggregates(a_actions_before, a_actions.iloc[0], a_setback_action)
    vaep_after = vaep_aggregates(a_actions_after, a_setback_action, a_actions.iloc[-1])

    time_on_ball_before = time_on_ball_aggregates(a_actions_before, a_actions.iloc[0], a_setback_action)
    time_on_ball_after = time_on_ball_aggregates(a_actions_after, a_setback_action, a_actions.iloc[-1])

    actions_before, actions_after = get_before_after(setback, actions)

    success_rate_before = success_rate(actions_before)
    success_rate_after = success_rate(actions_after)

    comparison = pd.DataFrame(
        data={"vaep_per_action": data_to_list(vaep_before[0], vaep_after[0]),
              "vaep_per_minute": data_to_list(vaep_before[1], vaep_after[1]),
              "avg_time_on_ball": data_to_list(time_on_ball_before[0], time_on_ball_after[0]),
              "time_on_ball_per_minute": data_to_list(time_on_ball_before[1], time_on_ball_after[1]),
              "success_rate": data_to_list(success_rate_before, success_rate_after)},
        index=["before_setback", "after_setback", "difference", "relative difference"]
    )
    return comparison


def data_to_list(data_before: float, data_after: float) -> List[float]:
    diff = data_after - data_before
    rel_diff = diff / data_before
    return [data_before, data_after, diff, rel_diff]


def get_before_after(setback: pd.Series, actions: pd.DataFrame) -> pd.DataFrame:
    actions = utils.add_total_seconds(actions)
    player_actions = actions[actions['player_id'] == setback.player_id]
    before_setback_selector = ((player_actions['period_id'] < setback.period_id) |
                               ((player_actions['period_id'] == setback.period_id) &
                                (player_actions['time_seconds'] < setback.period_id)))
    actions_before = player_actions[before_setback_selector]
    actions_after = player_actions[~before_setback_selector].iloc[1:]  # This can be empty
    return actions_before, actions_after


def vaep_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series) -> Tuple[float]:
    vaep_action = player_actions['vaep_value'].mean()
    vaep_minute = player_actions['vaep_value'].sum() / ((last.total_seconds - first.total_seconds) / 60)

    return vaep_action, vaep_minute


def time_on_ball_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series) -> Tuple[float]:
    grouped_by_cons_actions = player_actions.groupby(
        [(player_actions['action_id'] != player_actions.shift(1)['action_id'] + 1).cumsum(), 'period_id'])

    time_on_ball = []
    for _, cons_actions in grouped_by_cons_actions:
        time_on_ball.append(cons_actions.iloc[-1]['time_seconds'] - cons_actions.iloc[0]['time_seconds'])

    avg_time_on_ball = stat.mean(time_on_ball)
    time_on_ball_per_minute = sum(time_on_ball) / ((last['total_seconds'] - first['total_seconds']) / 60)

    return avg_time_on_ball, time_on_ball_per_minute


def success_rate(player_actions: pd.DataFrame) -> float:  # Requires non-atomic actions!!
    success = player_actions[player_actions['result_name'] == 'success'].shape[0]
    fail = player_actions[player_actions['result_name'] != 'success'].shape[0]

    return success / (success + fail)


def compare_ingame_setbacks():
    atomic_h5 = os.path.join("atomic_data", "spadl.h5")
    default_h5 = os.path.join("default_data", "spadl.h5")
    setbacks_h5 = os.path.join("default_data", "setbacks.h5")
    predictions_h5 = os.path.join("atomic_data", "predictions.h5")

    with pd.HDFStore(atomic_h5) as atomicstore:
        games = atomicstore["games"]
        player_games = atomicstore["player_games"]

    games = games[games['competition_id'] == 524]

    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_setbacks = setbackstore["teams_setbacks"]

    a_actions = []
    d_actions = []
    for game in tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(atomic_h5, f"actions/game_{game.game_id}")
        values = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
        actions = pd.concat([actions, values], axis=1)
        a_actions.append(actions[actions['period_id'] != 5])

        actions = pd.read_hdf(default_h5, f"actions/game_{game.game_id}")
        actions = spadl.add_names(actions)
        d_actions.append(actions[actions['period_id'] != 5])

    a_actions = pd.concat(a_actions).reset_index(drop=True)
    d_actions = pd.concat(d_actions).reset_index(drop=True)

    for player_setback in tqdm(list(player_setbacks.itertuples()), desc="Comparing performance: "):
        player_game = player_games[(player_games['game_id'] == player_setback.game_id) & (
                    player_games['player_id'] == player_setback.player_id)].iloc[0]
        print(compare(player_setback, a_actions[a_actions['game_id'] == player_setback.game_id],
                      d_actions[d_actions['game_id'] == player_setback.game_id], player_game))
        break
