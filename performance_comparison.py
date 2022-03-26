import os
import pandas as pd
import numpy as np
import statistics as stat
import socceraction.spadl as spadl
import utils

from tqdm import tqdm
from typing import List, Tuple


def compare(setback: pd.Series, a_actions: pd.DataFrame, d_actions: pd.DataFrame, player_game: pd.Series,
            game_duration: int) -> pd.DataFrame:
    a_actions = get_player_on_field_actions(a_actions, player_game, game_duration)
    d_actions = get_player_on_field_actions(d_actions, player_game, game_duration)

    a_setback_action = a_actions[(a_actions['period_id'] == setback.period_id) &
                                 (a_actions['time_seconds'] == setback.time_seconds)].iloc[0]
    a_actions_before, a_actions_after = get_before_after(a_setback_action, a_actions)

    vaep_before = vaep_aggregates(a_actions_before, a_actions.iloc[0], a_setback_action)
    vaep_after = vaep_aggregates(a_actions_after, a_setback_action, a_actions.iloc[-1])

    time_on_ball_before = time_on_ball_aggregates(a_actions_before, a_actions.iloc[0], a_setback_action)
    time_on_ball_after = time_on_ball_aggregates(a_actions_after, a_setback_action, a_actions.iloc[-1])

    d_actions_before, d_actions_after = get_before_after(a_setback_action, d_actions)

    success_rate_before = success_rate(d_actions_before)
    success_rate_after = success_rate(d_actions_after)

    minutes_played_before = (a_setback_action.total_seconds - a_actions.iloc[0]['total_seconds']) / 60
    minutes_played_after = player_game.minutes_played - minutes_played_before

    comparison = pd.DataFrame(
        data={"vaep_per_action": data_to_list(vaep_before[0], vaep_after[0]),
              "vaep_per_minute": data_to_list(vaep_before[1], vaep_after[1]),
              "avg_time_on_ball": data_to_list(time_on_ball_before[0], time_on_ball_after[0]),
              "time_on_ball_per_minute": data_to_list(time_on_ball_before[1], time_on_ball_after[1]),
              "success_rate": data_to_list(success_rate_before, success_rate_after),
              "minutes_played": data_to_list(minutes_played_before, minutes_played_after)},
        index=["before_setback", "after_setback", "difference", "relative difference"]
    )
    return comparison


def get_player_on_field_actions(actions: pd.DataFrame, player_game: pd.Series, game_duration: int) -> pd.DataFrame:
    if player_game.minutes_played == game_duration:
        return actions
    if player_game.is_starter:
        return actions[(actions['total_seconds'] / 60) < player_game.minutes_played]
    else:
        return actions[(actions['total_seconds'] / 60) > (game_duration - player_game.minutes_played)]


def data_to_list(data_before: float, data_after: float) -> List[float]:
    diff = data_after - data_before
    rel_diff = np.float64(diff) / data_before
    return [data_before, data_after, diff, rel_diff]


def get_before_after(setback: pd.Series, actions: pd.DataFrame) -> pd.DataFrame:
    player_actions = actions[actions['player_id'] == setback.player_id]
    actions_before = player_actions[player_actions['total_seconds'] < setback['total_seconds']]
    actions_after = player_actions[player_actions['total_seconds'] > setback['total_seconds']]
    return actions_before, actions_after


def vaep_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series) -> Tuple[float]:
    vaep_action = player_actions['vaep_value'].mean() if not player_actions.empty else 0
    vaep_minute = player_actions['vaep_value'].sum() / ((last.total_seconds - first.total_seconds) / 60)

    return vaep_action, vaep_minute


def time_on_ball_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series) -> Tuple[float]:
    grouped_by_cons_actions = player_actions.groupby(
        [(player_actions['action_id'] != player_actions.shift(1)['action_id'] + 1).cumsum(), 'period_id'])

    time_on_ball = []
    for _, cons_actions in grouped_by_cons_actions:
        time_diff = cons_actions.iloc[-1]['time_seconds'] - cons_actions.iloc[0]['time_seconds']
        time_on_ball.append(time_diff if time_diff != 0 else 1)

    avg_time_on_ball = stat.mean(time_on_ball) if time_on_ball else 0
    time_on_ball_per_minute = sum(time_on_ball) / ((last['total_seconds'] - first['total_seconds']) / 60)

    return avg_time_on_ball, time_on_ball_per_minute


def success_rate(player_actions: pd.DataFrame) -> float:  # Requires non-atomic actions!!
    success = player_actions[player_actions['result_name'] == 'success'].shape[0]
    fail = player_actions[player_actions['result_name'] != 'success'].shape[0]
    if success == 0 and fail == 0:
        return 0
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
    for game_id in tqdm(list(games.game_id), desc="Rating actions"):
        actions = pd.read_hdf(atomic_h5, f"actions/game_{game_id}")
        actions = actions[actions['period_id'] != 5]
        values = pd.read_hdf(predictions_h5, f"game_{game_id}")
        a_actions.append(pd.concat([actions, values], axis=1))

        actions = pd.read_hdf(default_h5, f"actions/game_{game_id}")
        actions[actions['period_id'] != 5]
        d_actions.append(spadl.add_names(actions))

    a_actions = pd.concat(a_actions).reset_index(drop=True)
    a_actions = utils.add_total_seconds(a_actions, games)
    d_actions = pd.concat(d_actions).reset_index(drop=True)
    d_actions = utils.add_total_seconds(d_actions, games)

    for player_setback in tqdm(list(player_setbacks.itertuples()), desc="Comparing performance: "):
        players_in_game = player_games[player_games['game_id'] == player_setback.game_id].set_index('player_id')
        player_game = players_in_game.loc[player_setback.player_id]
        game_duration = players_in_game['minutes_played'].max()
        atomic_game_actions = a_actions[a_actions['game_id'] == player_setback.game_id]
        game_actions = d_actions[d_actions['game_id'] == player_setback.game_id]
        print(player_setback)
        print()
        print(compare(player_setback, atomic_game_actions, game_actions, player_game, game_duration))
        print()
        print()
        # break
