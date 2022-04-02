import json
import os
import pandas as pd
import numpy as np
import statistics as stat
import socceraction.spadl as spadl
import utils

from tqdm import tqdm
from typing import List, Tuple, Dict, Optional


def compare(setback: pd.Series, a_actions: pd.DataFrame, d_actions: pd.DataFrame, player_game: pd.Series,
            game_duration: int, consider_goals=True) -> pd.DataFrame:
    # Get the action that is a setback
    a_setback_action = a_actions[(a_actions['period_id'] == setback.period_id) &
                                 (a_actions['time_seconds'] == setback.time_seconds)].iloc[0]

    # Only consider actions where player of setback is on field
    a_actions = get_player_on_field_actions(a_actions, player_game, game_duration, a_setback_action)
    d_actions = get_player_on_field_actions(d_actions, player_game, game_duration, a_setback_action)

    # Get atomic player actions before and after the setback
    a_actions_before, a_actions_after = get_before_after(a_setback_action, a_actions)

    # Get vaep-rating before and after the setback
    vaep_before = vaep_aggregates(a_actions_before, a_actions.iloc[0], a_setback_action, consider_goals)
    vaep_after = vaep_aggregates(a_actions_after, a_setback_action, a_actions.iloc[-1], consider_goals, False)

    # Get time on the ball before and after the setback
    time_on_ball_before = time_on_ball_aggregates(a_actions_before, a_actions.iloc[0], a_setback_action)
    time_on_ball_after = time_on_ball_aggregates(a_actions_after, a_setback_action, a_actions.iloc[-1])

    # Get non-atomic player actions before and after the setback
    d_actions_before, d_actions_after = get_before_after(a_setback_action, d_actions)

    # Get the success rate of actions before and after the setback
    success_rate_before = success_rate(d_actions_before)
    success_rate_after = success_rate(d_actions_after)

    # Get #minutes played before and after the setback
    minutes_played_before = (a_setback_action.total_seconds - a_actions.iloc[0]['total_seconds']) / 60
    minutes_played_after = player_game.minutes_played - minutes_played_before

    # Get the risk the player takes in his actions before and after the setback
    risk_before = get_average_risk(d_actions_before)
    risk_after = get_average_risk(d_actions_after)

    # Put data in dataframe
    comparison = pd.DataFrame(
        data={"vaep_per_action": data_to_list(vaep_before[0], vaep_after[0]),
              "vaep_per_minute": data_to_list(vaep_before[1], vaep_after[1]),
              "avg_time_on_ball": data_to_list(time_on_ball_before[0], time_on_ball_after[0]),
              "time_on_ball_per_minute": data_to_list(time_on_ball_before[1], time_on_ball_after[1]),
              "success_rate": data_to_list(success_rate_before, success_rate_after),
              "avg_risk": data_to_list(risk_before, risk_after),
              "minutes_played": data_to_list(minutes_played_before, minutes_played_after)},
        index=["before_setback", "after_setback", "difference", "relative difference"]
    )
    return round(comparison, 4)


def get_player_on_field_actions(actions: pd.DataFrame, player_game: pd.Series, game_duration: int,
                                setback: pd.Series) -> pd.DataFrame:
    if player_game.minutes_played == game_duration:
        return actions

    maximum = max(player_game.minutes_played, setback['total_seconds'] / 60)
    if player_game.is_starter:
        return actions[(actions['total_seconds'] / 60) <= maximum]
    else:
        return actions[(actions['total_seconds'] / 60) >= (game_duration - player_game.minutes_played) - 1]


def data_to_list(data_before: float, data_after: float) -> List[float]:
    diff = data_after - data_before
    rel_diff = np.float64(diff) / data_before
    return [data_before, data_after, diff, rel_diff]


def get_before_after(setback: pd.Series, actions: pd.DataFrame) -> pd.DataFrame:
    player_actions = actions[actions['player_id'] == setback.player_id]
    actions_before = player_actions[player_actions['total_seconds'] < setback['total_seconds']]
    actions_after = player_actions[player_actions['total_seconds'] > setback['total_seconds']]
    return actions_before, actions_after


def vaep_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series, consider_goals=False,
                    before=True) -> Tuple[float]:
    if player_actions.empty:
        return 0, 0
    game_rating_progression_h5 = "results/gr_progression.h5"
    with pd.HDFStore(game_rating_progression_h5) as store:
        per_action = store["per_action"]
        per_minute = store["per_minute"]
        mean_action = per_action['Mean'].mean()
        mean_minute = per_minute['Mean'].mean()
        action_scale = per_action['Mean'] / mean_action
        minute_scale = per_minute['Mean'] / mean_minute

    # if not consider_goals:
    #     goal_actions = ['goal', 'owngoal', 'bad_touch']
    #     vaep_action = player_actions[~player_actions['type_name'].isin(goal_actions)][
    #         'vaep_value'].mean() if not player_actions.empty else 0
    #     vaep_minute = [~player_actions['type_name'].isin(goal_actions)]['vaep_value'].sum() / (
    #             (last.total_seconds - first.total_seconds) / 60)
    # else:

    chunck_size = 10
    player_actions['time_chunck'] = player_actions.apply(
        lambda x: (x['total_seconds'] // (chunck_size * 60)) if (x['total_seconds'] < 5400) else -1, axis=1)
    group_by_time_chunks = player_actions.groupby('time_chunck')

    vaep_action = []
    vaep_minute = []
    for index, time_chunk in group_by_time_chunks:
        if not time_chunk.empty:
            vaep_action.append(time_chunk['vaep_value'].mean() / action_scale.iloc[int(index)])

        if int(index) == -1:
            time_comp = first.total_seconds if (not before) else 90
            vaep_minute.append(
                (time_chunk['vaep_value'].sum() / ((last.total_seconds / 60) - time_comp)) / minute_scale.iloc[
                    int(index)])
        else:
            vaep_minute.append((time_chunk['vaep_value'].sum() / chunck_size) / minute_scale.iloc[int(index)])

    vaep_action = stat.mean(vaep_action) if vaep_action else 0
    vaep_minute = stat.mean(vaep_minute)

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


def get_average_risk(actions: pd.DataFrame) -> float:
    predictions_h5 = "expected_passing/predictions.h5"
    with pd.HDFStore(predictions_h5) as predictionstore:
        predictions = predictionstore["predictions"]

    actions_on_id = actions.dropna().set_index('original_event_id')
    actions_on_id = actions_on_id[actions_on_id['type_name'] != 'interception']
    actions_with_risk = actions_on_id.join(predictions).dropna()
    actions_with_risk['risk'] = 1 - actions_with_risk['exp_accuracy']

    risk = actions_with_risk['risk'].mean() if not actions_with_risk.empty else 0

    return risk


def get_responses(player_setbacks: pd.DataFrame, player_games: pd.DataFrame, a_actions: pd.DataFrame,
                  d_actions: pd.DataFrame) -> Dict[Tuple, List[pd.DataFrame]]:
    responses: Dict[int, pd.DataFrame] = {}
    for index, player_setback in tqdm(list(player_setbacks.iterrows()), desc="Comparing performance: "):
        players_in_game = player_games[player_games['game_id'] == player_setback.game_id].set_index('player_id')
        player_game = players_in_game.loc[player_setback.player_id]
        game_duration = players_in_game['minutes_played'].max()
        atomic_game_actions = a_actions[a_actions['game_id'] == player_setback.game_id]
        game_actions = d_actions[d_actions['game_id'] == player_setback.game_id]

        response = compare(player_setback, atomic_game_actions, game_actions, player_game, game_duration, True)
        responses[index] = response

        # print(player_setback)
        # print()
        # print(compare(player_setback, atomic_game_actions, game_actions, player_game, game_duration))
        # print()
        # print()
        # break

    return responses


def compare_ingame_setbacks():
    atomic_h5 = os.path.join("atomic_data", "spadl.h5")
    default_h5 = os.path.join("default_data", "spadl.h5")
    setbacks_h5 = os.path.join("default_data", "setbacks.h5")
    predictions_h5 = os.path.join("atomic_data", "predictions.h5")

    with pd.HDFStore(atomic_h5) as atomicstore:
        games = atomicstore["games"]
        player_games = atomicstore["player_games"]

    # Player_id's of players with more than 900 minutes played
    player_ids = []
    grouped_by_player_id = player_games.groupby('player_id')
    for player_id, p_games in grouped_by_player_id:
        if p_games['minutes_played'].sum() > 900:
            player_ids.append(player_id)

    games = games[games['competition_id'] != 426]
    # games = games[games['competition_id'] == 524]

    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_as_player_setbacks = setbackstore['team_as_player_setbacks']

    player_setbacks = pd.concat([player_setbacks, team_as_player_setbacks])
    player_setbacks = player_setbacks[player_setbacks['player_id'].isin(player_ids)]
    player_setbacks = player_setbacks[player_setbacks['game_id'].isin(games['game_id'].tolist())].reset_index(drop=True)
    player_setbacks = player_setbacks[player_setbacks['setback_type'] != "goal conceded"]

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

    responses = get_responses(player_setbacks, player_games, a_actions, d_actions)

    compstore_h5 = "results/comparisons.h5"
    with pd.HDFStore(compstore_h5) as compstore:
        for index in responses.keys():
            compstore["ingame_comp_{}".format(index)] = responses[index]


def get_average_response():
    atomic_h5 = os.path.join("atomic_data", "spadl.h5")
    with pd.HDFStore(atomic_h5) as atomicstore:
        games = atomicstore["games"]
        player_games = atomicstore["player_games"]

    games = games[games['competition_id'] != 426]

    # Player_id's of players with more than 900 minutes played
    player_ids = []
    grouped_by_player_id = player_games.groupby('player_id')
    for player_id, p_games in grouped_by_player_id:
        if p_games['minutes_played'].sum() > 900:
            player_ids.append(player_id)

    setbacks_h5 = "default_data/setbacks.h5"
    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_as_player_setbacks = setbackstore['team_as_player_setbacks']

    player_setbacks = pd.concat([player_setbacks, team_as_player_setbacks])
    player_setbacks = player_setbacks[player_setbacks['player_id'].isin(player_ids)]
    player_setbacks = player_setbacks[player_setbacks['game_id'].isin(games['game_id'].tolist())].reset_index(drop=True)
    player_setbacks = player_setbacks[player_setbacks['setback_type'] != "goal conceded"]

    responses: Dict[Tuple, List[pd.DataFrame]] = {}
    compstore_h5 = "results/comparisons.h5"
    with pd.HDFStore(compstore_h5) as compstore:
        for index, setback in player_setbacks.iterrows():
            setback_response = compstore["ingame_comp_{}".format(index)]
            key = (setback.player_id, setback.setback_type)
            # Only consider setbacks with more than 10 minutes played before and after
            if (setback_response.at['before_setback', 'minutes_played'] >= 10) and (
                    setback_response.at['after_setback', 'minutes_played'] >= 10):
                if key in responses:
                    responses[key].append(setback_response)
                else:
                    responses[key] = [setback_response]

    avg_response_h5 = "results/avg_response.h5"
    with pd.HDFStore(avg_response_h5) as store:
        for key in responses.keys():
            number = len(responses[key])
            response = pd.concat(responses[key])
            grouped_by_index = response.groupby(response.index, sort=False)
            mean = grouped_by_index.mean()
            mean.loc['relative difference'] = (mean.loc['after_setback'] - mean.loc['before_setback']) / mean.loc[
                'before_setback']
            new_key = "{}: {} ({})".format(key[0], key[1], number)
            store[new_key] = mean


def compare_for_setback(setback_type: str, player_id: Optional = None):
    avg_responses = {}
    avg_response_h5 = "results/avg_response.h5"
    with pd.HDFStore(avg_response_h5) as store:
        for key in store.keys():
            number = int(key.split("(")[1][:-1])
            if setback_type in key and number >= 3:
                if player_id is not None:
                    if str(player_id) in key:
                        avg_responses[key] = store[key]
                else:
                    avg_responses[key] = store[key]

    print(len(avg_responses))
    # avg_responses = sorted(avg_responses.items(), key=lambda x: x[1].at['difference', 'vaep_per_minute'], reverse=False)
    avg_responses = sorted(avg_responses.items(), key=lambda x: abs(x[1].at['difference', 'vaep_per_minute']), reverse=False)


    for response in avg_responses[:10]:
        print(response[0])
        print()
        print(response[1])
        print()
        print()