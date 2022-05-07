import math
import os
import warnings

import pandas as pd
import numpy as np
import statistics as stat
import socceraction.spadl as spadl
import socceraction.atomic.spadl as aspadl
import utils

from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)


def compare(setback: pd.Series, pof_a_actions: pd.DataFrame, pof_d_actions: pd.DataFrame, player_game: pd.Series,
            game_duration: int, consider_goals=True) -> pd.DataFrame:
    """
    Compare the performance of the player suffering the setback before and after the setback.

    :param setback: a setback
    :param pof_a_actions: atomic actions
    :param pof_d_actions: non atomic actions
    :param player_game: the game in which the setback occurs
    :param game_duration: the duration of the game in which the setback occurs
    :param consider_goals: whether or not to consider goals in the performance comparison
    :return: a dataframe comparing performance before and after the setback
    """
    # Get the action that is a setback
    # a_setback_action = a_actions[(a_actions['period_id'] == setback.period_id) &
    #                              (a_actions['time_seconds'] == setback.time_seconds)].iloc[0]

    # Only consider actions where player of setback is on field
    pof_a_actions = utils.get_player_on_field_actions(pof_a_actions, player_game, game_duration, setback)
    pof_d_actions = utils.get_player_on_field_actions(pof_d_actions, player_game, game_duration, setback)

    # Get atomic player actions before and after the setback
    a_actions_before, a_actions_after = get_before_after(setback, pof_a_actions)

    # Get vaep-rating before and after the setback
    vaep_before = get_vaep_aggregates(a_actions_before, pof_a_actions.iloc[0], setback, consider_goals)
    vaep_after = get_vaep_aggregates(a_actions_after, setback, pof_a_actions.iloc[-1], consider_goals, False)

    # Get time on the ball before and after the setback
    time_on_ball_before = get_time_on_ball_aggregates(a_actions_before, pof_a_actions.iloc[0], setback)
    time_on_ball_after = get_time_on_ball_aggregates(a_actions_after, setback, pof_a_actions.iloc[-1])

    # Get non-atomic player actions before and after the setback
    d_actions_before, d_actions_after = get_before_after(setback, pof_d_actions)

    # Get the success rate of actions before and after the setback
    success_rate_before = get_success_rate(d_actions_before)
    success_rate_after = get_success_rate(d_actions_after)

    # Get #minutes played before and after the setback
    minutes_played_before = (setback.total_seconds - pof_a_actions.iloc[0]['total_seconds']) / 60
    minutes_played_after = player_game.minutes_played - minutes_played_before

    # Get the risk the player takes in his actions before and after the setback
    risk_before = get_average_risk(d_actions_before)
    risk_after = get_average_risk(d_actions_after)

    # Put data in dataframe
    comparison = pd.DataFrame(
        data={'vaep_per_action': data_to_list(vaep_before[0], vaep_after[0]),
              'vaep_per_minute': data_to_list(vaep_before[1], vaep_after[1]),
              'avg_time_on_ball': data_to_list(time_on_ball_before[0], time_on_ball_after[0]),
              'time_on_ball_per_minute': data_to_list(time_on_ball_before[1], time_on_ball_after[1]),
              'success_rate': data_to_list(success_rate_before, success_rate_after),
              'avg_risk': data_to_list(risk_before, risk_after),
              'minutes_played': data_to_list(minutes_played_before, minutes_played_after)},
        index=['before_setback', 'after_setback', 'difference', 'relative difference']
    )

    summary = round(comparison, 4)

    return summary


def data_to_list(data_before: float, data_after: float) -> List[float]:
    """
    Converts the given data to a list format.

    :param data_before: the data before the setback
    :param data_after: the data after the setback
    :return: a list representation of the performance comparison
    """
    diff = data_after - data_before
    rel_diff = np.float64(diff) / data_before

    return [data_before, data_after, diff, rel_diff]


def get_before_after(setback: pd.Series, actions: pd.DataFrame) -> pd.DataFrame:
    """
    Get the actions executed before and after the setback.

    :param setback: a setback
    :param actions: the actions in the game
    :return: the actions executed before and after the setback
    """
    player_actions = actions[actions['player_id'] == setback.player_id]
    actions_before = player_actions[player_actions['total_seconds'] < setback.total_seconds]
    actions_after = player_actions[player_actions['total_seconds'] > setback.total_seconds]

    return actions_before, actions_after


def get_vaep_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series, consider_goals=True,
                        before=True) -> Tuple[float]:
    if player_actions.empty:
        return 0, 0

    game_rating_progression_h5 = 'results/game_rating_progression.h5'
    with pd.HDFStore(game_rating_progression_h5) as store:
        per_action = store['per_action_with_goal_diff']
        per_minute = store['per_minute_with_goal_diff']

    mean_action = per_action.stack().mean()
    mean_minute = per_minute.stack().mean()

    action_scale = per_action / mean_action
    minute_scale = per_minute / mean_minute

    # if not consider_goals:
    #     goal_actions = ['goal', 'owngoal', 'bad_touch']
    #     vaep_action = player_actions[~player_actions['type_name'].isin(goal_actions)][
    #         'vaep_value'].mean() if not player_actions.empty else 0
    #     vaep_minute = [~player_actions['type_name'].isin(goal_actions)]['vaep_value'].sum() / (
    #             (last.total_seconds - first.total_seconds) / 60)
    # else:

    chunk_size = 10
    player_actions['time_chunck'] = player_actions.apply(
        lambda x: (x['total_seconds'] // (chunk_size * 60)) if (x['total_seconds'] < 5400) else -1, axis=1)
    group_by_time_score = player_actions.groupby(['score_diff', 'time_chunck'])

    vaep_action = []
    vaep_minute = []
    for index, time_score in group_by_time_score:
        score_diff = index[0]
        label = utils.get_label(index[1], chunk_size)

        if not time_score.empty:
            vaep_action.append(time_score['vaep_value'].mean() / action_scale.loc[score_diff, label])

        if index == -1:
            time_comp = first.total_seconds if (not before) else 90
            vaep_minute.append(
                (time_score['vaep_value'].sum() / (math.ceil(last.total_seconds / 60) - time_comp)) / minute_scale.loc[
                    score_diff, label])
        else:
            vaep_minute.append((time_score['vaep_value'].sum() / chunk_size) / minute_scale.loc[score_diff, label])

    vaep_action = stat.mean(vaep_action) if vaep_action else 0
    vaep_minute = stat.mean(vaep_minute)

    return vaep_action, vaep_minute


def get_time_on_ball_aggregates(player_actions: pd.DataFrame, first: pd.Series, last: pd.Series) -> Tuple[float]:
    grouped_by_cons_actions = player_actions.groupby(
        [(player_actions['action_id'] != player_actions.shift(1)['action_id'] + 1).cumsum(), 'period_id'])

    time_on_ball = []
    for _, cons_actions in grouped_by_cons_actions:
        time_diff = cons_actions.iloc[-1]['time_seconds'] - cons_actions.iloc[0]['time_seconds']
        time_on_ball.append(time_diff if time_diff != 0 else 1)

    avg_time_on_ball = stat.mean(time_on_ball) if time_on_ball else 0
    time_on_ball_per_minute = sum(time_on_ball) / ((last.total_seconds - first.total_seconds) / 60)

    return avg_time_on_ball, time_on_ball_per_minute


def get_success_rate(player_actions: pd.DataFrame) -> float:  # Requires non-atomic actions!!
    success = player_actions[player_actions['result_name'] == 'success'].shape[0]
    fail = player_actions[player_actions['result_name'] != 'success'].shape[0]
    if success == 0 and fail == 0:
        return 0

    success_rate = success / (success + fail)

    return success_rate


# def get_expected_success_rate(player_actions: pd.DataFrame) -> float:



def get_average_risk(actions: pd.DataFrame) -> float:
    predictions_h5 = 'xP_data/predictions.h5'
    with pd.HDFStore(predictions_h5) as predictionstore:
        predictions = predictionstore['predictions']

    actions_on_id = actions.dropna().set_index('original_event_id')
    actions_on_id = actions_on_id[actions_on_id['type_name'] != 'interception']
    actions_with_risk = actions_on_id.join(predictions).dropna()
    actions_with_risk['risk'] = 1 - actions_with_risk['exp_accuracy']

    risk = actions_with_risk['risk'].mean() if not actions_with_risk.empty else 0

    return risk


def get_responses(player_setbacks: pd.DataFrame, player_games: pd.DataFrame, a_actions: pd.DataFrame,
                  d_actions: pd.DataFrame) -> Dict[Tuple, List[pd.DataFrame]]:
    responses: Dict[int, pd.DataFrame] = {}
    for player_setback in tqdm(list(player_setbacks.itertuples()), desc="Comparing performance: "):
        try:
            players_in_game = player_games[player_games['game_id'] == player_setback.game_id].set_index('player_id')
            player_game = players_in_game.loc[player_setback.player_id]
            game_duration = players_in_game['minutes_played'].max()
            atomic_game_actions = a_actions[a_actions['game_id'] == player_setback.game_id]
            game_actions = d_actions[d_actions['game_id'] == player_setback.game_id]
            response = compare(player_setback, atomic_game_actions, game_actions, player_game, game_duration, True)
            responses[player_setback.setback_id] = response
        except:
            continue

        # print()
        # print(response)
        # print()

    return responses


def get_performance_per_minute_in_games(game_ids: List[int], team: str, actions: pd.DataFrame, player_games: pd.DataFrame):
    team_actions = actions[actions['team_name_short'] == team]
    total_vaep = 0
    total_minutes = 0
    for game_id in game_ids:
        actions_in_game = team_actions[team_actions['game_id'] == game_id]
        total_vaep += actions_in_game['vaep_value'].sum()
        total_minutes += player_games[player_games['game_id'] == game_id].max()

    vaep_per_minute = total_vaep / total_minutes

    return vaep_per_minute


def get_next_game(setback: pd.Series, games: pd.DataFrame):
    games_in_competition = games[games['competition_name'] == setback.competition]
    team_games = games_in_competition[(games_in_competition['home_team_name_short'] == setback.team) |
                                      (games_in_competition['away_team_name_short'] == setback.team)]
    games_after_setback = team_games[team_games['game_date'] > setback.game_date_last_loss]
    first_game_after_setback = games_after_setback.iloc[0]

    return first_game_after_setback


def compare_response_by_losing_chance():
    atomic_h5 = 'atomic_data/spadl.h5'
    setbacks_h5 = 'setback_data/setbacks.h5'
    a_predictions_h5 = 'atomic_data/predictions.h5'

    with pd.HDFStore(atomic_h5) as store:
        games = (
            store["games"]
                .merge(store["competitions"], how='left')
                .merge(store["teams"].add_prefix('home_'), how='left')
                .merge(store["teams"].add_prefix('away_'), how='left')
        )
        player_games = store['player_games'].merge(store['teams'], how='left')
        teams = store['teams']

    # games = games[games['competition_name'].isin(utils.test_competitions)]
    games = games[games['competition_name'].isin(['World Cup'])]

    print(games.head())

    with pd.HDFStore(setbacks_h5) as store:
        cons_loss_setbacks = store['team_setbacks_over_matches']

    cons_loss_setbacks = cons_loss_setbacks[
        cons_loss_setbacks.apply(lambda x: set(x['lost_games']).issubset(set(games['game_id'])), axis=1)]

    all_actions = []
    for game_id in tqdm(list(games.game_id), desc="Rating actions"):
        actions = pd.read_hdf(atomic_h5, 'actions/game_{}'.format(game_id))
        actions = actions[actions['period_id'] != 5]
        actions = aspadl.add_names(actions)
        actions = actions.merge(teams, how='left')
        values = pd.read_hdf(a_predictions_h5, 'game_{}'.format(game_id))
        all_actions.append(pd.concat([actions, values], axis=1))

    all_actions = pd.concat(all_actions).reset_index(drop=True)

    print(all_actions.head())

    # for setback in tqdm(list(cons_loss_setbacks.iterrows()), desc="Comparing response by losing chance: "):
        # TODO: how to compare performance


def compare_multiple_games_setbacks():
    atomic_h5 = 'atomic_data/spadl.h5'
    default_h5 = 'default_data/spadl.h5'
    setbacks_h5 = 'setback_data/setbacks.h5'
    a_predictions_h5 = 'atomic_data/predictions.h5'

    with pd.HDFStore(atomic_h5) as store:
        games = store['games'].merge(store['competitions'], how='left')
        player_games = store['player_games']

    games = games[games['competition_name'].isin(utils.test_competitions)]

    with pd.HDFStore(setbacks_h5) as store:
        team_setbacks = store['team_setbacks']


def compare_ingame_setbacks():
    atomic_h5 = 'atomic_data/spadl.h5'
    default_h5 = 'default_data/spadl.h5'
    setbacks_h5 = 'setback_data/setbacks.h5'
    a_predictions_h5 = 'atomic_data/predictions.h5'

    with pd.HDFStore(atomic_h5) as store:
        games = store['games'].merge(store['competitions'], how='left')
        player_games = store['player_games']

    games = games[games['competition_name'].isin(utils.test_competitions)]
    # games = games[games['competition_name'].isin(['World Cup'])]

    with pd.HDFStore(setbacks_h5) as store:
        player_setbacks = store['player_setbacks']
        team_as_player_setbacks = store['team_as_player_setbacks']

    player_setbacks = pd.concat([player_setbacks, team_as_player_setbacks])
    player_setbacks = player_setbacks[player_setbacks['game_id'].isin(games['game_id'].tolist())]

    setback_names = [
        # 'missed penalty',
        # 'missed shot',
        'goal conceded',
        # 'foul leading to goal',
        # 'bad pass leading to goal',
        # 'bad consecutive passes',
    ]
    player_setbacks = player_setbacks[player_setbacks['setback_type'].isin(setback_names)]

    a_actions = []
    d_actions = []
    for game_id in tqdm(list(games.game_id), desc="Rating actions"):
        actions = pd.read_hdf(atomic_h5, 'actions/game_{}'.format(game_id))
        actions = actions[actions['period_id'] != 5]
        actions = aspadl.add_names(actions)
        actions = utils.add_goal_diff_atomic(actions)
        actions = utils.map_big_goal_diff(actions)
        values = pd.read_hdf(a_predictions_h5, 'game_{}'.format(game_id))
        a_actions.append(pd.concat([actions, values], axis=1))

        actions = pd.read_hdf(default_h5, 'actions/game_{}'.format(game_id))
        actions[actions['period_id'] != 5]
        d_actions.append(spadl.add_names(actions))

    a_actions = pd.concat(a_actions).reset_index(drop=True)
    a_actions = utils.add_total_seconds(a_actions, games)
    d_actions = pd.concat(d_actions).reset_index(drop=True)
    d_actions = utils.add_total_seconds(d_actions, games)

    responses = get_responses(player_setbacks, player_games, a_actions, d_actions)

    compstore_h5 = 'setback_data/comparisons.h5'
    with pd.HDFStore(compstore_h5) as compstore:
        for setback_id in responses.keys():
            compstore['ingame_comp_{}'.format(setback_id)] = responses[setback_id]


def get_average_response():
    atomic_h5 = 'atomic_data/spadl.h5'
    with pd.HDFStore(atomic_h5) as store:
        games = store['games'].merge(store["competitions"], how='left')

    games = games[games['competition_name'].isin(utils.test_competitions)]

    setbacks_h5 = 'setback_data/setbacks.h5'
    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore['player_setbacks']
        team_as_player_setbacks = setbackstore['team_as_player_setbacks']

    player_setbacks = pd.concat([player_setbacks, team_as_player_setbacks])
    player_setbacks = player_setbacks[player_setbacks['game_id'].isin(games['game_id'].tolist())]

    setback_names = [
        'missed penalty',
        'missed shot',
        # 'goal conceded',
        'foul leading to goal',
        'bad pass leading to goal',
        'bad consecutive passes',
    ]
    player_setbacks = player_setbacks[player_setbacks['setback_type'].isin(setback_names)]

    responses: Dict[Tuple, List[pd.DataFrame]] = {}
    compstore_h5 = 'setback_data/comparisons.h5'
    with pd.HDFStore(compstore_h5) as compstore:
        for setback in player_setbacks.itertuples():
            try:
                setback_response = compstore['ingame_comp_{}'.format(setback.setback_id)]
            except:
                continue
            key = (setback.player_id, setback.setback_type)

            if key in responses:
                responses[key].append(setback_response)
            else:
                responses[key] = [setback_response]

    avg_response_h5 = 'setback_data/avg_response.h5'
    # TODO: rescale with minutes played before/after
    with pd.HDFStore(avg_response_h5) as store:
        for key in responses.keys():
            number = len(responses[key])
            print()
            print()
            response = pd.concat(responses[key])
            grouped_by_index = response.groupby(response.index, sort=False)
            mean = grouped_by_index.mean()
            mean.loc['relative difference'] = (mean.loc['after_setback'] - mean.loc['before_setback']) / mean.loc[
                'before_setback']
            new_key = '{}: {} ({})'.format(key[0], key[1], number)
            store[new_key] = mean


def compare_for_setback(setback_type: str, metric='vaep_per_minute'):
    atomic_h5 = 'atomic_data/spadl.h5'
    with pd.HDFStore(atomic_h5) as store:
        players = store['players'].set_index('player_id')

    avg_responses = {}
    avg_response_h5 = 'setback_data/avg_response.h5'
    with pd.HDFStore(avg_response_h5) as store:
        for key in store.keys():
            number = int(key.split('(')[1][:-1])
            if setback_type in key and number >= 10:
                avg_responses[key] = store[key]

    print(len(avg_responses))

    best_responses = sorted(avg_responses.items(), key=lambda x: x[1].at['difference', metric], reverse=True)
    zero_responses = sorted(avg_responses.items(), key=lambda x: abs(x[1].at['difference', metric]), reverse=False)
    worst_responses = sorted(avg_responses.items(), key=lambda x: x[1].at['difference', metric], reverse=False)

    print("Best responses: ")
    print()
    for response in best_responses[:5]:
        player_id = int(response[0].split(':')[0][1:])
        print(players.loc[player_id, 'nickname'])
        print(response[0])
        print()
        print(response[1])
        print()
        print()
    print("-----------------------------------------------------------------------------------------------------------")
    print()
    print("Average responses: ")
    print()
    for response in zero_responses[:5]:
        player_id = int(response[0].split(':')[0][1:])
        print(players.loc[player_id, 'nickname'])
        print(response[0])
        print()
        print(response[1])
        print()
        print()
    print("-----------------------------------------------------------------------------------------------------------")
    print()
    print("Worst responses: ")
    print()
    for response in worst_responses[:5]:
        player_id = int(response[0].split(':')[0][1:])
        print(players.loc[player_id, 'nickname'])
        print(response[0])
        print()
        print(response[1])
        print()
        print()


def compare_for_players(setback_type: str, player_ids: List[int], metric='vaep_per_minute'):
    atomic_h5 = 'atomic_data/spadl.h5'
    with pd.HDFStore(atomic_h5) as store:
        players = store['players'].set_index('player_id')

    avg_responses = {}
    avg_response_h5 = 'setback_data/avg_response.h5'
    with pd.HDFStore(avg_response_h5) as store:
        for key in store.keys():
            number = int(key.split('(')[1][:-1])
            if setback_type in key and number >= 10:
                player_id = int(key.split(':')[0][1:])
                if player_id in player_ids:
                    avg_responses[key] = store[key]

    print(len(avg_responses))
    print()

    best_responses = sorted(avg_responses.items(), key=lambda x: x[1].at['difference', metric], reverse=True)

    print("Best responses: ")
    print()
    index = min(30, len(best_responses))
    for response in best_responses[:index]:
        player_id = int(response[0].split(':')[0][1:])
        print(players.loc[player_id, 'nickname'])
        print(response[0])
        print()
        print(response[1])
        print()
        print()


def main():
    compare_response_by_losing_chance()
    # compare_ingame_setbacks()
    # get_average_response()

    mp = 'missed penalty'
    ms = 'missed shot'
    gc = 'goal conceded'
    fg = 'foul leading to goal'
    bpg = 'bad pass leading to goal'
    bp = 'bad consecutive passes'

    vpa = 'vaep_per_action'
    vpm = 'vaep_per_minute'
    atob = 'avg_time_on_ball'
    tobpm = 'time_on_ball_per_minute'
    sr = 'success_rate'
    ar = 'avg_risk'

    # compare_for_setback(setback_type=bp, metric=ar)
    # compare_for_players(setback_type=ms, player_ids=utils.players, metric=ar)


if __name__ == '__main__':
    main()
