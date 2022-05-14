import math
import warnings
import utils

import pandas as pd
import numpy as np
import statistics as stat
import socceraction.spadl as spadl
import socceraction.atomic.spadl as aspadl
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Tuple, Dict

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)


def weighted_average(performance: pd.DataFrame) -> pd.DataFrame:
    total_actions = performance['number_of_actions'].sum()
    total_minutes = performance['minutes_played'].sum()

    vaep_per_action = performance['vaep_per_action'].mul(performance['number_of_actions']).sum() / total_actions
    vaep_per_minute = performance['vaep_per_minute'].mul(performance['minutes_played']).sum() / total_minutes

    time_on_ball_per_action = performance['time_on_ball_per_action'].mul(
        performance['number_of_actions']).sum() / total_actions
    time_on_ball_per_minute = performance['time_on_ball_per_minute'].mul(
        performance['minutes_played']).sum() / total_minutes

    success_rate_per_action = performance['success_rate_per_action'].mul(
        performance['number_of_actions']).sum() / total_actions
    risk_per_action = performance['risk_per_action'].mul(performance['number_of_actions']).sum() / total_actions
    exp_success_rate_per_action = 1 - risk_per_action

    minutes_played = performance['minutes_played'].sum()
    number_of_actions = performance['number_of_actions'].sum()
    actions_per_minute = number_of_actions / minutes_played

    performance = pd.DataFrame(
        data={'vaep_per_action': [vaep_per_action],
              'vaep_per_minute': [vaep_per_minute],
              'time_on_ball_per_action': [time_on_ball_per_action],
              'time_on_ball_per_minute': [time_on_ball_per_minute],
              'success_rate_per_action': [success_rate_per_action],
              'exp_success_rate_per_action': [exp_success_rate_per_action],
              'risk_per_action': [risk_per_action],
              'minutes_played': [minutes_played],
              'number_of_actions': [number_of_actions],
              'actions_per_minute': [actions_per_minute]},
        index=[0]
    )
    print()
    print(performance)
    print()

    return performance


def compare_over_games(player_id: int, a_actions: pd.DataFrame, d_actions: pd.DataFrame,
                       player_games: pd.DataFrame, previous_game_ids_player: List[int],
                       next_game_ids_player: List[int], consider_goals=True) -> pd.DataFrame:
    comparison = []
    for comp_games in [previous_game_ids_player, next_game_ids_player]:
        performance = []
        for game_id in comp_games:
            players_in_game = player_games[player_games['game_id'] == game_id].set_index('player_id')
            player_game = players_in_game.loc[player_id]
            game_duration = players_in_game['minutes_played'].max()

            a_game_actions = a_actions[a_actions['game_id'] == game_id]
            d_game_actions = d_actions[d_actions['game_id'] == game_id]

            pof_a_actions = utils.get_player_on_field_actions(a_game_actions, player_game, game_duration)
            pof_d_actions = utils.get_player_on_field_actions(d_game_actions, player_game, game_duration)

            a_player_actions = pof_a_actions[pof_a_actions['player_id'] == player_id]
            d_player_actions = pof_d_actions[pof_d_actions['player_id'] == player_id]

            first_action = pof_a_actions.iloc[0]
            last_action = pof_a_actions.iloc[-1]

            vaep = get_vaep_aggregates(a_player_actions, first_action, last_action, consider_goals, True)
            time_on_ball = get_time_on_ball_aggregates(a_player_actions, first_action, last_action)
            success_rate = get_success_rate(d_player_actions)
            minutes_played = player_game.minutes_played
            number_of_actions = d_player_actions.shape[0]
            risk = get_average_risk(d_player_actions)
            exp_success_rate = 1 - risk
            actions_per_minute = d_player_actions.shape[0] / minutes_played

            performance.append(pd.DataFrame(
                data={'vaep_per_action': [vaep[0]],
                      'vaep_per_minute': [vaep[1]],
                      'time_on_ball_per_action': [time_on_ball[0]],
                      'time_on_ball_per_minute': [time_on_ball[1]],
                      'success_rate_per_action': [success_rate],
                      'exp_success_rate_per_action': [exp_success_rate],
                      'risk_per_action': [risk],
                      'minutes_played': [minutes_played],
                      'number_of_actions': [number_of_actions],
                      'actions_per_minute': [actions_per_minute]},
                index=[game_id]
            ))

        comparison.append(weighted_average(pd.concat(performance)))

    diff = comparison[1] - comparison[0]
    rel_diff = diff / comparison[0]

    comparison.append(diff)
    comparison.append(rel_diff)

    comparison = pd.concat(comparison).reset_index(drop=True).rename(
        index={0: 'before_setback', 1: 'after_setback', 2: 'difference', 3: 'relative_difference'})

    summary = round(comparison, 4)

    return summary


def compare(setback: pd.Series, a_actions: pd.DataFrame, d_actions: pd.DataFrame, player_game: pd.Series,
            game_duration: int, consider_goals=True) -> pd.DataFrame:
    """
    Compare the performance of the player suffering the setback before and after the setback.

    :param setback: a setback
    :param a_actions: atomic actions during the game of the setback
    :param d_actions: non atomic actions during the game of the setback
    :param player_game: the game in which the setback occurs
    :param game_duration: the duration of the game in which the setback occurs
    :param consider_goals: whether or not to consider goals in the performance comparison
    :return: a dataframe comparing performance before and after the setback
    """
    # Only consider actions where player of setback is on field
    pof_a_actions = utils.get_player_on_field_actions(a_actions, player_game, game_duration, setback)
    pof_d_actions = utils.get_player_on_field_actions(d_actions, player_game, game_duration, setback)

    # Get atomic player actions before and after the setback
    a_actions_before, a_actions_after = get_before_after(setback, pof_a_actions)

    # Get vaep-rating before and after the setback
    vaep_before = get_vaep_aggregates(a_actions_before, pof_a_actions.iloc[0], setback, consider_goals, True)
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

    # Get the expected success rate, based on the risk
    exp_success_rate_before = 1 - risk_before
    exp_success_rate_after = 1 - risk_after

    # Get the number of actions before and after the setback
    number_of_actions_before = d_actions_before.shape[0]
    number_of_actions_after = d_actions_after.shape[0]

    # Get the actions per minute
    actions_per_minute_before = d_actions_before.shape[0] / minutes_played_before
    actions_per_minute_after = d_actions_after.shape[0] / minutes_played_after

    # Put data in dataframe
    comparison = pd.DataFrame(
        data={'vaep_per_action': data_to_list(vaep_before[0], vaep_after[0]),
              'vaep_per_minute': data_to_list(vaep_before[1], vaep_after[1]),
              'time_on_ball_per_action': data_to_list(time_on_ball_before[0], time_on_ball_after[0]),
              'time_on_ball_per_minute': data_to_list(time_on_ball_before[1], time_on_ball_after[1]),
              'success_rate_per_action': data_to_list(success_rate_before, success_rate_after),
              'exp_success_rate_per_action': data_to_list(exp_success_rate_before, exp_success_rate_after),
              'risk_per_action': data_to_list(risk_before, risk_after),
              'minutes_played': data_to_list(minutes_played_before, minutes_played_after),
              'number_of_actions': data_to_list(number_of_actions_before, number_of_actions_after),
              'actions_per_minute': data_to_list(actions_per_minute_before, actions_per_minute_after)},
        index=['before_setback', 'after_setback', 'difference', 'relative_difference']
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

    # TODO redo consider goals
    if not consider_goals:
        goal_actions = ['goal', 'owngoal', 'bad_touch']


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

    time_on_ball_per_action = stat.mean(time_on_ball) if time_on_ball else 0
    time_on_ball_per_minute = sum(time_on_ball) / ((last.total_seconds - first.total_seconds) / 60)

    return time_on_ball_per_action, time_on_ball_per_minute


def get_success_rate(player_actions: pd.DataFrame) -> float:  # Requires non-atomic actions!!
    passlike = ['pass', 'cross', 'freekick_crossed', 'freekick_short', 'corner_crossed', 'corner_short', 'clearance',
                'throw_in', 'goalkick', 'take_on']
    passes = player_actions[player_actions['type_name'].isin(passlike)]

    success = passes[passes['result_name'] == 'success'].shape[0]
    fail = passes[passes['result_name'] != 'success'].shape[0]

    if success == 0 and fail == 0:
        return 0

    success_rate = success / (success + fail)

    return success_rate


def get_average_risk(actions: pd.DataFrame) -> float:
    predictions_h5 = 'xP_data/predictions.h5'
    with pd.HDFStore(predictions_h5) as predictionstore:
        predictions = predictionstore['predictions']

    actions_on_id = actions.dropna().set_index('original_event_id')
    passlike = ['pass', 'cross', 'freekick_crossed', 'freekick_short', 'corner_crossed', 'corner_short', 'clearance',
                'throw_in', 'goalkick', 'take_on']
    actions_on_id = actions_on_id[actions_on_id['type_name'].isin(passlike)]
    actions_with_risk = actions_on_id.join(predictions).dropna()
    actions_with_risk['risk'] = 1 - actions_with_risk['exp_accuracy']

    risk = actions_with_risk['risk'].mean() if not actions_with_risk.empty else 0

    return risk


def get_responses(player_setbacks: pd.DataFrame, player_games: pd.DataFrame, a_actions: pd.DataFrame,
                  d_actions: pd.DataFrame) -> Dict[int, pd.DataFrame]:
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

    return responses


def get_responses_over_multiple_games(team_setbacks: pd.DataFrame, player_games: pd.DataFrame, a_actions: pd.DataFrame,
                                      d_actions: pd.DataFrame, games: pd.DataFrame) -> Dict[
    Tuple[int, int], pd.DataFrame]:
    responses: Dict[Tuple[int, int], pd.DataFrame] = {}
    for team_setback in tqdm(list(team_setbacks.itertuples()), desc="Comparing performance: "):
        # Only consider players who played in at least half of the lost games
        player_ids = utils.players_in_majority_of_games(player_games, team_setback.lost_games)

        previous_game_ids = get_previous_games(team_setback, games)
        next_game_ids = get_next_games(team_setback, games)

        if len(previous_game_ids) < 3 or len(next_game_ids) < 3:
            continue

        # Of the players who played in at least half of the lost games, consider only those who played in the majority
        # of comparison games (3-5 games) before and after the lost games
        player_ids = list(set(player_ids) & set(utils.players_in_majority_of_games(player_games, previous_game_ids)))
        player_ids = list(set(player_ids) & set(utils.players_in_majority_of_games(player_games, next_game_ids)))

        for player_id in player_ids:
            previous_game_ids_player = player_games[
                player_games['game_id'].isin(previous_game_ids) & (player_games['player_id'] == player_id)][
                'game_id'].tolist()
            next_game_ids_player = player_games[
                player_games['game_id'].isin(next_game_ids) & (player_games['player_id'] == player_id)][
                'game_id'].tolist()

            response = compare_over_games(player_id, a_actions, d_actions, player_games,
                                          previous_game_ids_player, next_game_ids_player)
            break

        break


def get_performance_per_minute_in_games(game_ids: List[int], team: str, actions: pd.DataFrame,
                                        player_games: pd.DataFrame) -> float:
    team_actions = actions[actions['team_name_short'] == team]
    total_vaep = 0
    total_minutes = 0
    for game_id in game_ids:
        actions_in_game = team_actions[team_actions['game_id'] == game_id]
        total_vaep += actions_in_game['vaep_value'].sum()
        total_minutes += player_games.loc[player_games['game_id'] == game_id, 'minutes_played'].max()

    vaep_per_minute = total_vaep / total_minutes

    return vaep_per_minute


def get_next_games(setback: pd.Series, games: pd.DataFrame) -> List[str]:
    # Get the games in the competition in which the setback occurred
    games_in_competition = games[games['competition_name'] == setback.competition]

    # Get the games in the competition played by the team suffering the setback
    team_games = games_in_competition[(games_in_competition['home_team_name_short'] == setback.team) |
                                      (games_in_competition['away_team_name_short'] == setback.team)]

    # Get the games played after the setback occurred
    games_after_setback = team_games[team_games['game_date'].dt.date > setback.game_date_last_loss]

    # Get the number of games played after the setback occurred
    nb_games_after_setback = games_after_setback.shape[0]

    # Get the first 5 games after the setback. If there are no 5 games, take all remaining games
    nb_next_games = min(5, nb_games_after_setback)
    next_games = games_after_setback.iloc[:nb_next_games]
    next_game_ids = next_games['game_id'].tolist()

    return next_game_ids


def get_previous_games(setback: pd.Series, games: pd.DataFrame) -> List[str]:
    # Get the games in the competition in which the setback occurred
    games_in_competition = games[games['competition_name'] == setback.competition]

    # Get the games in the competition played by the team suffering the setback
    team_games = games_in_competition[(games_in_competition['home_team_name_short'] == setback.team) |
                                      (games_in_competition['away_team_name_short'] == setback.team)]

    # Get the games played after the setback occurred
    games_before_setback = team_games[team_games['game_date'].dt.date < setback.game_date_first_loss]

    # Get the number of games played after the setback occurred
    nb_games_before_setback = games_before_setback.shape[0]

    # Get the first 5 games after the setback. If there are no 5 games, take all remaining games
    nb_previous_games = min(5, nb_games_before_setback)
    previous_games = games_before_setback.iloc[:nb_previous_games]
    previous_game_ids = previous_games['game_id'].tolist()

    return previous_game_ids


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

    games = games[games['competition_name'].isin(utils.test_competitions)]

    with pd.HDFStore(setbacks_h5) as store:
        cons_loss_setbacks = store['team_setbacks_over_matches']

    cons_loss_setbacks = cons_loss_setbacks[
        cons_loss_setbacks.apply(lambda x: set(x['lost_games']).issubset(set(games['game_id'])), axis=1)]

    cons_loss_setbacks['nb_games'] = cons_loss_setbacks.apply(lambda x: len(x['lost_games']), axis=1)
    cons_loss_setbacks = cons_loss_setbacks[cons_loss_setbacks['nb_games'] > 0]

    all_actions = []
    for game_id in tqdm(list(games.game_id), desc="Rating actions"):
        actions = pd.read_hdf(atomic_h5, 'actions/game_{}'.format(game_id))
        actions = actions[actions['period_id'] != 5]
        actions = aspadl.add_names(actions)
        actions = actions.merge(teams, how='left')
        values = pd.read_hdf(a_predictions_h5, 'game_{}'.format(game_id))
        all_actions.append(pd.concat([actions, values], axis=1))

    all_actions = pd.concat(all_actions).reset_index(drop=True)

    setback_aggregates = []

    for setback in tqdm(list(cons_loss_setbacks.itertuples()), desc="Comparing response by losing chance: "):
        previous_game_ids = get_previous_games(setback, games)
        next_game_ids = get_next_games(setback, games)

        if len(previous_game_ids) < 3 or len(next_game_ids) < 3:
            continue

        performance_before = get_performance_per_minute_in_games(previous_game_ids, setback.team, all_actions,
                                                                 player_games)

        performance_after = get_performance_per_minute_in_games(next_game_ids, setback.team, all_actions, player_games)

        setback_aggregates.append(pd.DataFrame(
            data={'chance': [setback.chance],
                  'nb_lost_games': [len(setback.lost_games)],
                  'perf_diff': [performance_after - performance_before]}
        ))

    setback_aggregates = pd.concat(setback_aggregates).reset_index(drop=True)

    bins = pd.cut(x=setback_aggregates['chance'], bins=np.arange(0, 0.35, 0.05),
                  labels=['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0'])
    perf_diff = setback_aggregates.groupby(bins).agg({'perf_diff': ['mean', 'std', 'median']})

    print(perf_diff)

    fig, ax = plt.subplots()
    ax.errorbar(perf_diff.index, perf_diff['perf_diff', 'mean'], yerr=perf_diff['perf_diff', 'std'], ecolor='black',
                capsize=10)
    ax.set_ylabel("Performance difference")
    ax.set_xlabel("Chance of losing sequence")
    ax.set_title("Performance difference in function of losing chance")
    plt.tight_layout()
    plt.show()


def compare_multiple_games_setbacks():
    atomic_h5 = 'atomic_data/spadl.h5'
    default_h5 = 'default_data/spadl.h5'
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

    # games = games[games['competition_name'].isin(utils.test_competitions)]
    games = games[games['competition_name'] == 'Italian first division']

    with pd.HDFStore(setbacks_h5) as store:
        cons_loss_setbacks = store['team_setbacks_over_matches']

    cons_loss_setbacks = cons_loss_setbacks[cons_loss_setbacks['chance'] < 0.20]
    cons_loss_setbacks = cons_loss_setbacks[cons_loss_setbacks['competition'] == 'Italian first division']

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

    responses = get_responses_over_multiple_games(cons_loss_setbacks, player_games, a_actions, d_actions, games)


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
    # compare_response_by_losing_chance()
    compare_multiple_games_setbacks()
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
    tobpa = 'time_on_ball_per_action'
    tobpm = 'time_on_ball_per_minute'
    sr = 'success_rate'
    ar = 'avg_risk'

    # compare_for_setback(setback_type=bp, metric=ar)
    # compare_for_players(setback_type=ms, player_ids=utils.players, metric=ar)


if __name__ == '__main__':
    main()
