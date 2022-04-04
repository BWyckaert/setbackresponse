import json
import os
import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import utils

from typing import List
from tqdm import tqdm


def get_score(game: pd.Series, actions: pd.DataFrame, setback: pd.Series, atomic: bool) -> str:
    """
    Calculates the score in a game just before the given setback occurs

    :param game: the game for which the score should be found
    :param actions: all the actions in the game
    :param setback: the setback which defines the time in seconds when the score should be calculated
    :param atomic: boolean flag indicating whether or not the actions should be atomic
    :return: a string containing the score of the game just before the given setback occurs
    """
    # Select all actions in the game that occur before the given setback
    before_setback = actions[actions['total_seconds'] < setback.total_seconds]

    homescore = 0
    awayscore = 0

    if not atomic:
        shotlike = ['shot', 'shot_penalty', 'shot_freekick']
        # Select all actions in before_setback that result in a goal
        goal_actions = before_setback[
            (before_setback['type_name'].isin(shotlike) & (before_setback['result_name'] == 'success')) | (
                    before_setback['result_name'] == 'owngoal')]
        # Iterate over all goal actions and increment homescore and awayscore based on which team scores and whether it
        # is a normal goal or an owngoal
        for action in goal_actions.itertuples():
            if action.result_name == 'success':
                if action.team_id == game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
            if action.result_name == 'owngoal':
                if action.team_id != game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
    else:
        goal_like = ['goal', 'owngoal']
        # Select all actions in before_setback that result in a goal
        goal_actions = before_setback[before_setback['type_name'].isin(goal_like)]
        # Iterate over all goal actions and increment homescore and awayscore based on which team scores and whether it
        # is a normal goal or an owngoal
        for action in goal_actions.itertuples():
            if action.type_name == 'goal':
                if action.team_id == game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
            if action.type_name == 'owngoal':
                if action.team_id != game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1

    score = str(homescore) + ' - ' + str(awayscore)

    return score


def get_game_details(setback: pd.Series, games: pd.DataFrame) -> (pd.Series, bool, str):
    """
    Gets some game details for the game in which the given setback occurs, namely the game itself, whether or not
    the player who suffers the setback plays at home and who the opponent is

    :param setback: the setback for which the game details should be returned
    :param games: all the possible games
    :return: a series containing the game in which the setback occurs, a boolean home indicating whether the player who
    suffers the setback plays at home and a string representing the opponent
    """
    # Select the game in which the setback occurs
    game = games.set_index('game_id').loc[setback.game_id]
    # Retrieve the opponent of the player who suffers the setback and whether or not he plays at home or away
    if setback.team_id == game.home_team_id:
        home = True
        opponent = game.away_team_name_short
    else:
        home = False
        opponent = game.home_team_name_short

    return game, home, opponent


def get_missed_penalties(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Finds all missed penalties (not during penalty shootouts) in the given games and returns them in an appropriate
    dataframe

    :param games: the games for which the missed penalties must be found
    :param actions: all the actions in the given games
    :param atomic: boolean flag indicating whether or not the actions are atomic
    :return: a dataframe of all missed penalties in the given games
    """
    print("Finding missed penalties... ")
    print()

    # Select all penalties that do not occur during penalty shootouts
    missed_penalties = actions[actions['type_name'] == 'shot_penalty']

    shotlike = ['shot', 'shot_penalty', 'shot_freekick']
    for index, mp in missed_penalties.iterrows():
        # Get all actions within 5 seconds after the missed penalty (including the penalty)
        ca = actions[(actions['game_id'] == mp.game_id) & (actions['total_seconds'] >= mp.total_seconds) & (
                actions['total_seconds'] < (mp.total_seconds + 5)) & (actions['period_id'] == mp.period_id)]

        # If the penalty or one of the following actions results in a goal, remove the penalty
        if not atomic:
            ca = ca[
                ((ca['type_name'].isin(shotlike)) & (ca['result_name'] == 'success') & (
                        ca['team_id'] == mp.team_id)) | (
                        (ca['result_name'] == 'owngoal') & ~(ca['team_id'] == mp.team_id))]
            if not ca.empty:
                missed_penalties = missed_penalties.drop(index)
        else:
            ca = ca[((ca['type_name'] == 'goal') & (ca['team_id'] == mp.team_id)) | (
                    (ca['type_name'] == 'owngoal') & ~(ca['team_id'] == mp.team_id))]
            if not ca.empty:
                missed_penalties = missed_penalties.drop(index)

    # Construct a dataframe with all missed penalties
    setbacks = []
    for mp in missed_penalties.itertuples():
        game, home, opponent = get_game_details(mp, games)
        score = get_score(game, actions[actions['game_id'] == game.game_id], mp, atomic)

        setbacks.append(pd.DataFrame(
            data={'player': [mp.nickname],
                  'player_id': [mp.player_id],
                  'birth_date': [mp.birth_date],
                  'player_team': [mp.team_name_short],
                  'opponent_team': [opponent],
                  'game_id': [mp.game_id],
                  'home': [home],
                  'setback_type': ['missed penalty'],
                  'period_id': [mp.period_id],
                  'time_seconds': [mp.time_seconds],
                  'total_seconds': [mp.total_seconds],
                  'score:': [score]
                  }
        ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def get_missed_shots(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Finds all shots that did not result in a goal that are tagged by Wyscout as an opportunity and that occur within
    a certain distance from goal (11m for shots, 5m for headers)

    :param games: the games for which the missed shots must be found
    :param actions: all the actions in the given games
    :param atomic: boolean flag indicating whether or not the actions are atomic
    :return: a dataframe of all the missed shots in the given games
    """
    print("Finding missed shots... ")
    print()

    # Select all shots in the given games
    shots = actions[actions['type_name'] == 'shot']
    shots = shots.merge(games[['game_id', 'competition_id']], left_on='game_id', right_on='game_id')
    # Group shots by their competition_id, as Wyscout stores events of one competition in one file
    shots_grouped_by_competition_id = shots.groupby('competition_id')

    missed_shots = []
    # Select all shots that are tagged as an opportunity (id: 201) and inaccurate (id: 1802) by Wyscout annotators
    for competition_id, ms_by_competition in shots_grouped_by_competition_id:
        # Open Wyscout events of the competition represented by competition_id
        with open(os.path.join('wyscout_data', utils.competition_index.at[competition_id, 'db_events']), 'rt',
                  encoding='utf-8') as we:
            events = pd.DataFrame(json.load(we))
        ms_by_competition = ms_by_competition.merge(events[['id', 'tags']], left_on='original_event_id', right_on='id')

        # Reformat tags from list of dicts to list
        ms_by_competition['tags'] = ms_by_competition.apply(lambda x: [d['id'] for d in x['tags']], axis=1)

        # Select all shots that are tagged as an opportunity
        opportunity = ms_by_competition[pd.DataFrame(ms_by_competition['tags'].tolist()).isin([201]).any(axis=1).values]

        # Select all opportunities that are tagged as inaccurate
        inaccurate = opportunity[pd.DataFrame(opportunity['tags'].tolist()).isin([1802]).any(axis=1).values]
        missed_shots.append(inaccurate)

    missed_shots = pd.concat(missed_shots).reset_index(drop=True)

    # Select shots that do not result in a goal where the distance to goal is smaller than 11m for shots and 5m for
    # headers
    if not atomic:
        missed_shots = missed_shots[missed_shots['result_name'] == 'fail']
        missed_kicks = missed_shots[missed_shots['bodypart_name'] == 'foot']
        missed_headers = missed_shots[missed_shots['bodypart_name'] == 'head/other']
        missed_kicks = missed_kicks[pow(105 - missed_kicks['start_x'], 2) + pow(34 - missed_kicks['start_y'], 2) <= 121]
        missed_headers = missed_headers[
            pow(105 - missed_headers['start_x'], 2) + pow(34 - missed_headers['start_y'], 2) <= 25]
        missed_shots = pd.concat([missed_kicks, missed_headers])
    else:
        for index, action in missed_shots.iterrows():
            if actions.iloc[index + 1].type_name == 'goal':
                missed_shots.drop(index, inplace=True)
        missed_kicks = missed_shots[missed_shots['bodypart_name'] == 'foot']
        missed_headers = missed_shots[missed_shots['bodypart_name'] == 'head/other']
        missed_kicks = missed_kicks[pow(105 - missed_kicks['x'], 2) + pow(34 - missed_kicks['y'], 2) <= 121]
        missed_headers = missed_headers[pow(105 - missed_headers['x'], 2) + pow(34 - missed_headers['y'], 2) <= 25]
        missed_shots = pd.concat([missed_kicks, missed_headers])

    # Construct a dataframe with all missed shots
    setbacks = []
    for ms in missed_shots.itertuples():
        game, home, opponent = get_game_details(ms, games)
        score = get_score(game, actions[actions['game_id'] == game.game_id], ms, atomic)

        setbacks.append(pd.DataFrame(
            data={'player': [ms.nickname],
                  'player_id': [ms.player_id],
                  'birth_date': [ms.birth_date],
                  'player_team': [ms.team_name_short],
                  'opponent_team': [opponent],
                  'game_id': [ms.game_id],
                  'home': [home],
                  'setback_type': ['missed shot'],
                  'period_id': [ms.period_id],
                  'time_seconds': [ms.time_seconds],
                  'total_seconds': [ms.total_seconds],
                  'score:': [score]
                  }
        ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def get_game_details_gc(setback: pd.Series, games: pd.DataFrame, owngoal: bool) -> (pd.Series, bool, str, str):
    """
    Get game details for a goal conceded setback.

    :param setback: a series of a setback
    :param games: a dataframe of games
    :param owngoal: the goal is an owngoal
    :return: a series containing the game in which the setback occurs, a boolean home indicating whether the player who
    suffers the setback plays at home and a string representing the team of the player and the opponent
    """
    # Select the game in which the setback occurs
    game = games.set_index('game_id').loc[setback.game_id]
    # Retrieve the opponent of the player who suffers the setback, his own team and whether or not he plays at home or
    # away
    if ((setback.team_id == game.home_team_id) and not owngoal) or ((setback.team_id != game.home_team_id) and owngoal):
        home = False
        gc_team = game.away_team_name_short
        opponent = game.home_team_name_short
    else:
        home = True
        gc_team = game.home_team_name_short
        opponent = game.away_team_name_short

    return game, home, gc_team, opponent


def get_goal_conceded(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Finds all the goals conceded in the given games.

    :param games: a dataframe of games
    :param actions: a dataframe of actions
    :param atomic: the actions are atomic
    :return: a dataframe of all the conceded goals in the given games
    """
    print("Finding conceded goals... ")
    print()

    # Get all goals in all games
    if not atomic:
        shotlike = ['shot', 'shot_penalty', 'shot_freekick']
        goals = actions[((actions['type_name'].isin(shotlike)) & (actions['result_name'] == 'success')) | (
                actions['result_name'] == 'owngoal')]
    else:
        goals = actions[(actions['type_name'] == 'goal') | (actions['type_name'] == 'owngoal')]

    # Construct a dataframe with all conceded goals
    setbacks = []
    for goal in goals.itertuples():
        owngoal = goal.result_name == 'owngoal' if not atomic else goal.type_name == 'owngoal'
        game, home, gc_team, opponent = get_game_details_gc(goal, games, owngoal)
        score = get_score(game, actions[actions['game_id'] == game.game_id], goal, atomic)

        setbacks.append(pd.DataFrame(
            data={'team': [gc_team],
                  'opponent': [opponent],
                  'game_id': [goal.game_id],
                  'home': [home],
                  'setback_type': ['goal conceded'],
                  'period_id': [goal.period_id],
                  'time_seconds': [goal.time_seconds],
                  'total_seconds': [goal.total_seconds],
                  'score': [score]
                  }
        ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def get_foul_leading_to_goal(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Get all fouls in the given games that lead to a goal within 10 seconds after the freekick that follows.

    :param games: a dataframe of games
    :param actions: a dataframe of actions
    :param atomic: the actions are atomic
    :return: a dataframe of all the fouls leading to goals in the given games
    """
    print("Finding fouls leading to goals... ")
    print()

    # Get all freekick_like actions, both atomic and non-atomic
    freekick_like = ['shot_penalty', 'freekick', 'freekick_crossed', 'freekick_short', 'shot_freekick']
    freekicks = actions[actions['type_name'].isin(freekick_like) & (actions.shift(1)['type_name'] == 'foul')]

    # Only consider freekicks on the opponents half
    if not atomic:
        freekicks = freekicks[freekicks['start_x'] > 52.5]
    else:
        freekicks = freekicks[freekicks['x'] > 52.5]

    setbacks = []
    shotlike = ['shot', 'shot_penalty', 'shot_freekick']
    for index, freekick in freekicks.iterrows():
        # Get all actions within 10 seconds after the freekick
        ca = actions[
            (actions['game_id'] == freekick.game_id) & (actions['total_seconds'] >= freekick.total_seconds) & (
                    actions['total_seconds'] < (freekick.total_seconds + 10)) & (
                    actions['period_id'] == freekick.period_id)]

        if not atomic:
            ca = ca[
                ((ca['type_name'].isin(shotlike)) & (ca['result_name'] == 'success') & (
                        ca['team_id'] == freekick.team_id)) | (
                        (ca['result_name'] == 'owngoal') & ~(ca['team_id'] == freekick.team_id))]
        else:
            ca = ca[((ca['type_name'] == 'goal') & (ca['team_id'] == freekick.team_id)) | (
                    (ca['type_name'] == 'owngoal') & ~(ca['team_id'] == freekick.team_id))]

        # Check if previous action was a foul (can be offside)
        if (not ca.empty) and (actions.iloc[index - 1].type_name == 'foul'):
            foul = actions.iloc[index - 1]
            game, home, opponent = get_game_details(foul, games)
            score = get_score(game, actions[actions.game_id == game.game_id], foul, atomic)

            setbacks.append(pd.DataFrame(
                data={'player': [foul.nickname],
                      'player_id': [foul.player_id],
                      'birth_date': [foul.birth_date],
                      'player_team': [foul.team_name_short],
                      'opponent_team': [opponent],
                      'game_id': [foul.game_id],
                      'home': [home],
                      'setback_type': ['foul leading to goal'],
                      'period_id': [foul.period_id],
                      'time_seconds': [foul.time_seconds],
                      'total_seconds': [foul.total_seconds],
                      'score:': [score]
                      }
            ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def get_bad_pass_leading_to_goal(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Get all bad passes that lead to a goal within 15 seconds after the pass.

    :param games: a dataframe of games
    :param actions: a dataframe of actions
    :param atomic: the actions are atomic
    :return: a dataframe of all the bad passes leading to a goal
    """
    print("Finding bad pass leading to goal... ")
    print()

    if not atomic:
        shotlike = {'shot', 'shot_penalty', 'shot_freekick'}
        goals = actions[((actions['type_name'].isin(shotlike)) & (actions['result_name'] == 'success')) | (
                actions['result_name'] == 'owngoal')]
    else:
        goals = actions[(actions['type_name'] == 'goal') | (actions['type_name'] == 'owngoal')]

    setbacks = []
    for index, goal in goals.iterrows():
        # Get all actions within 15 seconds before the goal
        pa = actions[(actions['game_id'] == goal.game_id) & (actions['total_seconds'] < goal.total_seconds) & (
                actions['total_seconds'] > (goal.total_seconds - 15)) & (actions['period_id'] == goal.period_id)]

        if not atomic:
            bad_pass = pa[(pa['type_name'] == 'pass') & (pa['result_name'] == 'fail')]
            # Normal goal
            if goal.result_name == 'success':
                bad_pass = bad_pass[bad_pass['team_id'] != goal.team_id]
            # Owngoal
            else:
                bad_pass = bad_pass[bad_pass['team_id'] == goal.team_id]
            bad_pass = bad_pass[bad_pass['start_x'] < 65]

        else:
            bad_pass = pa[(pa['type_name'] == 'pass') & (pa.shift(-1)['type_name'] == 'interception')]
            # Normal goal
            if goal.type_name == 'goal':
                bad_pass = bad_pass[bad_pass['team_id'] != goal.team_id]
            # Owngoal
            else:
                bad_pass = bad_pass[bad_pass['team_id'] == goal.team_id]
            bad_pass = bad_pass[bad_pass['x'] < 65]

        if not bad_pass.empty:
            bad_pass = bad_pass.iloc[-1]
            # Don't consider bad passes where a change of possession occurs between the bad pass and the goal
            # A keeper save or tackle are not considered change of possession
            if not pa[(pa['total_seconds'] > bad_pass.total_seconds) & (pa['team_id'] == bad_pass.team_id) & (
                    ~pa['type_name'].isin(['keeper_save', 'tackle']))].empty:
                continue

            game, home, opponent = get_game_details(bad_pass, games)
            score = get_score(game, actions[actions.game_id == game.game_id], bad_pass, atomic)

            setbacks.append(pd.DataFrame(
                data={'player': [bad_pass.nickname],
                      'player_id': [bad_pass.player_id],
                      'birth_date': [bad_pass.birth_date],
                      'player_team': [bad_pass.team_name_short],
                      'opponent_team': [opponent],
                      'game_id': [bad_pass.game_id],
                      'home': [home],
                      'setback_type': ['bad pass leading to goal'],
                      'period_id': [bad_pass.period_id],
                      'time_seconds': [bad_pass.time_seconds],
                      'total_seconds': [bad_pass.total_seconds],
                      'score:': [score]
                      }
            ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def get_bad_consecutive_passes(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Gets all consecutive bad passes.

    :param games: a dataframe of games
    :param actions: a dataframe of actions
    :param atomic: the actions are atomic
    :return: a dataframe with all the last passes in a sequence of bad passes
    """
    # TODO: rewrite using majority of passes missed
    print("Finding bad consecutive passes... ")
    print()

    last_bad_pass_in_seq = []
    grouped_by_game_period = actions.groupby(['game_id', 'period_id'])
    for _, game_actions in grouped_by_game_period:
        grouped_by_player = game_actions.groupby('player_id')
        for player_id, player_actions in grouped_by_player:
            pass_actions = player_actions[player_actions['type_name'] == 'pass'].reset_index(drop=True)

            # If there are no pass actions, go to next player
            if pass_actions.empty:
                continue

            if not atomic:
                # Group consecutive failed and successful passes
                grouped_by_success_failure = pass_actions.groupby(
                    [(pass_actions.result_name != pass_actions.shift(1).result_name).cumsum(), 'period_id'])
            else:
                # Add result_name column to dataframe to mimic non-atomic action
                next_actions = game_actions[
                    (game_actions.shift(1)['type_name'] == 'pass') & (
                                game_actions.shift(1)['player_id'] == player_id) & (
                            game_actions.shift(1)['period_id'] == game_actions['period_id'])].reset_index(drop=True)
                pass_actions['result_name'] = next_actions.apply(
                    lambda y: 'success' if (y.type_name == 'receival') else 'fail', axis=1)

                grouped_by_success_failure = pass_actions.groupby(
                    [(pass_actions['result_name'] != pass_actions.shift(1)['result_name']).cumsum(), 'period_id'])

            for _, sf in grouped_by_success_failure:
                sf = sf.reset_index(drop=True)
                # Don't consider successful passes
                if sf.iloc[0].result_name == 'success':
                    continue
                # Only consider at least 3 bad consecutive passes
                if sf.shape[0] < 3:
                    continue
                # Only consider 3 bad passes within 300 seconds of each other
                while sf.shape[0] >= 3:
                    if sf.at[2, 'total_seconds'] - sf.at[0, 'total_seconds'] < 300:
                        last_bad_pass_in_seq.append(sf.iloc[2].to_frame().T)
                        break
                    else:
                        sf = sf.iloc[1:].reset_index(drop=True)

    last_bad_pass_in_seq = pd.concat(last_bad_pass_in_seq).reset_index(drop=True)

    setbacks = []
    for bad_pass in last_bad_pass_in_seq.itertuples():
        game, home, opponent = get_game_details(bad_pass, games)
        score = get_score(game, actions[actions.game_id == game.game_id], bad_pass, atomic)

        setbacks.append(pd.DataFrame(
            data={'player': [bad_pass.nickname],
                  'player_id': [bad_pass.player_id],
                  'birth_date': [bad_pass.birth_date],
                  'player_team': [bad_pass.team_name_short],
                  'opponent_team': [opponent],
                  'game_id': [bad_pass.game_id],
                  'home': [home],
                  'setback_type': ['bad consecutive passes'],
                  'period_id': [bad_pass.period_id],
                  'time_seconds': [bad_pass.time_seconds],
                  'total_seconds': [bad_pass.total_seconds],
                  'score:': [score]
                  }
        ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def lost_game(game: pd.Series, team_id: int) -> bool:
    """
    Finds out whether or not the given game was lost by the given team.

    :param game: a game series
    :param team_id: a team id
    :return: true if the team with team_id lost the game, false otherwise
    """
    home = game.home_team_id == team_id
    score = game.label.split(", ")[1][0:5]
    if home:
        return int(score[0]) < int(score[4])
    else:
        return int(score[0]) > int(score[4])


def convert_to_fair_chance(games_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the betting odds to fair winning/drawing/losing chances.

    :param games_odds: a dataframe of game odds
    :return: a dataframe with the 1x2 betting odds converted to fair chances
    """
    # Calculate the margin used (assuming equal margin)
    games_odds['margin'] = games_odds.apply(
        lambda x: (1 / x['B365H']) + (1 / x['B365D']) + (1 / x['B365A']), axis=1)

    # Convert each betting odds column to fair chance
    odds_columns = ['B365H', 'B365D', 'B365A']
    for column in odds_columns:
        games_odds[column] = games_odds.apply(
            lambda x: round((1 / x[column]) / x['margin'], 2), axis=1
        )

    games_odds = games_odds.rename(columns={'B365H': 'home_win', 'B365D': 'draw', 'B365A': 'away_win'})

    return games_odds


def add_chance_to_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Add winning/drawing/losing chance to the games dataframe.

    :param games: a dataframe of games
    :return: a dataframe of games with three added columns denoting the winning/drawing/losing chance
    """
    odds = []
    for competition in list(utils.competitions_mapping.values()):
        odds.append(pd.read_csv('betting_data/odds_{}.csv'.format(competition)))

    odds = pd.concat(odds).reset_index(drop=True)

    # Rename and convert necessary column and values to merge with games
    odds['Date'] = pd.to_datetime(odds['Date'], yearfirst=True, infer_datetime_format=True).dt.date
    odds = odds.replace({'HomeTeam': utils.teams_mapping, 'AwayTeam': utils.teams_mapping})
    odds = odds[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']]
    odds = odds.rename(
        columns={'HomeTeam': 'home_team_name_short', 'AwayTeam': 'away_team_name_short', 'Date': 'game_date'})

    # Take the date out of date_time to match the odds game_date
    games['game_date'] = games.game_date.dt.date

    games_odds = games.merge(odds, on=['home_team_name_short', 'away_team_name_short', 'game_date'])
    games_odds = convert_to_fair_chance(games_odds)

    return games_odds


def get_consecutive_losses(games: pd.DataFrame) -> pd.DataFrame:
    """
    Get consecutive losses with a chance of happening less that 30%.

    :param games: a dataframe of games
    :return: a dataframe with all consecutive losses
    """
    print("Finding consecutive losses... ")
    print()

    games_with_chance = add_chance_to_games(games)

    setbacks = []
    games_by_home_team = games_with_chance.groupby('home_team_id')
    games_by_away_team = games_with_chance.groupby('away_team_id')
    for team_id, games_for_team in games_by_home_team:
        cons_losses = []

        # Combine the teams home games with its away games
        games_for_team = pd.concat([games_for_team, games_by_away_team.get_group(team_id)]).sort_values('game_date')

        # Add column denoting lost/won game and losing chance
        games_for_team['lost_game'] = games_for_team.apply(lambda x: lost_game(x, team_id), axis=1)
        games_for_team['losing_chance'] = games_for_team.apply(
            lambda x: x['away_win'] if (x['home_team_id'] == team_id) else x['home_win'], axis=1)

        # Group games by consecutive wins/losses
        grouped_by_loss_wins = games_for_team.groupby(
            [(games_for_team['lost_game'] != games_for_team.shift(1)['lost_game']).cumsum()])

        for _, lw in grouped_by_loss_wins:
            lw = lw.reset_index(drop=True)
            # Don't consider wins
            if not lw.iloc[0]['lost_game']:
                continue
            # Add column denoting the chance of losing a sequence of games
            lw['cons_loss_chance'] = lw['losing_chance'].cumprod()
            for index, game in lw.iterrows():
                if game['cons_loss_chance'] < 0.30:
                    cons_losses.append(lw[:index + 1])
                    # break

        for cons_loss in cons_losses:
            last_loss = cons_loss.iloc[-1]
            team = last_loss.home_team_name_short if last_loss.home_team_id == team_id \
                else last_loss.away_team_name_short

            setbacks.append(pd.DataFrame(
                data={'team': [team],
                      'lost games': [cons_loss['game_id'].tolist()],
                      'game_date_last_loss': [last_loss['game_date']],
                      'competition': [last_loss['competition_name']],
                      'setback_type': ['consecutive losses'],
                      'chance': [last_loss['cons_loss_chance']]
                      }
            ))

    setbacks = pd.concat(setbacks).reset_index(drop=True)

    return setbacks


def convert_team_to_player_setbacks(team_setbacks: pd.DataFrame, player_games: pd.DataFrame, actions: pd.DataFrame,
                                    players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    Converts in game teams setbacks to individual player setbacks.

    :param team_setbacks: a dataframe of team setbacks
    :param player_games: a dataframe of player games
    :param actions: a dataframe of actions
    :param players: a dataframe of players
    :param teams: a dataframe of teams
    :return: a dataframe of player setbacks from the team setbacks
    """
    players_by_id = players.set_index('player_id', drop=True)
    player_setbacks = []
    player_games = player_games.merge(teams[['team_id', 'team_name_short']], on='team_id')

    # For each team setback, find which players were on the field at the time of the setback and create individual
    # setback for each of those players
    for team_setback in tqdm(list(team_setbacks.itertuples()), desc="Converting team to player setbacks:"):
        actions_in_game = actions[actions.game_id == team_setback.game_id]

        # Get the minute of the setback in the same way the substitution minute of spadl is calculated
        group_by_period = actions_in_game.groupby('period_id')
        periods_duration = []
        for _, period in group_by_period:
            periods_duration.append(round(period.time_seconds.max() / 60))
        minutes_before_current_period = sum(periods_duration[:team_setback.period_id - 1])
        minute_of_setback = round(team_setback.time_seconds / 60) + minutes_before_current_period

        # Get all the players in the game of the setback
        players_in_game = player_games[
            (player_games.game_id == team_setback.game_id) & (player_games.team_name_short == team_setback.team)]

        players_on_field = []

        # Get the players on the field at the time of the setback who were a starter
        for player in players_in_game[players_in_game.is_starter].itertuples():
            if player.minutes_played > minute_of_setback:
                players_on_field.append(player)

        # Get the players on the field at the time of the setback who were a substitute
        for player in players_in_game[~players_in_game.is_starter].itertuples():
            if (players_in_game.minutes_played.max() - player.minutes_played) < minute_of_setback:
                players_on_field.append(player)

        # Construct a player setback for every player on the field at the time of the team setback
        for player in players_on_field:
            player = players_by_id.loc[player.player_id]
            player_setbacks.append(pd.DataFrame(
                data={'player': [player.nickname],
                      'player_id': [player.player_id],
                      'birth_date': [player.birth_date],
                      'player_team': [team_setback.team],
                      'opponent_team': [team_setback.opponent],
                      'game_id': [team_setback.game_id],
                      'home': [team_setback.home],
                      'setback_type': ['goal conceded'],
                      'period_id': [team_setback.period_id],
                      'time_seconds': [team_setback.time_seconds],
                      'total_seconds': [team_setback.total_seconds],
                      'score:': [team_setback.score]
                      }
            ))

    player_setbacks = pd.concat(player_setbacks).reset_index(drop=True)

    return player_setbacks


def get_setbacks(competitions: List[str], atomic=False) -> tuple:
    """
    Gets all the setbacks in the given competitions.

    :param competitions: a list of competition names
    :param atomic: the actions are atomic
    :return: three dataframes of player setbacks, in game team setbacks and over game team setbacks
    """
    if atomic:
        _spadl = aspadl
        datafolder = 'atomic_data'
    else:
        _spadl = spadl
        datafolder = 'default_data'

    spadl_h5 = os.path.join(datafolder, 'spadl.h5')

    all_actions = []
    with pd.HDFStore(spadl_h5) as store:
        games = (
            store['games']
                .merge(store['competitions'], how='left')
                .merge(store['teams'].add_prefix('home_'), how='left')
                .merge(store['teams'].add_prefix('away_'), how='left')
        )
        # Only consider setbacks in the given competitions
        games = games[games['competition_name'].isin(competitions)]
        player_games = store['player_games']
        players = store['players']
        teams = store['teams']

        for game_id in tqdm(games.game_id, "Collecting all actions: "):
            actions = store['actions/game_{}'.format(game_id)]
            actions = actions[actions['period_id'] != 5]
            actions = (
                _spadl.add_names(actions).merge(players, how='left').merge(teams, how='left').sort_values(
                    ['game_id', 'period_id', 'action_id'])
            )
            all_actions.append(actions)

    all_actions = pd.concat(all_actions)
    all_actions = utils.add_total_seconds(all_actions, games)
    all_actions = utils.left_to_right(games, all_actions, _spadl)

    player_setbacks = [get_missed_penalties(games, all_actions, atomic),
                       get_missed_shots(games, all_actions, atomic),
                       get_foul_leading_to_goal(games, all_actions, atomic),
                       get_bad_pass_leading_to_goal(games, all_actions, atomic),
                       get_bad_consecutive_passes(games, all_actions, atomic)]

    player_setbacks = pd.concat(player_setbacks).reset_index(drop=True)

    team_setbacks = get_goal_conceded(games, all_actions, atomic)
    team_as_player_setbacks = convert_team_to_player_setbacks(team_setbacks, player_games, all_actions, players, teams)

    team_setbacks_over_matches = get_consecutive_losses(games)

    setbacks_h5 = os.path.join(datafolder, 'setbacks.h5')
    with pd.HDFStore(setbacks_h5) as store:
        store['player_setbacks'] = player_setbacks
        store['team_setbacks'] = team_setbacks
        store['team_as_player_setbacks'] = team_as_player_setbacks
        store['team_setbacks_over_matches'] = team_setbacks_over_matches

    return player_setbacks, team_setbacks, team_setbacks_over_matches


def main():
    competitions = utils.test_competitions
    get_setbacks(competitions=competitions, atomic=False)


if __name__ == '__main__':
    main()
