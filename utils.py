import json
import os
from typing import Dict, Any, Tuple, List

import pandas as pd

from tqdm import tqdm


def left_to_right(games: pd.DataFrame, actions: pd.DataFrame, _spadl) -> pd.DataFrame:
    """
    Changes the given actions such that all actions are performed as if the player plays from left to right.

    :param games: a dataframe with all the games from which the actions are taken
    :param actions: a dataframe with all the actions for which the direction of play should be changed
    :param _spadl:
    :return: a dataframe of actions where all actions are performed as if the player plays from left to right
    """
    ltr_actions = []
    for game in tqdm(list(games.itertuples()), desc="Converting direction of play: "):
        ltr_actions.append(_spadl.play_left_to_right(actions[actions.game_id == game.game_id], game.home_team_id))

    ltr_actions = pd.concat(ltr_actions).reset_index(drop=True)

    return ltr_actions


def add_total_seconds_to_game(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Add a total seconds column to the actions if it doesn't exist already.

    :param actions: a dataframe of actions
    :return: a dataframe of actions with an added column containing the total seconds
    """
    # No need to add total_seconds column if it exists already
    if 'total_seconds' in actions.columns:
        return actions

    # Find the timestamp of the last action in each period
    # Fifth period actions should already be removed at this point
    group_by_period = actions.groupby('period_id')
    last_action_in_period = []
    for _, period in group_by_period:
        last_action_in_period.append(period.time_seconds.max())

    # Define total seconds as the timestamp of the action in the current period plus the duration of all previous
    # periods combined
    actions['total_seconds'] = actions.apply(
        lambda x: x['time_seconds'] + sum(last_action_in_period[: int(x['period_id']) - 1]), axis=1)

    return actions


def add_total_seconds(actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Add a total seconds column the actions in the given games if it doesn't exist already.

    :param actions: a dataframe of actions
    :param games: a dataframe of games
    :return: a dataframe of actions with an added column containing the total seconds
    """
    # No need to add total_seconds column if it exists already
    if 'total_seconds' in actions.columns:
        return actions

    # Add a total_seconds column to all action in each of the given games
    extended_actions = []
    for game_id in tqdm(list(games.game_id), desc="Adding total_seconds to actions: "):
        extended_actions.append(add_total_seconds_to_game(actions[actions['game_id'] == game_id]))

    extended_actions = pd.concat(extended_actions).reset_index(drop=True)

    return extended_actions


def add_goal_diff_atomic(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Add a goal difference column to atomic actions.

    :param actions: a dataframe of actions
    :return: a dataframe of actions with an added column denoting the goal difference at the time of the actions
    """
    # Get all actions that result in a goal
    goallike = ['goal', 'owngoal']
    goal_actions = actions[actions['type_name'].isin(goallike)]

    # Initialise the score difference to 0 for all actions
    actions['score_diff'] = 0

    # For each action resulting in a goal, increase all consequent actions of the scoring team with 1 and decrease all
    # consequent actions of the conceding team with 1
    for index, goal in goal_actions.iterrows():

        # Check if the action resulting in a goal is not the last action in the game
        if not goal.equals(actions.iloc[-1].drop(labels=['score_diff'])):

            # If the action is a goal, increase the score difference for all consequent actions of team of player
            # performing the action and decrease the score difference for actions of the opponent
            if goal['type_name'] == 'goal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] + 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] - 1, axis=1)

            # If the action is a goal, decrease the score difference for all consequent actions of team of player
            # performing the action and increase the score difference for actions of the opponent
            if goal['type_name'] == 'owngoal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] - 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] + 1, axis=1)

    return actions


def add_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Add a goal difference column to non-atomic actions.

    :param actions: a dataframe of actions
    :return: a dataframe of actions with an added column denoting the goal difference at the time of the actions
    """
    # Get all actions that result in a goal
    shotlike = ['shot', 'shot_penalty', 'shot_freekick']
    goal_actions = actions[(actions.type_name.isin(shotlike) & (actions.result_name == 'success')) | (
            actions.result_name == 'owngoal')]

    # Initialise the score difference to 0 for all actions
    actions['score_diff'] = 0

    # For each action resulting in a goal, increase all consequent actions of the scoring team with 1 and decrease all
    # consequent actions of the conceding team with 1
    for index, goal in goal_actions.iterrows():

        # Check if the action resulting in a goal is not the last action in the game
        if not goal.equals(actions.iloc[-1].drop(labels=['score_diff'])):

            # If the action is a goal, increase the score difference for all consequent actions of team of player
            # performing the action and decrease the score difference for actions of the opponent
            if goal['result_name'] == 'success':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] + 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] - 1, axis=1)

            # If the action is a goal, decrease the score difference for all consequent actions of team of player
            # performing the action and increase the score difference for actions of the opponent
            if goal['result_name'] == 'owngoal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] - 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] + 1, axis=1)

    return actions


def add_player_diff(actions: pd.DataFrame, game: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """
    Add a player difference column to actions in the given game.

    :param actions: a dataframe of actions
    :param game: a series of a game
    :param events: a dataframe of wyscout events
    :return: a dataframe of actions with an added column denoting the player difference at the time of the action
    """
    # Get all wyscout events in the game resulting in a red card (1701) or second yellow (1703)
    match_events = events[events['matchId'] == game.game_id]
    match_events['tags'] = match_events.apply(lambda x: [d['id'] for d in x['tags']], axis=1)
    red_cards = match_events[pd.DataFrame(match_events.tags.tolist()).isin([1701, 1703]).any(axis=1).values]

    # Get the red cards by team id
    red_cards_both_teams: Dict[int, List[Tuple]] = {}
    for red_card in red_cards.itertuples():
        if red_card[0] in red_cards_both_teams:
            red_cards_both_teams[red_card.teamId].append((red_card.matchPeriod, red_card.eventSec))
        else:
            red_cards_both_teams[red_card.teamId] = [(red_card.matchPeriod, red_card.eventSec)]

    # Initialise player difference to 0 for all actions
    actions['player_diff'] = 0

    # For all red cards (if there are any), decrease player difference for all consequent actions of team of
    # player conceding the red card and increase player difference for actions of the opponent
    if red_cards_both_teams:
        for team_id in red_cards_both_teams.keys():
            for red_card in red_cards_both_teams[team_id]:
                selector = (actions['period_id'] > wp[red_card[0]]) | ((actions['period_id'] == wp[red_card[0]]) &
                                                                       (actions['time_seconds'] > red_card[1]))
                actions.loc[selector, 'player_diff'] = actions[selector].apply(
                    lambda x: x['player_diff'] - 1 if (x['team_id'] == team_id) else x['player_diff'] + 1, axis=1)

    return actions


def convert_wyscout_to_h5():
    """
    Store wyscout events as dataframes.
    """
    wyscout_h5 = 'wyscout_data/wyscout.h5'
    with pd.HDFStore(wyscout_h5) as store:
        for competition in competition_index.itertuples():
            with open(os.path.join('wyscout_data', competition.db_events), 'rt', encoding='utf-8') as wm:
                events = pd.DataFrame(json.load(wm))
            # [:-5] removes the .json at the end
            store[competition.db_events[:-5]] = events


competition_index = pd.DataFrame(
    [
        {
            'competition_id': 524,
            'season_id': 181248,
            'season_name': '2017/2018',
            'db_matches': 'matches_Italy.json',
            'db_events': 'events_Italy.json',
        },
        {
            'competition_id': 364,
            'season_id': 181150,
            'season_name': '2017/2018',
            'db_matches': 'matches_England.json',
            'db_events': 'events_England.json',
        },
        {
            'competition_id': 795,
            'season_id': 181144,
            'season_name': '2017/2018',
            'db_matches': 'matches_Spain.json',
            'db_events': 'events_Spain.json',
        },
        {
            'competition_id': 412,
            'season_id': 181189,
            'season_name': '2017/2018',
            'db_matches': 'matches_France.json',
            'db_events': 'events_France.json',
        },
        {
            'competition_id': 426,
            'season_id': 181137,
            'season_name': '2017/2018',
            'db_matches': 'matches_Germany.json',
            'db_events': 'events_Germany.json',
        },
        {
            'competition_id': 102,
            'season_id': 9291,
            'season_name': '2016',
            'db_matches': 'matches_European_Championship.json',
            'db_events': 'events_European_Championship.json',
        },
        {
            'competition_id': 28,
            'season_id': 10078,
            'season_name': '2018',
            'db_matches': 'matches_World_Cup.json',
            'db_events': 'events_World_Cup.json',
        },
    ]
).set_index('competition_id')

all_competitions = [
    'Italian first division',
    'English first division',
    'Spanish first division',
    'French first division',
    'German first division',
    'European Championship',
    'World Cup'
]

train_competitions = [
    'German first division'
]

test_competitions = [
    'Italian first division',
    'English first division',
    'Spanish first division',
    'French first division',
    'European Championship',
    'World Cup'
]

all_competition_ids = [
    524,
    364,
    795,
    412,
    426,
    102,
    28
]

train_competition_ids = [
    426
]

test_competition_ids = [
    524,
    364,
    795,
    412,
    102,
    28
]

competitions_mapping = {
    'Italian first division': 'Italy',
    'English first division': 'England',
    'Spanish first division': 'Spain',
    'French first division': 'France',
    'German first division': 'Germany',
    'European Championship': 'European_Championship',
    'World Cup': 'World_cup'
}

teams_mapping = {
    'Inter': 'Internazionale',
    'Spal': 'SPAL',
    'Verona': 'Hellas Verona',
    'Bournemouth': 'AFC Bournemouth',
    'West Brom': 'West Bromwich Albion',
    'Huddersfield': 'Huddersfield Town',
    'Brighton': 'Brighton & Hove Albion',
    'Man United': 'Manchester United',
    'Newcastle': 'Newcastle United',
    'Man City': 'Manchester City',
    'Swansea': 'Swansea City',
    'Stoke': 'Stoke City',
    'Leicester': 'Leicester City',
    'Tottenham': 'Tottenham Hotspur',
    'West Ham': 'West Ham United',
    'Sociedad': 'Real Sociedad',
    'Ath Madrid': 'Atlético Madrid',
    'Espanol': 'Espanyol',
    'Ath Bilbao': 'Athletic Club',
    'La Coruna': 'Deportivo La Coruña',
    'Alaves': 'Deportivo Alavés',
    'Malaga': 'Málaga',
    'Betis': 'Real Betis',
    'Leganes': 'Leganés',
    'Celta': 'Celta de Vigo',
    'Paris SG': 'PSG',
    'Lyon': 'Olympique Lyonnais',
    'Marseille': 'Olympique Marseille',
    'Amiens': 'Amiens SC',
    'St Etienne': 'Saint-Étienne',
    'Bayern Munich': 'Bayern München',
    'Dortmund': 'Borussia Dortmund',
    'Hertha': 'Hertha BSC',
    'Ein Frankfurt': 'Eintracht Frankfurt',
    'Hannover': 'Hannover 96',
    'Leverkusen': 'Bayer Leverkusen',
    "M'gladbach": "Borussia M'gladbach",
    'Hamburg': 'Hamburger SV',
    'Mainz': 'Mainz 05',
    'FC Koln': 'Köln',
    'Ireland': 'Republic of Ireland',
    'South Korea': 'Korea Republic'}

column_order = ['0 to 10', '10 to 20', '20 to 30', '30 to 40', '40 to 50', '50 to 60', '60 to 70', '70 to 80',
                '80 to 90', '90+']

wp = {'1H': 1, '2H': 2, 'E1': 3, 'E2': 4, 'P': 5}
