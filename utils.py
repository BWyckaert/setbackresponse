import json
import os
from typing import cast, Dict, Any, Tuple, List

import pandas as pd

from tqdm import tqdm


def left_to_right(games: pd.DataFrame, actions: pd.DataFrame, _spadl) -> pd.DataFrame:
    """
    Changes the given actions such that all actions are performed as if the player plays from left to right

    :param games: a dataframe with all the games from which the actions are taken
    :param actions: a dataframe with all the actions for which the direction of play should be changed
    :param _spadl:
    :return: a dataframe containing the same actions as in actions, but with the direction of play altered such that
    all actions are performed as if the player plays from left to right
    """
    return pd.concat(
        [
            _spadl.play_left_to_right(actions[actions.game_id == game.game_id], game.home_team_id) for game in
            tqdm(list(games.itertuples()), desc="Converting direction of play: ")
        ]).reset_index(drop=True)


def add_total_seconds_to_game(actions: pd.DataFrame) -> pd.DataFrame:
    if 'total_seconds' in actions.columns:
        return actions
    group_by_period = actions.groupby("period_id")
    last_action_in_period = []
    for _, period in group_by_period:
        last_action_in_period.append(period.time_seconds.max())

    actions['total_seconds'] = actions.apply(
        lambda x: x['time_seconds'] + sum(last_action_in_period[: int(x['period_id']) - 1]), axis=1)

    return actions


def add_total_seconds(actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    extended_actions = []
    for game_id in tqdm(list(games.game_id), desc="Adding total_seconds to actions: "):
        extended_actions.append(add_total_seconds_to_game(actions[actions['game_id'] == game_id]))

    return pd.concat(extended_actions).reset_index(drop=True)


def add_goal_diff_atomic(actions: pd.DataFrame) -> pd.DataFrame:
    goallike = ['goal', 'owngoal']
    goal_actions = actions[actions['type_name'].isin(goallike)]
    actions['score_diff'] = 0

    for index, goal in goal_actions.iterrows():
        if not goal.equals(actions.iloc[-1].drop(labels=['score_diff'])):
            if goal['type_name'] == 'goal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] + 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] - 1, axis=1)
            if goal['type_name'] == 'owngoal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] - 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] + 1, axis=1)

    return actions


def add_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    shotlike = ["shot", "shot_penalty", "shot_freekick"]
    goal_actions = actions[(actions.type_name.isin(shotlike) & (actions.result_name == "success")) | (
            actions.result_name == "owngoal")]
    actions['score_diff'] = 0

    for index, goal in goal_actions.iterrows():
        if not goal.equals(actions.iloc[-1].drop(labels=['score_diff'])):
            if goal['result_name'] == 'success':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] + 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] - 1, axis=1)
            if goal['result_name'] == 'owngoal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] - 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] + 1, axis=1)

    return actions


def add_player_diff(actions: pd.DataFrame, game: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    match_events = events[events['matchId'] == game.game_id]
    match_events['tags'] = match_events.apply(lambda x: [d['id'] for d in x['tags']], axis=1)
    red_cards = match_events[pd.DataFrame(match_events.tags.tolist()).isin([1701, 1703]).any(axis=1).values]

    red_cards_both_teams: Dict[int, List[Tuple]] = {}
    for red_card in red_cards.itertuples():
        if red_card[0] in red_cards_both_teams:
            red_cards_both_teams[red_card.teamId].append((red_card.matchPeriod, red_card.eventSec))
        else:
            red_cards_both_teams[red_card.teamId] = [(red_card.matchPeriod, red_card.eventSec)]

    actions['player_diff'] = 0
    if red_cards_both_teams:
        for team_id in red_cards_both_teams.keys():
            for red_card in red_cards_both_teams[team_id]:
                selector = (actions['period_id'] > wp[red_card[0]]) | ((actions['period_id'] == wp[red_card[0]]) &
                                                                       (actions['time_seconds'] > red_card[1]))
                actions.loc[selector, 'player_diff'] = actions[selector].apply(
                    lambda x: x['player_diff'] - 1 if (x['team_id'] == team_id) else x['player_diff'] + 1, axis=1)

    return actions


def convert_wyscout_to_h5():
    root = os.path.join(os.getcwd(), 'wyscout_data')
    wyscout_h5 = os.path.join('wyscout_data', 'wyscout.h5')
    with pd.HDFStore(wyscout_h5) as wyscoutstore:
        for competition in index.itertuples():
            with open(os.path.join(root, competition.db_events), 'rt', encoding='utf-8') as wm:
                events = pd.DataFrame(json.load(wm))
            wyscoutstore[competition.db_events[:-5]] = events


def convert_team_to_player_setback(team_setbacks: pd.DataFrame, player_games: pd.DataFrame, actions: pd.DataFrame,
                                   players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    players_by_id = players.set_index("player_id", drop=False)
    player_setbacks = []
    player_games = player_games.merge(teams[["team_id", "team_name_short"]], left_on="team_id", right_on="team_id")

    for team_setback in tqdm(list(team_setbacks.itertuples()), desc="Converting team to player setbacks:"):
        actions_in_game = actions[actions.game_id == team_setback.game_id]

        group_by_period = actions_in_game[actions_in_game.period_id != 5].groupby("period_id")
        last_action_in_period = []
        for _, period in group_by_period:
            last_action_in_period.append(round(period.time_seconds.max() / 60))
        minutes_before_current_period = sum(last_action_in_period[:team_setback.period_id - 1])
        minute_of_setback = round(team_setback.time_seconds / 60) + minutes_before_current_period

        players_in_game = player_games[
            (player_games.game_id == team_setback.game_id) & (player_games.team_name_short == team_setback.team)]
        players_on_field = []
        for player in players_in_game[players_in_game.is_starter].itertuples():
            if player.minutes_played > minute_of_setback:
                players_on_field.append(player)

        for player in players_in_game[~players_in_game.is_starter].itertuples():
            if (players_in_game.minutes_played.max() - player.minutes_played) < minute_of_setback:
                players_on_field.append(player)

        for player in players_on_field:
            player = players_by_id.loc[player.player_id]
            player_setbacks.append(pd.DataFrame(
                data={"player": [player.nickname], "player_id": [player.player_id], "birth_date": [player.birth_date],
                      "player_team": [team_setback.team], "opponent_team": [team_setback.opponent],
                      "game_id": [team_setback.game_id], "home": [team_setback.home], "setback_type": ["goal conceded"],
                      "period_id": [team_setback.period_id], "time_seconds": [team_setback.time_seconds],
                      "score:": [team_setback.score]}))

    player_setbacks = pd.concat(player_setbacks).reset_index(drop=True)
    return player_setbacks


index = pd.DataFrame(
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

competition_to_odds = {
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
