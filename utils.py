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
    group_by_period = actions.groupby("period_id")
    last_action_in_period = []
    for _, period in group_by_period:
        last_action_in_period.append(period.time_seconds.max())

    actions['total_seconds'] = actions.apply(
        lambda x: x['time_seconds'] + sum(last_action_in_period[: x['period_id'] - 1]), axis=1)

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

column_order = ['0 to 5', '5 to 10', '10 to 15', '15 to 20', '20 to 25', '25 to 30', '30 to 35', '35 to 40', '40 to 45',
                '45 to 50', '50 to 55', '55 to 60', '60 to 65', '65 to 70', '70 to 75', '75 to 80', '80 to 85',
                '85 to 90', '90 to 95', '95 to 100', '100 to 105', '105 to 110', '110 to 115', '115 to 120',
                '120 to 125', '125 to 130', '130 to 135']
