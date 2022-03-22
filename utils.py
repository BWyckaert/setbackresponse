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
