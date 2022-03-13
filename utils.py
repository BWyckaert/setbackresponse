import pandas as pd

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

competition_to_odds = {
    'Italian first division': 'Italy',
    'English first division': 'England',
    'Spanish first division': 'Spain',
    'French first division': 'France',
    'German first division': 'Germany',
    'European Championship': 'European_Championship',
    'World Cup': 'World_cup'
}
