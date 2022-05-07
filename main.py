import os
import warnings

import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl

from tqdm import tqdm

import utils

# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)


if __name__ == '__main__':

    atomic = False
    root = os.path.join(os.getcwd(), 'wyscout_data')
    if atomic:
        _spadl = aspadl
        datafolder = "atomic_data"
    else:
        _spadl = spadl
        datafolder = "default_data"

    spadl_h5 = os.path.join(datafolder, "spadl.h5")
    setbacks_h5 = os.path.join('setback_data', "setbacks.h5")
    predictions_h5 = os.path.join(datafolder, "predictions.h5")
    features_h5 = os.path.join(datafolder, 'features.h5')
    labels_h5 = os.path.join(datafolder, 'labels.h5')

    with pd.HDFStore(setbacks_h5) as store:
        team_as_player_setbacks = store['team_as_player_setbacks']

    with pd.HDFStore(spadl_h5) as spadlstore:
        games = (
            spadlstore["games"]
                .merge(spadlstore["competitions"], how='left')
                .merge(spadlstore["teams"].add_prefix('home_'), how='left')
                .merge(spadlstore["teams"].add_prefix('away_'), how='left')
        )
        competitions = spadlstore["competitions"]
        players = spadlstore["players"]
        teams = spadlstore["teams"]
        player_games = spadlstore['player_games'].merge(spadlstore['teams'], how='left')

    print(player_games.head())

    games = games[games['competition_name'].isin(utils.test_competitions)]
    team_as_player_setbacks = team_as_player_setbacks[team_as_player_setbacks['game_id'].isin(games['game_id'].tolist())]

    comparisons_h5 = 'setback_data/setbacks.h5'
    with pd.HDFStore(comparisons_h5) as store:
        player_setbacks = store['player_setbacks']
        team_setbacks = store['team_setbacks']
        team_as_player_setbacks = store['team_as_player_setbacks']
        team_setbacks_over_matches = store['team_setbacks_over_matches']
        # print(player_setbacks.head())
        # print(player_setbacks.tail())
        # print(team_setbacks.head())
        # print(team_setbacks.tail())
        # print(team_as_player_setbacks.head())
        # print(team_as_player_setbacks.tail())
        # print(team_setbacks_over_matches.head())
        # print(team_setbacks_over_matches.tail())

    # for c in ['team_name_short', 'team_name']:
    #     teams[c] = teams[c].apply(
    #         lambda x: x.encode('raw_unicode_escape').decode('utf-8')
    #     )

    # games = games[games.competition_name != 'German first division']
    games = games[games.competition_name == 'World Cup']
    # games = games[games.competition_name == 'Italian first division']
    all_actions = []
    for game in tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        actions = actions[actions['period_id'] != 5]
        actions = (
            _spadl.add_names(actions)
                .merge(players, how="left")
                .merge(teams, how="left")
                .sort_values(["game_id", "period_id", "action_id"])
                .reset_index(drop=True)
        )
        # values = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
        actions = utils.add_total_seconds_to_game(actions)
        # all_actions.append(pd.concat([actions, values], axis=1))
        # actions = utils.add_player_diff(actions, game, all_events)
        # actions = utils.add_goal_diff(actions)
        all_actions.append(actions)

    all_actions = pd.concat(all_actions).reset_index(drop=True)
    # all_actions = utils.left_to_right(games, pd.concat(all_actions), _spadl)
    # print(all_actions.head())
    print(all_actions[all_actions['type_name'] == 'dribble'])
    # # get_rating_analysis_and_store(games, all_actions)
    # store_rating_progression_per_player(all_actions, players, games, player_games)
    # print(round(all_actions, 4))
    # xp.compute_features_and_labels(games, all_actions, ['German first division'])
    # xp.train_model()


    # with pd.HDFStore(os.path.join("atomic_data", "spadl.h5")) as atomicstore:
    #     with pd.HDFStore(os.path.join("default_data", "spadl.h5")) as spadlstore:
    #         # atomicstore["games"] = spadlstore["games"]
    #         # atomicstore["competitions"] = spadlstore["competitions"]
    #         # atomicstore["players"] = spadlstore["players"]
    #         # atomicstore["player_games"] = spadlstore["player_games"]
    #         # atomicstore["teams"] = spadlstore["teams"]
    #         for game in tqdm(list(spadlstore["games"].itertuples()), desc="Converting to atomic SPADL: "):
    #             actions = spadlstore[f"actions/game_{game.game_id}"]
    #             atomicstore[f"actions/game_{game.game_id}"] = aspadl.convert_to_atomic(actions)
