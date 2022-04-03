import json
import os
import warnings
from io import BytesIO

import pandas as pd
import numpy as np
import socceraction
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import train_vaep_classifier as tvc
import rating_analysis as ra
import performance_comparison as pc

from tqdm import tqdm

import utils
from dataloader import load_and_convert_wyscout_data
from aggregates import get_competition_aggregates_and_store
from aggregates import competition_games_players
from setbacks import get_setbacks
from aggregates import extend_with_playerlist
from aggregates import get_player_aggregates_and_store
from train_vaep_classifier import compare_models
from rating_analysis import get_rating_progression
from rating_analysis import get_rating_progression_with_goal_diff
from rating_analysis import get_rating_analysis_and_store
from rating_analysis import store_rating_progression_per_player
from performance_comparison import compare_ingame_setbacks
import expected_passing.xp_model as xp
from performance_comparison import get_average_response
import xgboost

# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)


if __name__ == '__main__':
    # load_and_convert_wyscout_data(atomic=False)
    # tvc.train_model(False)
    # competition_games_players()
    # compare_models()
    # get_average_response()
    # compare_ingame_setbacks()
    # utils.convert_wyscout_to_h5()
    # data_h5 = "expected_passing/data.h5"
    #
    # filters = ['total_seconds', 'player_diff', 'score_diff']
    # with pd.HDFStore(data_h5) as datastore:
    #     X_test = datastore["X_test"]
    #     # X_train = datastore["X_train"]
    #     y_test = datastore["y_test"]
    #     # y_train = datastore["y_train"]
    # # xp.plot_calibration(X_test=X_test, y_test=y_test, filters=filters, learner="xgboost")
    # xp.evaluate("xgboost", X_test, y_test, filters)
    # xp.store_predictions("xgboost", X_test, filters)
    # tvc.compare_models()
    # pc.compare_for_setback("missed shot")

    # avg_response_h5 = "results/avg_response.h5"
    # with pd.HDFStore(avg_response_h5) as store:
    #     for key in store.keys()[:5]:
    #         print(key)
    #         print()
    #         print(store[key])
    #         print()
    # player_rating_progression_h5 = "results/pr_progression.h5"
    # with pd.HDFStore(player_rating_progression_h5) as store:
    #     for key in store.keys()[:10]:
    #         print(key)
    #         print()
    #         print(store[key])
    #         print()

    # game_rating_progression_h5 = "results/gr_progression.h5"
    # with pd.HDFStore(game_rating_progression_h5) as store:
    #     rp_per_action = store["per_action"]
    #     rp_per_minute = store["per_minute"]
    #
    # print(rp_per_action)
    # print(rp_per_minute)
    #


    # tvc.train_model(train_competitions=train_competitions, test_competitions=test_competitions, atomic=False,
    #                 learner="lightgbm", print_eval=True, store_eval=False, compute_features_labels=False,
    #                 validation_size=0.25)

    # get_setbacks(competitions=utils.all_competitions, atomic=False)
    # get_competition_aggregates_and_store()
    # get_player_aggregates_and_store()

    atomic = True
    root = os.path.join(os.getcwd(), 'wyscout_data')
    if atomic:
        _spadl = aspadl
        datafolder = "atomic_data"
    else:
        _spadl = spadl
        datafolder = "default_data"

    spadl_h5 = os.path.join(datafolder, "spadl.h5")
    setbacks_h5 = os.path.join(datafolder, "setbacks.h5")
    predictions_h5 = os.path.join(datafolder, "predictions.h5")
    features_h5 = os.path.join(datafolder, 'features.h5')
    labels_h5 = os.path.join(datafolder, 'labels.h5')

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
        player_games = spadlstore["player_games"]

    print(players)

    # with pd.HDFStore(setbacks_h5) as setbackstore:
    #     player_setbacks = setbackstore["player_setbacks"]
    #     team_setbacks = setbackstore["teams_setbacks"]
    #     team_setbacks_over_matches = setbackstore["team_setbacks_over_matches"]
    #
    # all_events = []
    # for competition in utils.index.itertuples():
    #     with open(os.path.join(root, competition.db_events), 'rt', encoding='utf-8') as wm:
    #         all_events.append(pd.DataFrame(json.load(wm)))
    #
    # all_events = pd.concat(all_events)

    # for c in ['team_name_short', 'team_name']:
    #     teams[c] = teams[c].apply(
    #         lambda x: x.encode('raw_unicode_escape').decode('utf-8')
    #     )

    # games = games[games.competition_name != 'German first division']
    # # games = games[games.competition_name == 'World Cup']
    # # games = games[games.competition_name == 'Italian first division']
    # all_actions = []
    # for game in tqdm(list(games.itertuples()), desc="Rating actions"):
    #     actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
    #     actions = actions[actions['period_id'] != 5]
    #     actions = (
    #         _spadl.add_names(actions)
    #             .merge(players, how="left")
    #             .merge(teams, how="left")
    #             .sort_values(["game_id", "period_id", "action_id"])
    #             .reset_index(drop=True)
    #     )
    #     values = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
    #     actions = utils.add_total_seconds_to_game(actions)
    #     all_actions.append(pd.concat([actions, values], axis=1))
    #     # actions = utils.add_player_diff(actions, game, all_events)
    #     # actions = utils.add_goal_diff(actions)
    #     # all_actions.append(actions)
    #
    # all_actions = utils.left_to_right(games, pd.concat(all_actions), _spadl)
    # get_rating_analysis_and_store(games, all_actions)
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
