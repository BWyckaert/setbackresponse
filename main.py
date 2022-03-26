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
from data.data_loader import load_and_convert_wyscout_data
from aggregates import get_competition_aggregates_and_store
from aggregates import competition_games_players
from setbacks import get_setbacks
from aggregates import convert_team_to_player_setback
from aggregates import extend_with_playerlist
from aggregates import get_player_aggregates_and_store
from train_vaep_classifier import compare_models
from rating_analysis import get_rating_progression
from rating_analysis import get_rating_progression_with_goal_diff
from rating_analysis import get_rating_analysis_and_store
from performance_comparison import compare_ingame_setbacks
import xp_model as xp

# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)


def get_time(period_id: int, time_seconds: float):
    if period_id == 1:
        base = 0
    elif period_id == 2:
        base = 45
    elif period_id == 3:
        base = 90
    elif period_id == 4:
        base = 105
    elif period_id == 5:
        base = 120

    m = int(base + time_seconds // 60)
    s = int(time_seconds % 60)
    return f"{m}m{s}s"


def get_game_length(game_id: int, player_games: pd.DataFrame):
    players_in_game = player_games[player_games.game_id == game_id]
    game_length = players_in_game[['minutes_played']].max()
    return game_length.at['minutes_played']


def get_player_minutes(game_id: int, player_id: int, player_games: pd.DataFrame):
    minutes_played = player_games[(player_games.player_id == player_id) &
                                  (player_games.game_id == game_id)].reset_index()
    return minutes_played.at[0, "minutes_played"]


def is_starter(game_id: int, player_games: pd.DataFrame, player_id: int):
    player_game = player_games[(player_games.game_id == game_id) &
                               (player_games.player_id == player_id)].reset_index()
    return player_game.at[0, "is_starter"]


def get_minutes_before_after(action: pd.Series, player_games: pd.DataFrame, actions_in_game: pd.DataFrame):
    game_length = get_game_length(action.game_id, player_games)
    player_minutes = get_player_minutes(action.game_id, action.player_id, player_games)
    group_by_period = actions_in_game[actions_in_game.period_id != 5].groupby("period_id")
    last_action_in_period = []
    for k, df in group_by_period:
        last_action_in_period.append(round(df.time_seconds.max() / 60))

    minutes_before = sum(last_action_in_period[:action.period_id - 1]) + round(action.time_seconds / 60)
    minutes_after = sum(last_action_in_period[action.period_id - 1:]) - round(action.time_seconds / 60)

    if is_starter(action.game_id, player_games, action.player_id):
        minutes_after -= game_length - player_minutes
    else:
        minutes_before -= game_length - player_minutes
    return [minutes_before, minutes_after]


if __name__ == '__main__':
    # load_and_convert_wyscout_data(atomic=False)
    # tvc.train_model(False)
    # competition_games_players()
    # compare_models()
    # compare_ingame_setbacks()

    train_competitions = ['German first division']
    test_competitions = list(set(utils.all_competitions) - set(train_competitions))

    # tvc.train_model(train_competitions=train_competitions, test_competitions=test_competitions, atomic=True,
    #                 learner="xgboost", print_eval=True, store_eval=False, compute_features_labels=False,
    #                 validation_size=0.25, tree_params=dict(n_estimators=100, max_depth=3))

    # tvc.train_model(train_competitions=train_competitions, test_competitions=test_competitions, atomic=True,
    #                 learner="catboost", print_eval=True, store_eval=False, compute_features_labels=False,
    #                 validation_size=0.25, tree_params=dict(eval_metric='BrierScore', loss_function='Logloss', iterations=100, depth=6))

    # tvc.train_model(train_competitions=train_competitions, test_competitions=test_competitions, atomic=False,
    #                 learner="lightgbm", print_eval=False, store_eval=False, compute_features_labels=False,
    #                 validation_size=0.25)

    # get_setbacks(competitions=utils.all_competitions, atomic=False)
    # get_competition_aggregates_and_store()
    # get_player_aggregates_and_store()

    atomic = False
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

    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_setbacks = setbackstore["teams_setbacks"]
        team_setbacks_over_matches = setbackstore["team_setbacks_over_matches"]


    # with pd.HDFStore(labels_h5) as labelstore:
    #     print(labelstore.keys())
    #     print(labelstore["game_1694391"])
    #
    # with pd.HDFStore(features_h5) as featurestore:
    #     print(featurestore.keys())
    #     print(featurestore["game_1694391"].shape)
    #
    # with pd.HDFStore(predictions_h5) as predictionsstore:
    #     print(predictionsstore.keys())
    #     print(predictionsstore["game_1694391"].shape)

    # for c in ['team_name_short', 'team_name']:
    #     teams[c] = teams[c].apply(
    #         lambda x: x.encode('raw_unicode_escape').decode('utf-8')
    #     )

    # games = games[games.competition_name != 'German first division']
    games = games[games.competition_name == 'World Cup']
    # games = games[games.competition_name == 'Italian first division']
    #
    all_actions = []
    first_period_last_action = []
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
        # all_actions.append(pd.concat([actions, values], axis=1))
        all_actions.append(utils.add_goal_diff(actions))
        break


    all_actions = utils.left_to_right(games, pd.concat(all_actions), _spadl)
    xp.get_data(games, all_actions)
    # print(round(all_actions, 4))
    # print(all_actions[all_actions['result_name'] == 'offside'])
    # all_actions = pd.concat(all_actions).reset_index(drop=True)
    # print(all_actions[all_actions.game_id == 2058012].round(5))
    # shift = all_actions.shift(-1)
    # print(shift[(all_actions.type_name == "shot_penalty")])
    # pc.vaep_comparison_in_game(player_setbacks.iloc[1], all_actions[all_actions.game_id == player_setbacks.iloc[1].game_id])
    # print(all_actions[(all_actions['type_name'] == 'foul') & (all_actions.shift(1)['type_name'] == 'foul')])

    # for index, action in all_actions[all_actions.type_name == "shot_penalty"].iterrows():
    #     print(all_actions[index-2: index+2]
    #           [["game_id", "original_event_id", "action_id", "period_id", "time_seconds", "team_id", "player_id",
    #             "start_x", "start_y", "end_x", "end_y", "type_id", "type_name", "result_name", "bodypart_id", "offensive_value",
    #             "defensive_value", "vaep_value"]])

    # get_rating_analysis_and_store(games, all_actions)
    # print(get_rating_progression(games, all_actions, False))
    # rating_prog_agg = get_rating_progression(games, all_actions, False)
    # count, mean, std = get_rating_progression_with_goal_diff(games, all_actions, False)
    #
    # print("Rating progression aggregates")
    # print()
    # print(rating_prog_agg)
    # print()
    # print("Count")
    # print()
    # print(count)
    # print()
    # print("Mean")
    # print()
    # print(mean)
    # print()
    # print("Standard deviation")
    # print()
    # print(std)

    #
    #
    # # all_actions = pd.concat(all_actions).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    # # all_actions = all_actions.astype({'player_id': 'int64'})
    # if atomic:
    #     all_actions = all_actions[["game_id", "nickname", "team_name_short", "period_id", "time_seconds", "x", "y",
    #                                "dx", "dy", "type_name", "offensive_value", "defensive_value", "vaep_value"]]
    # else:
    #     all_actions = all_actions[["game_id", "nickname", "team_name_short", "period_id", "time_seconds", "start_x",
    #                                "start_y", "end_x", "end_y", "type_name", "result_name", "offensive_value",
    #                                "defensive_value", "vaep_value"]]

    # print(all_actions[all_actions.game_id == 1694391].round(5))

    # print(all_actions[all_actions.game_id == 1694391].round(5))
    #
    #
    # all_actions["nb_actions"] = 1
    # d = {}  # Doesnt work when one player misses more than 1 penalty
    # for index, action in list(p.iterrows()):
    #     actions_in_game = all_actions[all_actions.game_id == action.game_id]
    #     player_actions = actions_in_game[actions_in_game.nickname == action.nickname]
    #     player_actions = player_actions[player_actions.period_id != 5]
    #
    #     before = [player_actions[player_actions.period_id < action.period_id],
    #               player_actions[(player_actions.period_id == action.period_id) &
    #                              (player_actions.time_seconds < action.time_seconds)]]
    #     before_setback = pd.concat(before)
    #
    #     after = [player_actions[(player_actions.period_id == action.period_id) &
    #                             (player_actions.time_seconds > action.time_seconds)],
    #              player_actions[player_actions.period_id > action.period_id]]
    #     after_setback = pd.concat(after)
    #
    #     ba = [before_setback, after_setback]
    #     minutes_before_after = get_minutes_before_after(action, player_games, actions_in_game)
    #
    #     for i, df in enumerate(ba):
    #         x = pd.DataFrame(df[["offensive_value", "defensive_value", "vaep_value", "nb_actions"]].sum()).T
    #         x["minutes_played"] = minutes_before_after[i]
    #         x["vaep_per_action"] = x.apply(lambda y: y["vaep_value"] / y["nb_actions"], axis=1)
    #         x["actions_per_minute"] = x.apply(lambda y: y["nb_actions"] / y["minutes_played"], axis=1)
    #         x["vaep_per_minute"] = x.apply(lambda y: y["vaep_value"] / y["minutes_played"], axis=1)
    #         x = x.astype({'nb_actions': 'int32'})
    #         ba[i] = x
    #
    #     d[action.nickname] = ba
    #
    # for key, value in d.items():
    #     print(f"Stats for {key} before the setback:")
    #     print(value[0])
    #     print()
    #     print(f"Stats for {key} after the setback:")
    #     print(value[1])
    #     print()
    #     print()
    #     print()

    # with pd.HDFStore(os.path.join("atomic_data", "spadl.h5")) as atomicstore:
    #     with pd.HDFStore(os.path.join("default_data", "spadl.h5")) as spadlstore:
    # atomicstore["games"] = spadlstore["games"]
    # atomicstore["competitions"] = spadlstore["competitions"]
    # atomicstore["players"] = spadlstore["players"]
    # atomicstore["player_games"] = spadlstore["player_games"]
    # atomicstore["teams"] = spadlstore["teams"]
    # for game in tqdm(spadlstore["games"].itertuples(), desc="Converting to atomic SPADL: "):
    #     actions = spadlstore[f"actions/game_{game.game_id}"]
    #     atomicstore[f"actions/game_{game.game_id}"] = aspadl.convert_to_atomic(actions)
