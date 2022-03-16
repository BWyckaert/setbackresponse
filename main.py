import json
import os
import warnings
from io import BytesIO

import pandas as pd
import socceraction
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import train_vaep_classifier as tvc

from tqdm import tqdm

import utils
from data.data_loader import load_and_convert_wyscout_data
from aggregates import get_competition_aggregates_and_store_to_excel
from aggregates import competition_games_players
from setbacks import get_setbacks
from aggregates import convert_team_to_player_setback

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
    # load_and_convert_wyscout_data(False)
    # tvc.train_model(False)
    # get_competition_aggregates_and_store_to_excel()
    # competition_games_players()

    all_competitions = [
        'Italian first division',
        'English first division',
        'Spanish first division',
        'French first division',
        'German first division',
        'European Championship',
        'World Cup'
    ]

    get_setbacks(all_competitions, False)

    # root = os.path.join(os.getcwd(), 'wyscout_data')
    # with open(os.path.join(root, "matches_Germany.json"), 'rt',
    #           encoding='unicode_escape') as wm:
    #     wyscout_matches = pd.DataFrame(json.load(wm))['label']
    # print(wyscout_matches)

    # test = pd.read_csv('betting_data/odds_World_cup.csv')
    # test['Date'] = pd.to_datetime(test['Date'], yearfirst=True, infer_datetime_format=True).dt.date


    atomic = False
    if atomic:
        _spadl = aspadl
        datafolder = "atomic_data"
    else:
        _spadl = spadl
        datafolder = "default_data"

    spadl_h5 = os.path.join(datafolder, "spadl.h5")
    predictions_h5 = os.path.join(datafolder, "predictions.h5")

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

    setbacks_h5 = os.path.join(datafolder, "setbacks.h5")
    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_setbacks = setbackstore["teams_setbacks"]
        team_setbacks_over_matches = setbackstore["team_setbacks_over_matches"]

    # print(type(games.game_date.dt.date.iloc[0]))
    # print(games.game_date.dt.date.iloc[0])
    # print(team_setbacks_over_matches)
    # print(games)

    # odds = []
    # for competition in list(utils.competition_to_odds.values()):
    #     odds.append(pd.read_csv('betting_data/odds_{}.csv'.format(competition)))
    # odds = pd.concat(odds).reset_index(drop=True)
    # odds['Date'] = pd.to_datetime(odds['Date'], yearfirst=True, infer_datetime_format=True).dt.date
    # odds = odds.replace({'HomeTeam': utils.teams_mapping, 'AwayTeam': utils.teams_mapping})
    # odds = odds.rename(
    #     columns={'HomeTeam': 'home_team_name_short', 'AwayTeam': 'away_team_name_short', 'Date': 'game_date'})
    # games['game_date'] = games.game_date.dt.date
    #
    # games_odds = games.merge(odds, on=['home_team_name_short', 'away_team_name_short', 'game_date'])
    # print(games_odds.head())
    #
    # games_odds['margin'] = games_odds.apply(
    #     lambda x: (1 / x.B365H) + (1 / x.B365D) + (1 / x.B365A), axis=1)
    # odds_columns = ['B365H', 'B365D', 'B365A']
    # for column in odds_columns:
    #     games_odds[column] = games_odds.apply(
    #         lambda x: round((1 / x[column]) / x.margin, 2), axis=1)
    #
    # print(games_odds.head())


    # print(team_setbacks.head())
    # print(team_setbacks_over_matches.head())
    # print(player_setbacks.head())

    # for c in ['team_name_short', 'team_name']:
    #     teams[c] = teams[c].apply(
    #         lambda x: x.encode('raw_unicode_escape').decode('utf-8')
    #     )
    # print(teams)
    # games = games[games.competition_name == "Italian first division"]
    # #
    # all_actions = []
    # for game in tqdm(list(games.itertuples()), desc="Rating actions"):
    #     actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
    #     actions = (
    #         _spadl.add_names(actions)
    #             .merge(players, how="left")
    #             .merge(teams, how="left")
    #             .sort_values(["game_id", "period_id", "action_id"])
    #             .reset_index(drop=True)
    #     )
    #     all_actions.append(actions)
    #
    # all_actions = pd.concat(all_actions).reset_index(drop=True)
    # print(convert_team_to_player_setback(team_setbacks.iloc[0:10], player_games, all_actions, players, teams))

    #     values = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
    #     all_actions.append(pd.concat([actions, values], axis=1))
    # all_actions = pd.concat(all_actions).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    # all_actions = all_actions.astype({'player_id': 'int64'})
    # if atomic:
    #     all_actions = all_actions[["game_id", "action_id", "period_id", "player_id", "time_seconds", "type_name",
    #                                "nickname", "team_name_short", "offensive_value", "defensive_value", "vaep_value"]]
    # else:
    #     all_actions = all_actions[["game_id", "action_id", "period_id", "player_id", "time_seconds", "type_name",
    #                                "result_name", "nickname", "team_name_short", "offensive_value", "defensive_value",
    #                                "vaep_value"]]
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
    #         atomicstore["games"] = games[['game_id', 'competition_id', 'season_id', 'game_date', 'game_day', 'home_team_id', 'away_team_id', 'label']]
    #         spadlstore["games"] = games[['game_id', 'competition_id', 'season_id', 'game_date', 'game_day', 'home_team_id', 'away_team_id', 'label']]
            # atomicstore["games"] = spadlstore["games"]
            # atomicstore["competitions"] = spadlstore["competitions"]
            # atomicstore["players"] = spadlstore["players"]
            # atomicstore["player_games"] = spadlstore["player_games"]
            # atomicstore["teams"] = spadlstore["teams"]
            # for game in tqdm(spadlstore["games"].itertuples(), desc="Converting to atomic SPADL: "):
            #     actions = spadlstore[f"actions/game_{game.game_id}"]
            #     atomicstore[f"actions/game_{game.game_id}"] = aspadl.convert_to_atomic(actions)
