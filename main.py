import os
import warnings
import pandas as pd
import socceraction
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import train_vaep_classifier as tvc

from tqdm import tqdm
from data.data_loader import load_and_convert_wyscout_data
from aggregates import get_action_aggregates_and_store_to_excel
from setbacks import get_setbacks

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
    # get_action_aggregates_and_store_to_excel()
    all_competitions = [
        # 'Italian first division',
        # 'English first division',
        # 'Spanish first division',
        # 'French first division',
        # 'German first division',
        # 'European Championship',
        'World Cup'
    ]

    # get_setbacks(all_competitions, False)
    #
    atomic = True
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
                .merge(spadlstore["teams"].add_prefix('away_'), how='left'))
        competitions = spadlstore["competitions"]
        players = spadlstore["players"]
        teams = spadlstore["teams"]
        player_games = spadlstore["player_games"]


    games = games[games.competition_name == "World Cup"]

    all_actions = []
    for game in tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{2057960}")
        actions = (
            _spadl.add_names(actions)
                .merge(players, how="left")
                .merge(teams, how="left")
                .sort_values(["game_id", "period_id", "action_id"])
                .reset_index(drop=True)
        )
        # all_actions.append(actions[~(actions.shift(-1).team_id == actions.team_id)])
        # all_actions.append(actions[actions.type_name == "dribble"])
        # print()
        print(actions)
        # print()
        break

    # print(pd.concat(all_actions).reset_index(drop=True))


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
