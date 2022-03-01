import os
import pandas as pd
import numpy as np
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl

from typing import List
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
            games.itertuples()
        ])


def get_score(game: pd.DataFrame, actions: pd.DataFrame, setback: pd.Series, atomic: bool) -> str:
    """
    Calculates the score in a game just before the given setback occurs

    :param game: the game for which the score should be found
    :param actions: all the actions in the game
    :param setback: the setback which defines the time in seconds when the score should be calculated
    :param atomic: boolean flag indicating whether or not the actions should be atomic
    :return: a string containing the score of the game just before the given setback occurs
    """
    before_setback = actions[(actions.period_id < setback.period_id) |
                             ((actions.period_id == setback.period_id) &
                              (actions.time_seconds < setback.time_seconds))]

    homescore = 0
    awayscore = 0

    if not atomic:
        shotlike = {"shot", "shot_penalty", "shot_freekick"}
        bs_goal_actions = before_setback[
            (before_setback.type_name.isin(shotlike) & (before_setback.result_name == "success")) | (
                    before_setback.result_name == "owngoal")]
        for action in bs_goal_actions.itertuples():
            if action.result_name == "success":
                if action.team_id == game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
            if action.result_name == "owngoal":
                if action.team_id != game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
    else:
        goal_like = {"goal", "owngoal"}
        bs_goal_actions = before_setback[before_setback.type_name.isin(goal_like)]
        for action in bs_goal_actions.itertuples():
            if action.type_name == "goal":
                if action.team_id == game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
            if action.type_name == "owngoal":
                if action.team_id != game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
    score = str(homescore) + " - " + str(awayscore)
    return score


def get_game_details(setback: pd.Series, games: pd.DataFrame) -> (pd.Series, bool, str):
    """
    Gets some game details for the game in which the given setback occurs, namely the game itself, whether or not
    the player who suffers the setback plays at home and who the opponent is

    :param setback: the setback for which the game details should be returned
    :param games: all the possible games
    :return: a series containing the game in which the setback occurs, a boolean home indicating whether the player who
    suffers the setback plays at home and a string representing the opponent
    """
    game = games[games.game_id == setback.game_id].iloc[0]
    if setback.team_id == game.home_team_id:
        home = True
        opponent = game.away_team_name_short
    else:
        home = False
        opponent = game.home_team_name_short
    return game, home, opponent


def get_missed_penalties(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    missed_penalties = actions[(actions.type_name == "shot_penalty") & (~(actions.period_id == 5))]

    if not atomic:
        missed_penalties = missed_penalties[missed_penalties.result_name == "fail"]
    else:
        for index, action in missed_penalties.iterrows():
            if actions.iloc[index + 1].type_name == "goal":
                missed_penalties.drop(index, inplace=True)

    mp_setbacks = []
    for mp in missed_penalties.itertuples():
        game, home, opponent = get_game_details(mp, games)
        score = get_score(game, actions[actions.game_id == game.game_id], mp, atomic)

        mp_setbacks.append(
            pd.DataFrame(data=np.array(
                [[mp.nickname, mp.player_id, mp.birth_date, mp.team_name_short, opponent, mp.game_id, home,
                  "missed penalty", mp.period_id, mp.time_seconds, score]]),
                columns=["player", "player_id", "birth_date", "player_team", "opponent_team", "game_id",
                         "home", "setback_type", "period_id", "time_seconds", "score"]))

    mp_setbacks = pd.concat(mp_setbacks).reset_index(drop=True)
    return mp_setbacks


def get_missed_shots(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    missed_shot = actions[actions.type_name == "shot"]

    if not atomic:
        missed_shot = missed_shot[missed_shot.result_name == "fail"]

        missed_kicks = missed_shot[missed_shot.bodypart_name == "foot"]
        missed_headers = missed_shot[missed_shot.bodypart_name == "head/other"]
        missed_kicks = missed_kicks[pow(105 - missed_kicks.start_x, 2) + pow(34 - missed_kicks.start_y, 2) <= 121]
        missed_headers = missed_headers[
            pow(105 - missed_headers.start_x, 2) + pow(34 - missed_headers.start_y, 2) <= 25]
        missed_shots = pd.concat([missed_kicks, missed_headers])
    else:
        for index, action in missed_shot.iterrows():
            if actions.iloc[index + 1].type_name == "goal":
                missed_shot.drop(index, inplace=True)

        missed_kicks = missed_shot[missed_shot.bodypart_name == "foot"]
        missed_headers = missed_shot[missed_shot.bodypart_name == "head/other"]
        missed_kicks = missed_kicks[pow(105 - missed_kicks.x, 2) + pow(34 - missed_kicks.y, 2) <= 121]
        missed_headers = missed_headers[pow(105 - missed_headers.x, 2) + pow(34 - missed_headers.y, 2) <= 25]
        missed_shots = pd.concat([missed_kicks, missed_headers])

    ms_setbacks = []
    for ms in missed_shots.itertuples():
        game, home, opponent = get_game_details(ms, games)
        score = get_score(game, actions[actions.game_id == game.game_id], ms, atomic)

        ms_setbacks.append(
            pd.DataFrame(data=np.array(
                [[ms.nickname, ms.player_id, ms.birth_date, ms.team_name_short, opponent, ms.game_id, home,
                  "missed shot", ms.period_id, ms.time_seconds, score]]),
                columns=["player", "player_id", "birth_date", "player_team", "opponent_team", "game_id",
                         "home", "setback_type", "period_id", "time_seconds", "score"]))

    ms_setbacks = pd.concat(ms_setbacks).reset_index(drop=True)
    return ms_setbacks


def get_game_details_gc(goal: pd.Series, games: pd.DataFrame, owngoal: bool) -> (pd.Series, bool, str):
    game = games[games.game_id == goal.game_id].iloc[0]
    if ((goal.team_id == game.home_team_id) and not owngoal) or ((goal.team_id != game.home_team_id) and owngoal):
        home = False
        gc_team = game.away_team_name_short
    else:
        home = True
        gc_team = game.home_team_name_short

    return game, home, gc_team


def get_goal_conceded(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    shotlike = {"shot", "shot_penalty", "shot_freekick"}
    if not atomic:
        goals = actions[((actions.type_name.isin(shotlike)) & (actions.result_name == "success")) | (
                actions.result_name == "owngoal")]
    else:
        goals = actions[(actions.type_name == "goal") | (actions.type_name == "owngoal")]

    gc_setbacks = []
    for goal in goals.itertuples():
        owngoal = goal.result_name == "owngoal" if not atomic else goal.type_name == "owngoal"
        game, home, gc_team = get_game_details_gc(goal, games, owngoal)
        score = get_score(game, actions[actions.game_id == game.game_id], goal, atomic)

        gc_setbacks.append(
            pd.DataFrame(data=np.array(
                [[gc_team, goal.team_name_short, goal.game_id, home, "goal conceded", goal.period_id, goal.time_seconds,
                  score]]
            ), columns=["team", "opponent", "game_id", "home", "setback_type", "period_id", "time_seconds", "score"]))

    gc_setbacks = pd.concat(gc_setbacks).reset_index(drop=True)
    return gc_setbacks


def foul_leading_to_goal(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    freekick_like = {"shot_penalty", "freekick", "freekick_crossed", "freekick_short",
                     "shot_freekick"}  # atomic (freekick) and default (others) (shot_penalty both) combined
    freekicks = actions[actions.type_name.isin(freekick_like) & (actions.shift(1).type_name == "foul")]
    if not atomic:
        freekicks = freekicks[freekicks.start_x > 75]
    else:
        freekicks = freekicks[freekicks.x > 75]

    fltg_setbacks = []
    for index, freekick in freekicks.iterrows():
        ca = actions[(actions.period_id == freekick.period_id) & (actions.time_seconds >= freekick.time_seconds) & (
                    actions.time_seconds < (freekick.time_seconds + 10))]  # ca = consecutive actions
        if not atomic:
            shotlike = {"shot", "shot_penalty", "shot_freekick"}
            ca = ca[
                ((ca.type_name.isin(shotlike)) & (ca.result_name == "success") & (ca.team_id == freekick.team_id)) | (
                        (ca.result_name == "owngoal") & ~(ca.team_id == freekick.team_id))]
        else:
            ca = ca[((ca.type_name == "goal") & (ca.team_id == freekick.team_id)) | (
                    (ca.type_name == "owngoal") & ~(ca.team_id == freekick.team_id))]

        if (not ca.empty) and (actions.iloc[index - 1].type_name == "foul"):
            foul = actions.iloc[index - 1]
            game, home, opponent = get_game_details(foul, games)
            score = get_score(game, actions[actions.game_id == game.game_id], foul, atomic)

            fltg_setbacks.append(
                pd.DataFrame(data=np.array(
                    [[foul.nickname, foul.player_id, foul.birth_date, foul.team_name_short, opponent, foul.game_id,
                      home,
                      "foul leading to goal", foul.period_id, foul.time_seconds, score]]
                ), columns=["player", "player_id", "birth_date", "player_team", "opponent_team", "game_id",
                            "home", "setback_type", "period_id", "time_seconds", "score"])
            )

    fltg_setbacks = pd.concat(fltg_setbacks).reset_index(drop=True)
    return fltg_setbacks


def get_setbacks(competitions: List[str], atomic=True) -> pd.DataFrame:
    if atomic:
        _spadl = aspadl
        datafolder = "atomic_data"
    else:
        _spadl = spadl
        datafolder = "default_data"

    spadl_h5 = os.path.join(datafolder, "spadl.h5")

    with pd.HDFStore(spadl_h5) as spadlstore:
        games = (
            spadlstore["games"]
                .merge(spadlstore["competitions"], how='left')
                .merge(spadlstore["teams"].add_prefix('home_'), how='left')
                .merge(spadlstore["teams"].add_prefix('away_'), how='left'))
        players = spadlstore["players"]
        teams = spadlstore["teams"]

    games = pd.concat([games[games.competition_name == competition]
                       for competition in competitions])

    all_actions = []
    with pd.HDFStore(spadl_h5) as spadlstore:
        for game_id in tqdm(games.game_id, "Collecting all actions:"):
            actions = spadlstore[f"actions/game_{game_id}"]
            actions = (
                _spadl.add_names(actions)
                    .merge(players, how="left")
                    .merge(teams, how="left")
                    .sort_values(["game_id", "period_id", "action_id"])
            )
            all_actions.append(actions)

    all_actions = pd.concat(all_actions).reset_index()
    all_actions = left_to_right(games, all_actions, _spadl)

    player_setbacks = []
    # player_setbacks.append(get_missed_penalties(games, all_actions, atomic))
    # player_setbacks.append(get_missed_shots(games, all_actions, atomic))
    player_setbacks.append(foul_leading_to_goal(games, all_actions, atomic))

    # team_setbacks = []
    # team_setbacks.append(get_goal_conceded(games, all_actions, atomic))

    print()
    print(pd.concat(player_setbacks))
    # print()
    # print()
    # print(pd.concat(team_setbacks))
