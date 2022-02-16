import os
import pandas as pd
import numpy as np
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl

from typing import List
from tqdm import tqdm


def get_score(game: pd.DataFrame, actions: pd.DataFrame, setback: pd.Series, atomic: bool) -> str:
    before = [actions[actions.period_id < setback.period_id],
              actions[(actions.period_id == setback.period_id) &
                      (actions.time_seconds < setback.time_seconds)]]
    before_setback = pd.concat(before)

    if not atomic:
        homescore = 0
        awayscore = 0
        for _, action in before_setback.iterrows():
            if (action.type_name == "shot" or action.type_name == "shot_penalty" or action.type_name == "shot_freekick") and action.result == "succes":
                if action.team_id == game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
    else:
        homescore = 0
        awayscore = 0
        for _, action in before_setback.iterrows():
            if action.type_name == "goal":
                if action.team_id == game.home_team_id:
                    homescore += 1
                else:
                    awayscore += 1
    score = str(homescore) + " - " + str(awayscore)
    return score


def get_missed_penalties(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    missed_penalties = actions[actions.type_name == "shot_penalty"]
    missed_penalties = missed_penalties[~(missed_penalties.period_id == 5)]
    if not atomic:
        missed_penalties = missed_penalties[missed_penalties.result_name == "success"]
    else:
        for index, action in missed_penalties.iterrows():
            if actions.iloc[index + 1].type_name == "goal":
                missed_penalties.drop(index, inplace=True)
    mp_setbacks = []
    for _, mp in missed_penalties.iterrows():
        game = games[games.game_id == mp.game_id].iloc[0]
        if mp.team_id == game.home_team_id:
            home = True
            opponent = game.away_team_name_short
        else:
            home = False
            opponent = game.home_team_name_short

        score = get_score(game, actions[actions.game_id == game.game_id], mp, atomic)

        mp_setbacks.append(
            pd.DataFrame(data=np.array(
                [[mp.nickname, mp.player_id, mp.birth_date, mp.team_name_short, opponent, mp.game_id, home,
                 "missed penalty", mp.period_id, mp.time_seconds, score]]),
                columns=["player", "player_id", "birth_date", "player_team", "opponent_team", "game_id",
                         "home", "setback_type", "period_id", "time_seconds", "score"]))

    mp_setbacks = pd.concat(mp_setbacks).reset_index()
    return mp_setbacks


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
    missed_penalties = get_missed_penalties(games, all_actions, atomic)
    print(missed_penalties)
