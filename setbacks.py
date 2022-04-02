import json
import os
import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import utils

from typing import List
from tqdm import tqdm


def get_score(game: pd.DataFrame, actions: pd.DataFrame, setback: pd.Series, atomic: bool) -> str:
    """
    Calculates the score in a game just before the given setback occurs

    :param game: the game for which the score should be found
    :param actions: all the actions in the game
    :param setback: the setback which defines the time in seconds when the score should be calculated
    :param atomic: boolean flag indicating whether or not the actions should be atomic
    :return: a string containing the score of the game just before the given setback occurs
    """
    # Select all actions in the game that occur before the given setback
    before_setback = actions[(actions.period_id < setback.period_id) |
                             ((actions.period_id == setback.period_id) &
                              (actions.time_seconds < setback.time_seconds))]

    homescore = 0
    awayscore = 0

    if not atomic:
        shotlike = {"shot", "shot_penalty", "shot_freekick"}
        # Select all actions in before_setback that result in a goal
        bs_goal_actions = before_setback[
            (before_setback.type_name.isin(shotlike) & (before_setback.result_name == "success")) | (
                    before_setback.result_name == "owngoal")]
        # Iterate over all goal actions and increment homescore and awayscore based on which team scores and whether it
        # is a normal goal or an owngoal
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
        # Select all actions in before_setback that result in a goal
        bs_goal_actions = before_setback[before_setback.type_name.isin(goal_like)]
        # Iterate over all goal actions and increment homescore and awayscore based on which team scores and whether it
        #         # is a normal goal or an owngoal
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
    # Select the game in which the setback occurs
    game = games[games.game_id == setback.game_id].iloc[0]
    # Retrieve the opponent of the player who suffers the setback and whether or not he plays at home or away
    if setback.team_id == game.home_team_id:
        home = True
        opponent = game.away_team_name_short
    else:
        home = False
        opponent = game.home_team_name_short
    return game, home, opponent


def get_missed_penalties(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Finds all missed penalties (not during penalty shootouts) in the given games and returns them in an appropriate
    dataframe

    :param games: the games for which the missed penalties must be found
    :param actions: all the actions in the given games
    :param atomic: boolean flag indicating whether or not the actions are atomic
    :return: a dataframe of all missed penalties in the given games
    """
    print("Finding missed penalties... ")
    print()

    # Select all penalties that do not occur during penalty shootouts
    missed_penalties = actions[(actions.type_name == "shot_penalty") & (~(actions.period_id == 5))]

    # Remove penalties which result in a goal
    for index, mp in missed_penalties.iterrows():
        fa = actions[(actions.game_id == mp.game_id) & (actions.period_id == mp.period_id) & (
                actions.time_seconds >= mp.time_seconds) & (
                             actions.time_seconds < (mp.time_seconds + 5))]  # fa = following actions
        if not atomic:
            shotlike = ["shot", "shot_penalty", "shot_freekick"]
            fa = fa[
                ((fa.type_name.isin(shotlike)) & (fa.result_name == "success") & (fa.team_id == mp.team_id)) | (
                        (fa.result_name == "owngoal") & ~(fa.team_id == mp.team_id))]
            if not fa.empty:
                missed_penalties.drop(index, inplace=True)

        else:
            fa = fa[((fa.type_name == "goal") & (fa.team_id == mp.team_id)) | (
                    (fa.type_name == "owngoal") & ~(fa.team_id == mp.team_id))]
            if not fa.empty:
                missed_penalties.drop(index, inplace=True)

    # Construct a dataframe with all missed penalties
    mp_setbacks = []
    for mp in missed_penalties.itertuples():
        game, home, opponent = get_game_details(mp, games)
        score = get_score(game, actions[actions.game_id == game.game_id], mp, atomic)

        mp_setbacks.append(pd.DataFrame(
            data={"player": [mp.nickname], "player_id": [mp.player_id], "birth_date": [mp.birth_date],
                  "player_team": [mp.team_name_short], "opponent_team": [opponent], "game_id": [mp.game_id],
                  "home": [home], "setback_type": ["missed penalty"], "period_id": [mp.period_id],
                  "time_seconds": [mp.time_seconds], "score:": [score]}))

    mp_setbacks = pd.concat(mp_setbacks).reset_index(drop=True)
    return mp_setbacks


def get_missed_shots(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    """
    Finds all shots that did not result in a goal that are tagged by Wyscout as an opportunity and that occur within
    a certain distance from goal (11m for shots, 5m for headers)

    :param games: the games for which the missed shots must be found
    :param actions: all the actions in the given games
    :param atomic: boolean flag indicating whether or not the actions are atomic
    :return: a dataframe of all the missed shots in the given games
    """
    print("Finding missed shots... ")
    print()

    # Select all shots in the given games
    shots = actions[actions.type_name == "shot"]
    shots = shots.merge(games[["game_id", "competition_id"]], left_on="game_id", right_on="game_id")
    # Group shots by their competition_id, as Wyscout stores events of one competition in one json file
    shots_grouped_by_competition_id = shots.groupby("competition_id")

    root = os.path.join(os.getcwd(), 'wyscout_data')

    shots = []
    # Select all shots that are tagged as an opportunity (id: 201) and inaccurate (id: 1802) by Wyscout annotators
    for competition_id, ms_by_competition in shots_grouped_by_competition_id:
        # Open Wyscout events of the competition represented by competition_id
        with open(os.path.join(root, utils.index.at[competition_id, 'db_events']), 'rt', encoding='utf-8') as we:
            events = pd.DataFrame(json.load(we))
        ms_by_competition = ms_by_competition.merge(events[['id', 'tags']], left_on='original_event_id', right_on='id')
        # Reformat tags from list of dicts to list
        ms_by_competition['tags'] = ms_by_competition.apply(lambda x: [d['id'] for d in x['tags']], axis=1)
        # Select all shots that are tagged as an opportunity
        opportunity = ms_by_competition[pd.DataFrame(ms_by_competition.tags.tolist()).isin([201]).any(axis=1).values]
        # Select all opportunities that are tagged as inaccurate
        inaccurate = opportunity[pd.DataFrame(opportunity.tags.tolist()).isin([1802]).any(axis=1).values]
        shots.append(inaccurate)

    shots = pd.concat(shots).reset_index(drop=True)

    # Select shots that do not result in a goal where the distance to goal is smaller than 11m for shots and 5m for
    # headers
    if not atomic:
        missed_shots = shots[shots.result_name == "fail"]

        missed_kicks = missed_shots[missed_shots.bodypart_name == "foot"]
        missed_headers = missed_shots[missed_shots.bodypart_name == "head/other"]
        missed_kicks = missed_kicks[pow(105 - missed_kicks.start_x, 2) + pow(34 - missed_kicks.start_y, 2) <= 121]
        missed_headers = missed_headers[
            pow(105 - missed_headers.start_x, 2) + pow(34 - missed_headers.start_y, 2) <= 25]
        missed_shots = pd.concat([missed_kicks, missed_headers])
    else:
        for index, action in shots.iterrows():
            if actions.iloc[index + 1].type_name == "goal":
                shots.drop(index, inplace=True)

        missed_shots = shots

        missed_kicks = missed_shots[missed_shots.bodypart_name == "foot"]
        missed_headers = missed_shots[missed_shots.bodypart_name == "head/other"]
        missed_kicks = missed_kicks[pow(105 - missed_kicks.x, 2) + pow(34 - missed_kicks.y, 2) <= 121]
        missed_headers = missed_headers[pow(105 - missed_headers.x, 2) + pow(34 - missed_headers.y, 2) <= 25]
        missed_shots = pd.concat([missed_kicks, missed_headers])

    # Construct a dataframe with all missed shots
    ms_setbacks = []
    for ms in missed_shots.itertuples():
        game, home, opponent = get_game_details(ms, games)
        score = get_score(game, actions[actions.game_id == game.game_id], ms, atomic)

        ms_setbacks.append(pd.DataFrame(
            data={"player": [ms.nickname], "player_id": [ms.player_id], "birth_date": [ms.birth_date],
                  "player_team": [ms.team_name_short], "opponent_team": [opponent], "game_id": [ms.game_id],
                  "home": [home], "setback_type": ["missed shot"], "period_id": [ms.period_id],
                  "time_seconds": [ms.time_seconds], "score:": [score]}))

    ms_setbacks = pd.concat(ms_setbacks).reset_index(drop=True)
    return ms_setbacks


def get_game_details_gc(goal: pd.Series, games: pd.DataFrame, owngoal: bool) -> (pd.Series, bool, str, str):
    game = games[games.game_id == goal.game_id].iloc[0]
    if ((goal.team_id == game.home_team_id) and not owngoal) or ((goal.team_id != game.home_team_id) and owngoal):
        home = False
        gc_team = game.away_team_name_short
        opponent = game.home_team_name_short
    else:
        home = True
        gc_team = game.home_team_name_short
        opponent = game.away_team_name_short

    return game, home, gc_team, opponent


def get_goal_conceded(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    print("Finding conceded goals... ")
    print()

    remove_penalty_shootouts = actions[actions.period_id != 5]

    if not atomic:
        shotlike = {"shot", "shot_penalty", "shot_freekick"}
        goals = remove_penalty_shootouts[((remove_penalty_shootouts.type_name.isin(shotlike)) &
                                          (remove_penalty_shootouts.result_name == "success")) |
                                         (remove_penalty_shootouts.result_name == "owngoal")]
    else:
        goals = remove_penalty_shootouts[(remove_penalty_shootouts.type_name == "goal") |
                                         (remove_penalty_shootouts.type_name == "owngoal")]

    gc_setbacks = []
    for goal in goals.itertuples():
        owngoal = goal.result_name == "owngoal" if not atomic else goal.type_name == "owngoal"
        game, home, gc_team, opponent = get_game_details_gc(goal, games, owngoal)
        score = get_score(game, actions[actions.game_id == game.game_id], goal, atomic)

        gc_setbacks.append(pd.DataFrame(
            data={"team": [gc_team], "opponent": [opponent], "game_id": [goal.game_id], "home": [home],
                  "setback_type": ["goal conceded"], "period_id": [goal.period_id], "time_seconds": [goal.time_seconds],
                  "score": [score]}))

    gc_setbacks = pd.concat(gc_setbacks).reset_index(drop=True)
    return gc_setbacks


def foul_leading_to_goal(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    print("Finding fouls leading to goals... ")
    print()

    freekick_like = {"shot_penalty", "freekick", "freekick_crossed", "freekick_short",
                     "shot_freekick"}  # atomic (freekick) and default (others) (shot_penalty both) combined
    freekicks = actions[actions.type_name.isin(freekick_like) & (actions.shift(1).type_name == "foul")]

    if not atomic:
        freekicks = freekicks[freekicks.start_x > 75]
    else:
        freekicks = freekicks[freekicks.x > 75]

    fltg_setbacks = []
    for index, freekick in freekicks.iterrows():
        fa = actions[(actions.game_id == freekick.game_id) & (actions.period_id == freekick.period_id) & (
                actions.time_seconds >= freekick.time_seconds) & (
                             actions.time_seconds < (freekick.time_seconds + 10))]  # fa = following actions

        if not atomic:
            shotlike = {"shot", "shot_penalty", "shot_freekick"}
            fa = fa[
                ((fa.type_name.isin(shotlike)) & (fa.result_name == "success") & (fa.team_id == freekick.team_id)) | (
                        (fa.result_name == "owngoal") & ~(fa.team_id == freekick.team_id))]
        else:
            fa = fa[((fa.type_name == "goal") & (fa.team_id == freekick.team_id)) | (
                    (fa.type_name == "owngoal") & ~(fa.team_id == freekick.team_id))]

        if (not fa.empty) and (actions.iloc[index - 1].type_name == "foul"):
            foul = actions.iloc[index - 1]
            game, home, opponent = get_game_details(foul, games)
            score = get_score(game, actions[actions.game_id == game.game_id], foul, atomic)

            fltg_setbacks.append(pd.DataFrame(
                data={"player": [foul.nickname], "player_id": [foul.player_id], "birth_date": [foul.birth_date],
                      "player_team": [foul.team_name_short], "opponent_team": [opponent], "game_id": [foul.game_id],
                      "home": [home], "setback_type": ["foul leading to goal"], "period_id": [foul.period_id],
                      "time_seconds": [foul.time_seconds], "score:": [score]}))

    fltg_setbacks = pd.concat(fltg_setbacks).reset_index(drop=True)
    return fltg_setbacks


def bad_pass_leading_to_goal(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    print("Finding bad pass leading to goal... ")
    print()

    if not atomic:
        shotlike = {"shot", "shot_penalty", "shot_freekick"}
        goals = actions[((actions.type_name.isin(shotlike)) & (actions.result_name == "success")) | (
                actions.result_name == "owngoal")]
    else:
        goals = actions[(actions.type_name == "goal") | (actions.type_name == "owngoal")]

    bpltg_setbacks = []
    for index, goal in goals.iterrows():
        pa = actions[(actions.game_id == goal.game_id) & (actions.period_id == goal.period_id) & (
                actions.time_seconds < goal.time_seconds) & (
                             actions.time_seconds > (goal.time_seconds - 15))]  # pa = previous actions

        if not atomic:
            bad_pass = pa[((pa.type_name == "pass") & (pa.result_name == "fail")) & (
                    (~(pa.team_id == goal.team_id) & (goal.result_name == "success")) | (
                    (pa.team_id == goal.team_id) & (goal.result_name == "owngoal")))]
            bad_pass = bad_pass[bad_pass.start_x < 65]

        else:
            bad_pass = pa[((pa.type_name == "pass") & (pa.shift(-1).type_name == "interception")) & (
                    (~(pa.team_id == goal.team_id) & (goal.type_name == "goal")) | (
                    (pa.team_id == goal.team_id) & (goal.type_name == "owngoal")))]
            bad_pass = bad_pass[bad_pass.x < 65]

        if not bad_pass.empty:
            bad_pass = bad_pass.iloc[-1]
            if pa[(pa.time_seconds > bad_pass.time_seconds) & (pa.team_id == bad_pass.team_id)].empty:
                continue
            game, home, opponent = get_game_details(bad_pass, games)
            score = get_score(game, actions[actions.game_id == game.game_id], bad_pass, atomic)

            bpltg_setbacks.append(pd.DataFrame(
                data={"player": [bad_pass.nickname], "player_id": [bad_pass.player_id],
                      "birth_date": [bad_pass.birth_date], "player_team": [bad_pass.team_name_short],
                      "opponent_team": [opponent], "game_id": [bad_pass.game_id], "home": [home],
                      "setback_type": ["bad pass leading to goal"], "period_id": [bad_pass.period_id],
                      "time_seconds": [bad_pass.time_seconds], "score:": [score]}))

    bpltg_setbacks = pd.concat(bpltg_setbacks).reset_index(drop=True)
    return bpltg_setbacks


def bad_consecutive_passes(games: pd.DataFrame, actions: pd.DataFrame, atomic: bool) -> pd.DataFrame:
    print("Finding bad consecutive passes... ")
    print()

    last_bad_pass_in_seq = []
    grouped_by_game = actions.groupby("game_id")
    for game_id, game_actions in grouped_by_game:
        grouped_by_player = game_actions.groupby("player_id")
        for player_id, player_actions in grouped_by_player:
            pass_actions = player_actions[
                (player_actions.type_name == "pass") & (
                        game_actions.shift(-1).period_id == game_actions.period_id)].reset_index(drop=True)
            if pass_actions.empty:
                continue
            if not atomic:
                # groups consecutive failed and successful passes
                grouped_by_success_failure = pass_actions.groupby(
                    [(pass_actions.result_name != pass_actions.shift(1).result_name).cumsum(), "period_id"])
            else:
                # add result_name column to dataframe to mimic not atomic dataframe
                next_actions = game_actions[
                    (game_actions.shift(1).type_name == "pass") & (game_actions.shift(1).player_id == player_id) & (
                            game_actions.shift(1).period_id == game_actions.period_id)].reset_index(drop=True)
                pass_actions["result_name"] = next_actions.apply(
                    lambda y: "success" if (y.type_name == "receival") else "fail", axis=1)

                grouped_by_success_failure = pass_actions.groupby(
                    [(pass_actions.result_name != pass_actions.shift(1).result_name).cumsum(), "period_id"])

            for _, sf in grouped_by_success_failure:
                sf = sf.reset_index(drop=True)
                # don't consider successful passes
                if sf.iloc[0].result_name == "success":
                    continue
                # only consider at least 3 bad consecutive passes
                if sf.shape[0] < 3:
                    continue
                # only consider 3 bad passes within 300 seconds of each other
                while sf.shape[0] >= 3:
                    if sf.at[2, "time_seconds"] - sf.at[0, "time_seconds"] < 300:
                        last_bad_pass_in_seq.append(sf.iloc[2].to_frame().T)
                        # print(sf)
                        break
                    else:
                        sf = sf.iloc[1:].reset_index(drop=True)

    last_bad_pass_in_seq = pd.concat(last_bad_pass_in_seq).reset_index(drop=True)

    bpis_setbacks = []
    for bad_pass in last_bad_pass_in_seq.itertuples():
        game, home, opponent = get_game_details(bad_pass, games)
        score = get_score(game, actions[actions.game_id == game.game_id], bad_pass, atomic)

        bpis_setbacks.append(pd.DataFrame(
            data={"player": [bad_pass.nickname], "player_id": [bad_pass.player_id],
                  "birth_date": [bad_pass.birth_date], "player_team": [bad_pass.team_name_short],
                  "opponent_team": [opponent], "game_id": [bad_pass.game_id], "home": [home],
                  "setback_type": ["bad consecutive passes"], "period_id": [bad_pass.period_id],
                  "time_seconds": [bad_pass.time_seconds], "score:": [score]}))

    bpis_setbacks = pd.concat(bpis_setbacks).reset_index(drop=True)
    return bpis_setbacks


def lost_game(game: pd.Series, team_id: int) -> bool:
    home = game.home_team_id == team_id
    score = game.label.split(", ")[1][0:5]
    if home:
        return int(score[0]) < int(score[4])
    else:
        return int(score[0]) > int(score[4])


def convert_to_fair_odds(games_odds: pd.DataFrame) -> pd.DataFrame:
    games_odds['margin'] = games_odds.apply(
        lambda x: (1 / x['B365H']) + (1 / x['B365D']) + (1 / x['B365A']), axis=1)
    odds_columns = ['B365H', 'B365D', 'B365A']
    for column in odds_columns:
        games_odds[column] = games_odds.apply(
            lambda x: round((1 / x[column]) / x['margin'], 2), axis=1
        )
    games_odds = games_odds.rename(columns={'B365H': 'home_win', 'B365D': 'draw', 'B365A': 'away_win'})
    return games_odds


def add_odds_to_games(games: pd.DataFrame) -> pd.DataFrame:
    odds = []
    for competition in list(utils.competitions_mapping.values()):
        odds.append(pd.read_csv('betting_data/odds_{}.csv'.format(competition)))
    odds = pd.concat(odds).reset_index(drop=True)
    odds['Date'] = pd.to_datetime(odds['Date'], yearfirst=True, infer_datetime_format=True).dt.date
    odds = odds.replace({'HomeTeam': utils.teams_mapping, 'AwayTeam': utils.teams_mapping})
    odds = odds[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']]
    odds = odds.rename(
        columns={'HomeTeam': 'home_team_name_short', 'AwayTeam': 'away_team_name_short', 'Date': 'game_date'})

    games['game_date'] = games.game_date.dt.date

    games_odds = games.merge(odds, on=['home_team_name_short', 'away_team_name_short', 'game_date'])
    games_odds = convert_to_fair_odds(games_odds)

    return games_odds


def consecutive_losses(games: pd.DataFrame) -> pd.DataFrame:
    print("Finding consecutive losses... ")
    print()

    games_odds = add_odds_to_games(games)

    cl_setbacks = []
    games_by_home_team = games_odds.groupby('home_team_id')
    games_by_away_team = games_odds.groupby('away_team_id')
    for team_id, games_for_team in games_by_home_team:
        cons_losses = []
        games_for_team = pd.concat([games_for_team, games_by_away_team.get_group(team_id)]).sort_values('game_date')
        games_for_team['lost_game'] = games_for_team.apply(lambda x: lost_game(x, team_id), axis=1)
        games_for_team['losing_chance'] = games_for_team.apply(
            lambda x: x['away_win'] if (x['home_team_id'] == team_id) else x['home_win'], axis=1)
        grouped_by_loss_wins = games_for_team.groupby(
            [(games_for_team['lost_game'] != games_for_team.shift(1)['lost_game']).cumsum()])

        for _, lw in grouped_by_loss_wins:
            lw = lw.reset_index(drop=True)
            # don't consider wins
            if not lw.iloc[0]['lost_game']:
                continue
            lw['cons_loss_chance'] = lw['losing_chance'].cumprod()
            for index, game in lw.iterrows():
                if game['cons_loss_chance'] < 0.30:
                    cons_losses.append(lw[:index + 1])
                    # break

        for cons_loss in cons_losses:
            last_loss = cons_loss.iloc[-1]
            team = last_loss.home_team_name_short if last_loss.home_team_id == team_id \
                else last_loss.away_team_name_short

            cl_setbacks.append(pd.DataFrame(
                data={"team": [team], "lost game(s)": [cons_loss['game_id'].tolist()],
                      "game_date_last_loss": [last_loss['game_date']], "competition": [last_loss['competition_name']],
                      "setback_type": ["consecutive losses"], "chance": [last_loss['cons_loss_chance']]}))

    cl_setbacks = pd.concat(cl_setbacks).reset_index(drop=True)
    return cl_setbacks


def get_setbacks(competitions: List[str], atomic=False) -> tuple:
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

    all_actions = pd.concat(all_actions)
    all_actions = utils.left_to_right(games, all_actions, _spadl)

    player_setbacks = [get_missed_penalties(games, all_actions, atomic), get_missed_shots(games, all_actions, atomic),
                       foul_leading_to_goal(games, all_actions, atomic),
                       bad_pass_leading_to_goal(games, all_actions, atomic),
                       bad_consecutive_passes(games, all_actions, atomic)]
    player_setbacks = pd.concat(player_setbacks).reset_index(drop=True)

    team_setbacks = [get_goal_conceded(games, all_actions, atomic)]
    team_setbacks = pd.concat(team_setbacks).reset_index(drop=True)

    team_setbacks_over_matches = consecutive_losses(games)

    setbacks_h5 = os.path.join(datafolder, "setbacks.h5")
    with pd.HDFStore(setbacks_h5) as setbackstore:
        setbackstore["player_setbacks"] = player_setbacks
        setbackstore["teams_setbacks"] = team_setbacks
        setbackstore["team_setbacks_over_matches"] = team_setbacks_over_matches

    return player_setbacks, team_setbacks, team_setbacks_over_matches
