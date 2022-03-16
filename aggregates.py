import os
import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import setbacks as sb
import numpy as np

from tqdm import tqdm


def _get_action_aggregates_in_competition(spadl_h5: str, competition_id: int, games: pd.DataFrame, _spadl,
                                          normalize) -> pd.DataFrame:
    competition_games = games[games.competition_id == competition_id]
    all_actions = []
    with pd.HDFStore(spadl_h5) as spadlstore:
        for game_id in competition_games.game_id:
            actions = spadlstore[f"actions/game_{game_id}"]
            actions = _spadl.add_names(actions)
            all_actions.append(actions)

    all_actions = pd.concat(all_actions).reset_index()
    action_counts = pd.Series.to_frame(
        all_actions["type_name"].value_counts(normalize=normalize, dropna=False)).T.sort_index(axis=1)
    action_counts["competition_id"] = competition_id
    action_counts.set_index("competition_id", inplace=True)
    return action_counts


def _get_action_aggregates(atomic=True, normalize=False) -> pd.DataFrame:
    if atomic:
        _spadl = aspadl
        datafolder = "atomic_data"
    else:
        _spadl = spadl
        datafolder = "default_data"

    spadl_h5 = os.path.join(datafolder, "spadl.h5")

    with pd.HDFStore(spadl_h5) as spadlstore:
        competitions = spadlstore["competitions"]
        games = spadlstore["games"]

    aggregates = []
    for competition_id in competitions.competition_id:
        aggregates.append(_get_action_aggregates_in_competition(spadl_h5, competition_id, games, _spadl, normalize))

    aggregates = pd.concat(aggregates).merge(
        competitions[["competition_id", "competition_name"]], left_index=True, right_on="competition_id").set_index(
        ["competition_id", "competition_name"])
    aggregates["actions"] = aggregates.sum(axis=1)
    return aggregates


def competition_games_players(player_setbacks: pd.DataFrame, team_setbacks: pd.DataFrame,
                              team_setbacks_over_matches: pd.DataFrame) -> pd.DataFrame:
    spadl_h5 = os.path.join("atomic_data", "spadl.h5")

    with pd.HDFStore(spadl_h5) as spadlstore:
        competitions = spadlstore["competitions"]
        games = spadlstore["games"]
        player_games = spadlstore["player_games"]

    aggregates = []
    for competition in competitions.itertuples():
        games_in_competition = games[games.competition_id == competition.competition_id]
        nb_games = games_in_competition.shape[0]
        nb_players = len(player_games[player_games.game_id.isin(games_in_competition.game_id)].player_id.unique())

        nb_missed_penalties = player_setbacks[player_setbacks.setback_type == "missed penalty"].shape[0]
        nb_missed_shots = player_setbacks[player_setbacks.setback_type == "missed shot"].shape[0]
        nb_foul_leading_to_goal = player_setbacks[player_setbacks.setback_type == "foul leading to goal"].shape[0]
        nb_bad_pass_leading_to_goal = player_setbacks[player_setbacks.setback_type == "bad pass leading to goal"].shape[
            0]
        nb_bad_consecutive_passes = player_setbacks[player_setbacks.setback_type == "consecutive bad passes"].shape[0]

        nb_goals_conceded = team_setbacks[team_setbacks.setback_type == "goal conceded"].shape[0]
        nb_consecutive_losses = team_setbacks_over_matches[
            team_setbacks_over_matches.setback_type == "consecutive losses"].shape[0]

        aggregates.append(pd.DataFrame(
            data={"competition_name": [competition.competition_name], "games": [nb_games], "players": [nb_players],
                  "missed penalties": [nb_missed_penalties], "missed shots": [nb_missed_shots],
                  "foul leading to goal": [nb_foul_leading_to_goal],
                  "bad pass leading to goal": [nb_bad_pass_leading_to_goal],
                  "bad consecutive passes": [nb_bad_consecutive_passes], "goals conceded": [nb_goals_conceded],
                  "consecutive losses": [nb_consecutive_losses]}).set_index(
            "competition_name"))

    aggregates = pd.concat(aggregates).merge(
        competitions[["competition_id", "competition_name"]], left_index=True, right_on="competition_name").set_index(
        ["competition_id", "competition_name"])
    return aggregates


def convert_team_to_player_setback(team_setbacks: pd.DataFrame, player_games: pd.DataFrame, actions: pd.DataFrame,
                                   players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    players_by_id = players.set_index("player_id", drop=False)
    player_setbacks = []
    player_games = player_games.merge(teams[["team_id", "team_name_short"]], left_on="team_id", right_on="team_id")

    for team_setback in tqdm(list(team_setbacks.itertuples()), desc="Converting team to player setbacks:"):
        actions_in_game = actions[actions.game_id == team_setback.game_id]

        group_by_period = actions_in_game[actions_in_game.period_id != 5].groupby("period_id")
        last_action_in_period = []
        for _, period in group_by_period:
            last_action_in_period.append(round(period.time_seconds.max() / 60))
        minutes_before_period = sum(last_action_in_period[:team_setback.period_id - 1])
        minute_of_setback = round(team_setback.time_seconds / 60) + minutes_before_period

        players_in_game = player_games[
            (player_games.game_id == team_setback.game_id) & (player_games.team_name_short == team_setback.team)]
        players_on_field = []
        for player in players_in_game[players_in_game.is_starter].itertuples():
            if player.minutes_played > minute_of_setback:
                players_on_field.append(player)

        for player in players_in_game[~players_in_game.is_starter].itertuples():
            if (players_in_game.minutes_played.max() - player.minutes_played) < minute_of_setback:
                players_on_field.append(player)

        for player in players_on_field:
            player = players_by_id.loc[player.player_id]
            player_setbacks.append(pd.DataFrame(
                data={"player": [player.nickname], "player_id": [player.player_id], "birth_date": [player.birth_date],
                      "player_team": [team_setback.team], "opponent_team": [team_setback.opponent],
                      "game_id": [team_setback.game_id], "home": [team_setback.home], "setback_type": ["goal conceded"],
                      "period_id": [team_setback.period_id], "time_seconds": [team_setback.time_seconds],
                      "score:": [team_setback.score]}))

    player_setbacks = pd.concat(player_setbacks).reset_index(drop=True)
    return player_setbacks


def players_in_all_games(player_games: pd.DataFrame, game_ids: list) -> list:
    games = player_games[player_games['game_id'].isin(game_ids)]
    nb_games = games['player_id'].value_counts()
    all_games_played = nb_games[nb_games > int(len(game_ids) / 2)]
    return all_games_played.index.tolist()


def extend_with_playerlist(team_setbacks_over_matches: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    team_setbacks_over_matches['playerlist'] = team_setbacks_over_matches.apply(
        lambda x: players_in_all_games(player_games, x['lost game(s)']), axis=1
    )
    return team_setbacks_over_matches


def get_player_aggregates(normalize=False) -> pd.DataFrame:
    _spadl = spadl
    datafolder = "default_data"

    spadl_h5 = os.path.join(datafolder, "spadl.h5")
    setbacks_h5 = os.path.join(datafolder, "setbacks.h5")

    with pd.HDFStore(spadl_h5) as spadlstore:
        players = spadlstore["players"]
        player_games = spadlstore["player_games"]
        games = spadlstore["games"]
        teams = spadlstore["teams"]
        all_actions = []
        for game_id in tqdm(list(games.game_id), desc="Collecting all actions: "):
            actions = spadlstore[f"actions/game_{game_id}"]
            actions = _spadl.add_names(actions)
            all_actions.append(actions)

    all_actions = pd.concat(all_actions).reset_index(drop=True)

    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_setbacks = setbackstore["teams_setbacks"]
        team_setbacks_over_matches = setbackstore["team_setbacks_over_matches"]

    # player_setbacks = pd.concat(
    #     [player_setbacks, convert_team_to_player_setback(team_setbacks, player_games, actions, players, teams)])
    team_setbacks_over_matches = extend_with_playerlist(team_setbacks_over_matches, player_games)

    for player in tqdm(list(players.itertuples()), desc="Collecting aggregates for players: "):
        games = player_games[player_games.player_id == player.player_id]
        nb_games = games.shape[0]
        nb_started = games[games['is_starter']].shape[0]
        minutes_played = games['minutes_played'].sum()
        avg_minutes = minutes_played / nb_games

        actions = all_actions[all_actions.player_id == player.player_id]
        action_counts = pd.Series.to_frame(
            actions["type_name"].value_counts(normalize=normalize, dropna=False)).T.sort_index(axis=1)

        setbacks = player_setbacks[player_setbacks.player_id == player.player_id]
        setback_type1 = setbacks[['setback_type']]
        setback_type2 = team_setbacks_over_matches[pd.DataFrame(team_setbacks_over_matches['playerlist'].tolist()).isin(
            [player.player_id]).any(axis=1).values][['setback_type']]
        setback_type = pd.concat([setback_type1, setback_type2]).reset_index(drop=True)
        # inaccurate = opportunity[pd.DataFrame(opportunity.tags.tolist()).isin([1802]).any(axis=1).values]

        setback_counts = pd.Series.to_frame(setback_type["setback_type"].value_counts(dropna=False)).T.sort_index(
            axis=1)
        print(player.player_name)
        print(player.position)
        print(nb_games)
        print(nb_started)
        print(minutes_played)
        print(avg_minutes)
        print(action_counts)
        print(setback_counts)
        print()


def get_competition_aggregates_and_store_to_excel():
    atomic = _get_action_aggregates(True, False)
    atomic_normalized = _get_action_aggregates(True, True)
    default = _get_action_aggregates(False, False)
    default_normalized = _get_action_aggregates(False, True)

    datafolder = "default_data"
    setbacks_h5 = os.path.join(datafolder, "setbacks.h5")
    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore["player_setbacks"]
        team_setbacks = setbackstore["teams_setbacks"]
        team_setbacks_over_matches = setbackstore["team_setbacks_over_matches"]

    other_aggregates = competition_games_players(player_setbacks, team_setbacks, team_setbacks_over_matches)

    atomic = pd.merge(atomic, other_aggregates, left_index=True, right_index=True)
    default = default.merge(other_aggregates, left_index=True, right_index=True)

    with pd.ExcelWriter("results/aggregates.xlsx") as writer:
        atomic.to_excel(writer, "Atomic")
        atomic_normalized.to_excel(writer, "Atomic", startrow=atomic.shape[0] + 2)
        default.to_excel(writer, "Default")
        default_normalized.to_excel(writer, "Default", startrow=default.shape[0] + 2)
