import os
import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl

from tqdm import tqdm


def _get_action_aggregates_in_competition(spadl_h5: str, competition_id: int, games: pd.DataFrame, _spadl,
                                          normalize) -> pd.DataFrame:
    competition_games = games[games.competition_id == competition_id]
    all_actions = []
    with pd.HDFStore(spadl_h5) as spadlstore:
        for game_id in tqdm(list(competition_games.game_id), desc="Collecting all actions in competition: "):
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
    for competition in tqdm(list(competitions.itertuples()), desc="Collecting competition aggregates: "):
        games_in_competition = games[games.competition_id == competition.competition_id]
        game_ids = games_in_competition.game_id.tolist()
        nb_games = games_in_competition.shape[0]
        nb_players = len(player_games[player_games.game_id.isin(games_in_competition.game_id)].player_id.unique())

        ps_in_competition = player_setbacks[player_setbacks.game_id.isin(game_ids)]
        ts_in_competition = team_setbacks[team_setbacks.game_id.isin(game_ids)]
        team_setbacks_over_matches['first_game'] = team_setbacks_over_matches.apply(lambda x: x['lost game(s)'][0],
                                                                                    axis=1)
        ts_over_matches_in_competition = team_setbacks_over_matches[
            team_setbacks_over_matches.first_game.isin(game_ids)]

        nb_missed_penalties = ps_in_competition[ps_in_competition.setback_type == "missed penalty"].shape[0]
        nb_missed_shots = ps_in_competition[ps_in_competition.setback_type == "missed shot"].shape[0]
        nb_foul_leading_to_goal = ps_in_competition[ps_in_competition.setback_type == "foul leading to goal"].shape[0]
        nb_bad_pass_leading_to_goal = ps_in_competition[
            ps_in_competition.setback_type == "bad pass leading to goal"].shape[0]
        nb_bad_consecutive_passes = ps_in_competition[
            ps_in_competition.setback_type == "bad consecutive passes"].shape[0]

        nb_goals_conceded = ts_in_competition[ts_in_competition.setback_type == "goal conceded"].shape[0]
        nb_consecutive_losses = ts_over_matches_in_competition[
            ts_over_matches_in_competition.setback_type == "consecutive losses"].shape[0]

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
        minutes_before_current_period = sum(last_action_in_period[:team_setback.period_id - 1])
        minute_of_setback = round(team_setback.time_seconds / 60) + minutes_before_current_period

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


def get_player_aggregates(player_setbacks: pd.DataFrame, team_as_player_setbacks: pd.DataFrame,
                          team_setbacks_over_matches: pd.DataFrame, player_games: pd.DataFrame, players: pd.DataFrame,
                          all_actions: pd.DataFrame, normalize=False) -> pd.DataFrame:
    player_setbacks = pd.concat(
        [player_setbacks, team_as_player_setbacks])
    team_setbacks_over_matches = extend_with_playerlist(team_setbacks_over_matches, player_games)
    all_setback_types = ["missed penalty", "missed shot", "goal conceded", "foul leading to goal",
                         "bad pass leading to goal", "bad consecutive passes", "consecutive losses"]
    all_action_types = ["bad_touch", "clearance", "corner_crossed", "corner_short", "cross", "dribble", "foul",
                        "freekick_crossed", "freekick_short", "goalkick", "interception", "keeper_save", "pass", "shot",
                        "shot_freekick", "shot_penalty", "tackle", "take_on", "throw_in"]

    aggregates = []
    for player in tqdm(list(players.itertuples()), desc="Collecting aggregates for players: "):
        games = player_games[player_games.player_id == player.player_id]
        nb_games = games.shape[0]
        nb_started = games[games['is_starter']].shape[0]
        minutes_played = games['minutes_played'].sum()
        avg_minutes = minutes_played / nb_games

        actions = all_actions[all_actions.player_id == player.player_id]
        action_counts = pd.Series.to_frame(
            actions["type_name"].value_counts(normalize=normalize, dropna=False)).T.reset_index(drop=True)

        for a_type in all_action_types:
            if a_type not in action_counts:
                action_counts[a_type] = 0
        action_counts = action_counts.sort_index(axis=1)

        setbacks = player_setbacks[player_setbacks.player_id == player.player_id]
        setback_type1 = setbacks[['setback_type']]
        setback_type2 = team_setbacks_over_matches[pd.DataFrame(team_setbacks_over_matches['playerlist'].tolist()).isin(
            [player.player_id]).any(axis=1).values][['setback_type']]
        setback_type = pd.concat([setback_type1, setback_type2]).reset_index(drop=True)

        setback_counts = pd.Series.to_frame(setback_type["setback_type"].value_counts(dropna=False)).T.reset_index(
            drop=True)

        for sb_type in all_setback_types:
            if sb_type not in setback_counts:
                setback_counts[sb_type] = 0
        setback_counts.sort_index(axis=1)

        other_counts = pd.DataFrame(
            data={"player": [player.player_name], "player_id": [player.player_id], "position": [player.position],
                  "games": [nb_games], "started": [nb_started], "minutes played": [minutes_played],
                  "average minutes per game": [avg_minutes]}
        )
        aggregates.append(
            pd.concat([other_counts, action_counts, setback_counts], axis=1).set_index(['player_id', 'player']))

    aggregates = pd.concat(aggregates)
    return aggregates


def get_player_aggregates_and_store():
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
        setbackstore['team_as_player_setbacks'] = convert_team_to_player_setback(team_setbacks, player_games, actions,
                                                                                 players, teams)
        team_as_player_setbacks = setbackstore['team_as_player_setbacks']

    aggregates = get_player_aggregates(player_setbacks, team_as_player_setbacks, team_setbacks_over_matches,
                                       player_games, players, all_actions, normalize=False)
    aggregates_normalized = get_player_aggregates(player_setbacks, team_as_player_setbacks, team_setbacks_over_matches,
                                                  player_games, players, all_actions, normalize=True)

    with pd.ExcelWriter("results/player_aggregates.xlsx") as writer:
        aggregates.to_excel(writer, "Player")
        aggregates_normalized.to_excel(writer, "Player_Norm")

    datafolder = "aggregates"
    aggregates_h5 = os.path.join(datafolder, "aggregates.h5")
    with pd.HDFStore(aggregates_h5) as aggregatesstore:
        aggregatesstore["player_agg"] = aggregates
        aggregatesstore["player_agg_norm"] = aggregates_normalized


def get_competition_aggregates_and_store():
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

    with pd.ExcelWriter("results/competition_aggregates.xlsx") as writer:
        atomic.to_excel(writer, "Atomic")
        atomic_normalized.to_excel(writer, "Atomic", startrow=atomic.shape[0] + 2)
        default.to_excel(writer, "Default")
        default_normalized.to_excel(writer, "Default", startrow=default.shape[0] + 2)

    datafolder = "aggregates"
    aggregates_h5 = os.path.join(datafolder, "aggregates.h5")
    with pd.HDFStore(aggregates_h5) as aggregatesstore:
        aggregatesstore["competition_agg_atomic"] = atomic
        aggregatesstore["competition_agg_atomic_norm"] = atomic_normalized
        aggregatesstore["competition_agg"] = default
        aggregatesstore["competition_agg_norm"] = default_normalized
