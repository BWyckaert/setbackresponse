import os
import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import setbacks as sb
import numpy as np


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


def _get_action_aggregates(atomic=True, normalize=True) -> pd.DataFrame:
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


def competition_games_players():
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
        player_setbacks, team_setbacks, team_setbacks_over_matches = sb.get_setbacks([competition.competition_name],
                                                                                     atomic=False)

        nb_missed_penalties = player_setbacks[player_setbacks.setback_type == "missed penalty"].shape[0]
        nb_missed_shots = player_setbacks[player_setbacks.setback_type == "missed shot"].shape[0]
        nb_foul_leading_to_goal = player_setbacks[player_setbacks.setback_type == "foul leading to goal"].shape[0]
        nb_bad_pass_leading_to_goal = player_setbacks[player_setbacks.setback_type == "bad pass leading to goal"].shape[
            0]
        nb_bad_consecutive_passes = player_setbacks[player_setbacks.setback_type == "consecutive bad passes"].shape[0]

        nb_goals_conceded = team_setbacks[team_setbacks.setback_type == "goal conceded"].shape[0]
        nb_consecutive_losses = team_setbacks_over_matches[
            team_setbacks_over_matches.setback_type == "consecutive losses"].shape[0]

        aggregates.append(pd.DataFrame(data=np.array(
            [[competition.competition_name, nb_games, nb_players, nb_missed_penalties,
              nb_missed_shots, nb_foul_leading_to_goal, nb_bad_pass_leading_to_goal, nb_bad_consecutive_passes,
              nb_goals_conceded, nb_consecutive_losses]]),
            columns=["competition_name", "games", "players", "missed penalties", "missed shots",
                     "foul leading to goal", "bad pass leading to goal",
                     "bad consecutive passes", "goals conceded", "consecutive losses"]).set_index(
            "competition_name"))

    aggregates = pd.concat(aggregates).merge(
        competitions[["competition_id", "competition_name"]], left_index=True, right_on="competition_name").set_index(
        ["competition_id", "competition_name"])
    return aggregates


def get_competition_aggregates_and_store_to_excel():
    atomic = _get_action_aggregates(True, False)
    atomic_normalized = _get_action_aggregates(True, True)
    default = _get_action_aggregates(False, False)
    default_normalized = _get_action_aggregates(False, True)

    other_aggregates = competition_games_players()

    atomic = pd.merge(atomic, other_aggregates, left_index=True, right_index=True)
    default = default.merge(other_aggregates, left_index=True, right_index=True)

    with pd.ExcelWriter("results/aggregates.xlsx") as writer:
        atomic.to_excel(writer, "Atomic")
        atomic_normalized.to_excel(writer, "Atomic", startrow=atomic.shape[0] + 2)
        default.to_excel(writer, "Default")
        default_normalized.to_excel(writer, "Default", startrow=default.shape[0] + 2)
