import os
import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl


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
    return aggregates


def get_action_aggregates_and_store_to_excel():
    atomic = _get_action_aggregates(True, False)
    atomic_normalized = _get_action_aggregates(True, True)
    default = _get_action_aggregates(False, False)
    default_normalized = _get_action_aggregates(False, True)

    with pd.ExcelWriter("results/aggregates.xlsx") as writer:
        atomic.to_excel(writer, "Atomic")
        atomic_normalized.to_excel(writer, "Atomic", startrow=atomic.shape[0] + 2)
        default.to_excel(writer, "Default")
        default_normalized.to_excel(writer, "Default", startrow=default.shape[0] + 2)
