import os
import pandas as pd

from socceraction.data.wyscout import PublicWyscoutLoader
from socceraction.spadl.wyscout import convert_to_actions
from socceraction.atomic.spadl import convert_to_atomic
from typing import Dict
from tqdm import tqdm


def _select_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe containing all selected competitions.

    :param competitions: a dataframe with all available competitions
    :return: a dataframe containing all selected competitions
    """
    # Uncomment all wanted competitions
    all_competitions = [
        'Italian first division',
        'English first division',
        'Spanish first division',
        'French first division',
        'German first division',
        'European Championship',
        'World Cup'
    ]
    selected_competitions = pd.concat([competitions[competitions.competition_name == competition]
                                       for competition in all_competitions])
    return selected_competitions


def _get_games_in_competitions(competitions: pd.DataFrame, pwl: PublicWyscoutLoader) -> pd.DataFrame:
    """
    Returns a dataframe containing all games in the given competitions.

    :param pwl: the public Wyscout data loader
    :param competitions: a dataframe of the competitions for which the games must be returned
    :return: a dataframe containing all games in the given competitions
    """
    return pd.concat([pwl.games(competition.competition_id, competition.season_id)
                      for competition in competitions.itertuples()])


def _load_and_convert_data(games: pd.DataFrame, pwl: PublicWyscoutLoader, atomic: bool) -> (pd.DataFrame, pd.DataFrame,
                                                                                            Dict[int, pd.DataFrame]):
    """
    Returns the teams, players and actions in the given games.

    :param atomic: boolean flag indicating whether or not the actions should be atomic
    :param pwl: the public Wyscout data loader
    :param games: a dataframe of games for which the teams, players and actions should be returned
    :return: a dataframe containing all teams participating in the given games,
             a dataframe containing all players participating in the given games and
             a dictionary mapping the game_id's of the given games to their respective dataframe of actions (in (atomic)
             spadl format
    """
    teams, players = [], []
    actions = {}
    for game in tqdm(list(games.itertuples()), desc="Loading and converting game data"):
        teams.append(pwl.teams(game.game_id))
        players.append(pwl.players(game.game_id))
        # events = pwl.events(game.game_id)
        # actions[game.game_id] = convert_to_actions(events, game.home_team_id)
        # if atomic:
        #     actions[game.game_id] = convert_to_atomic(actions[game.game_id])

    teams = pd.concat(teams).drop_duplicates(subset='team_id')
    players = pd.concat(players)
    return teams, players, actions


def _store_data(competitions: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame, players: pd.DataFrame,
                actions: pd.DataFrame, datafolder: str):
    """
    Stores the given dataframes in a shared h5 file in the given datafolder.

    :param datafolder: the datafolder where the data should be stored
    :param competitions: a competitions dataframe
    :param games: a games dataframe
    :param teams: a teams dataframe
    :param players: a players dataframe
    :param actions: a dictionary of action dataframes
    """
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    spadl_h5 = os.path.join(datafolder, "spadl.h5")

    with pd.HDFStore(spadl_h5) as spadlstore:
        # for key in spadlstore.keys():
        #     if "actions/game_" in key:
        #         del spadlstore[key]
        spadlstore["competitions"] = competitions
        spadlstore["games"] = games.reset_index(drop=True)
        spadlstore["teams"] = teams.reset_index(drop=True)
        spadlstore["players"] = players[['player_id', 'player_name', 'nickname', 'birth_date']].drop_duplicates(
            subset='player_id').reset_index(drop=True)
        spadlstore["player_games"] = players[
            ['player_id', 'game_id', 'team_id', 'is_starter', 'minutes_played']].reset_index(drop=True)
        # for game_id in actions.keys():
        #     spadlstore[f"actions/game_{game_id}"] = actions[game_id]


def load_and_convert_wyscout_data(atomic=True, download=False):
    """
    Downloads public Wyscout dataset if necessary, converts it to spadl format and writes data corresponding to the
    requested competitions to an h5 file.

    :param atomic: boolean flag indicating whether or not the actions should be atomic
    :param download: boolean flag indicating whether or not the Wyscout dataset should be (re)downloaded
    """
    if atomic:
        datafolder = "atomic_data"
    else:
        datafolder = "default_data"

    if download:
        print("Downloading public Wyscout dataset...")

    pwl = PublicWyscoutLoader(download=download)
    competitions = pwl.competitions()
    selected_competitions = _select_competitions(competitions)
    games = _get_games_in_competitions(selected_competitions, pwl)
    teams, players, actions = _load_and_convert_data(games, pwl, atomic)
    _store_data(selected_competitions, games, teams, players, actions, datafolder)
