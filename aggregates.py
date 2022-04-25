import os
from typing import List

import pandas as pd
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl

from tqdm import tqdm

import utils


def _get_action_aggregates_in_competition(spadl_h5: str, competition_id: int, games: pd.DataFrame, _spadl,
                                          normalize) -> pd.DataFrame:
    # Get the games in the given competition
    competition_games = games[games.competition_id == competition_id]

    # Get the actions in the given competition
    all_actions = []
    with pd.HDFStore(spadl_h5) as spadlstore:
        for game_id in tqdm(list(competition_games.game_id), desc="Collecting all actions in competition: "):
            actions = spadlstore['actions/game_{}'.format(game_id)]
            actions = _spadl.add_names(actions)
            all_actions.append(actions)

    all_actions = pd.concat(all_actions).reset_index(drop=True)

    # Get the counts for every action
    action_counts = pd.Series.to_frame(
        all_actions['type_name'].value_counts(normalize=normalize, dropna=False)).T.sort_index(axis=1)

    # Set the index
    action_counts['competition_id'] = competition_id
    action_counts = action_counts.set_index('competition_id')

    return action_counts


def _get_action_aggregates(atomic=True, normalize=False) -> pd.DataFrame:
    if atomic:
        _spadl = aspadl
        datafolder = 'atomic_data'
    else:
        _spadl = spadl
        datafolder = 'default_data'

    spadl_h5 = os.path.join(datafolder, 'spadl.h5')

    with pd.HDFStore(spadl_h5) as spadlstore:
        competitions = spadlstore['competitions']
        games = spadlstore['games']

    # Get the action counts for every competition
    aggregates = []
    for competition_id in competitions.competition_id:
        aggregates.append(_get_action_aggregates_in_competition(spadl_h5, competition_id, games, _spadl, normalize))

    # Add competition name and set index to competition id and name
    aggregates = pd.concat(aggregates).merge(
        competitions[['competition_id', 'competition_name']], left_index=True, right_on='competition_id').set_index(
        ['competition_id', 'competition_name'])

    # Get the total number of actions in every competition
    aggregates['actions'] = aggregates.sum(axis=1)

    return aggregates


def _get_setback_aggregates(player_setbacks: pd.DataFrame, team_setbacks: pd.DataFrame,
                            team_setbacks_over_matches: pd.DataFrame) -> pd.DataFrame:
    spadl_h5 = 'atomic_data/spadl.h5'
    with pd.HDFStore(spadl_h5) as spadlstore:
        competitions = spadlstore['competitions']
        games = spadlstore['games']
        player_games = spadlstore['player_games']

    # Get the number of players, games and (different) setbacks in every competition
    aggregates = []
    for competition in tqdm(list(competitions.itertuples()), desc="Collecting competition aggregates: "):
        # Get number of players and games in the competition
        games_in_competition = games[games.competition_id == competition.competition_id]
        game_ids = games_in_competition.game_id.tolist()
        nb_games = games_in_competition.shape[0]
        nb_players = len(player_games[player_games.game_id.isin(games_in_competition.game_id)].player_id.unique())

        # Get the setbacks in the games of this competition
        ps_in_competition = player_setbacks[player_setbacks.game_id.isin(game_ids)]
        ts_in_competition = team_setbacks[team_setbacks.game_id.isin(game_ids)]
        team_setbacks_over_matches['first_game'] = team_setbacks_over_matches.apply(lambda x: x['lost_games'][0],
                                                                                    axis=1)
        ts_over_matches_in_competition = team_setbacks_over_matches[
            team_setbacks_over_matches.first_game.isin(game_ids)]

        # Get the number of setbacks in the competition
        nb_missed_penalties = ps_in_competition[ps_in_competition.setback_type == 'missed penalty'].shape[0]
        nb_missed_shots = ps_in_competition[ps_in_competition.setback_type == 'missed shot'].shape[0]
        nb_foul_leading_to_goal = ps_in_competition[ps_in_competition.setback_type == 'foul leading to goal'].shape[0]
        nb_bad_pass_leading_to_goal = ps_in_competition[
            ps_in_competition.setback_type == 'bad pass leading to goal'].shape[0]
        nb_bad_consecutive_passes = ps_in_competition[
            ps_in_competition.setback_type == 'bad consecutive passes'].shape[0]

        nb_goals_conceded = ts_in_competition[ts_in_competition.setback_type == 'goal conceded'].shape[0]
        nb_consecutive_losses = ts_over_matches_in_competition[
            ts_over_matches_in_competition.setback_type == 'consecutive losses'].shape[0]

        # Construct dataframe
        aggregates.append(pd.DataFrame(
            data={'competition_name': [competition.competition_name],
                  'games': [nb_games],
                  'players': [nb_players],
                  'missed penalties': [nb_missed_penalties],
                  'missed shots': [nb_missed_shots],
                  'foul leading to goal': [nb_foul_leading_to_goal],
                  'bad pass leading to goal': [nb_bad_pass_leading_to_goal],
                  'bad consecutive passes': [nb_bad_consecutive_passes],
                  'goals conceded': [nb_goals_conceded],
                  'consecutive losses': [nb_consecutive_losses]}).set_index(
            'competition_name'))

    # Set the index to the competition id and name
    aggregates = pd.concat(aggregates).merge(
        competitions[['competition_id', 'competition_name']], left_index=True, right_on='competition_name').set_index(
        ['competition_id', 'competition_name'])

    return aggregates


def _players_in_majority_of_games(player_games: pd.DataFrame, game_ids: list) -> list:
    # Get the players who played in the majority of the given games
    games = player_games[player_games['game_id'].isin(game_ids)]
    nb_games_per_player = games['player_id'].value_counts()
    majority_games_played = nb_games_per_player[nb_games_per_player > int(len(game_ids) / 2)]

    return majority_games_played.competition_index.tolist()


def _extend_with_playerlist(team_setbacks_over_matches: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    # Extend a teams setback over multiple games with a column containing the players who played the majority of those
    # games
    team_setbacks_over_matches['playerlist'] = team_setbacks_over_matches.apply(
        lambda x: _players_in_majority_of_games(player_games, x['lost_games']), axis=1
    )

    return team_setbacks_over_matches


def _get_player_aggregates(player_setbacks: pd.DataFrame, team_as_player_setbacks: pd.DataFrame,
                           team_setbacks_over_matches: pd.DataFrame, player_games: pd.DataFrame, players: pd.DataFrame,
                           all_actions: pd.DataFrame, normalize=False) -> pd.DataFrame:
    player_setbacks = pd.concat(
        [player_setbacks, team_as_player_setbacks])
    team_setbacks_over_matches = _extend_with_playerlist(team_setbacks_over_matches, player_games)

    all_setback_types = ['missed penalty', 'missed shot', 'goal conceded', 'foul leading to goal',
                         'bad pass leading to goal', 'bad consecutive passes', 'consecutive losses']
    all_action_types = ['bad_touch', 'clearance', 'corner_crossed', 'corner_short', 'cross', 'dribble', 'foul',
                        'freekick_crossed', 'freekick_short', 'goalkick', 'interception', 'keeper_save', 'pass', 'shot',
                        'shot_freekick', 'shot_penalty', 'tackle', 'take_on', 'throw_in']

    # Collect aggregates for individual players
    aggregates = []
    for player in tqdm(list(players.itertuples()), desc="Collecting aggregates for players: "):
        # Get games played/start, total minutes played and average minutes played
        games = player_games[player_games.player_id == player.player_id]
        nb_games = games.shape[0]
        nb_started = games[games['is_starter']].shape[0]
        minutes_played = games['minutes_played'].sum()
        avg_minutes = minutes_played / nb_games

        # Get the number of actions of this player
        actions = all_actions[all_actions.player_id == player.player_id]
        action_counts = pd.Series.to_frame(
            actions['type_name'].value_counts(normalize=normalize, dropna=False)).T.reset_index(drop=True)

        # Set the action count for non-encountered actions to 0
        for a_type in all_action_types:
            if a_type not in action_counts:
                action_counts[a_type] = 0
        action_counts = action_counts.sort_index(axis=1)

        # Get the number of setbacks of this player
        setbacks = player_setbacks[player_setbacks.player_id == player.player_id]
        setback_type1 = setbacks[['setback_type']]
        setback_type2 = team_setbacks_over_matches[pd.DataFrame(team_setbacks_over_matches['playerlist'].tolist()).isin(
            [player.player_id]).any(axis=1).values][['setback_type']]
        setback_type = pd.concat([setback_type1, setback_type2]).reset_index(drop=True)

        setback_counts = pd.Series.to_frame(setback_type['setback_type'].value_counts(dropna=False)).T.reset_index(
            drop=True)

        # Set the setback count for non-encountered setbacks to 0
        for sb_type in all_setback_types:
            if sb_type not in setback_counts:
                setback_counts[sb_type] = 0
        setback_counts.sort_index(axis=1)

        # Construct a dataframe
        other_counts = pd.DataFrame(
            data={'player': [player.player_name],
                  'player_id': [player.player_id],
                  'position': [player.position],
                  'games': [nb_games],
                  'started': [nb_started],
                  'minutes played': [minutes_played],
                  'average minutes per game': [avg_minutes]}
        )
        aggregates.append(
            pd.concat([other_counts, action_counts, setback_counts], axis=1).set_index(['player_id', 'player']))

    aggregates = pd.concat(aggregates).reset_index(drop=True)

    return aggregates


def get_player_aggregates_and_store(competitions: List[str]):
    _spadl = spadl

    spadl_h5 = 'default_data/spadl.h5'
    with pd.HDFStore(spadl_h5) as spadlstore:
        players = spadlstore['players']
        player_games = spadlstore['player_games']
        games = spadlstore['games']
        # Only consider games in the given competitions
        games = games[games['competition_name'].isin(competitions)]
        all_actions = []
        for game_id in tqdm(list(games.game_id), desc="Collecting all actions: "):
            actions = spadlstore[f'actions/game_{game_id}']
            actions = _spadl.add_names(actions)
            all_actions.append(actions)

    all_actions = pd.concat(all_actions).reset_index(drop=True)

    setbacks_h5 = 'setback_data/setbacks.h5'
    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore['player_setbacks']
        team_setbacks_over_matches = setbackstore['team_setbacks_over_matches']
        team_as_player_setbacks = setbackstore['team_as_player_setbacks']

    player_aggregates = _get_player_aggregates(player_setbacks, team_as_player_setbacks, team_setbacks_over_matches,
                                               player_games, players, all_actions, normalize=False)
    player_aggregates_normalized = _get_player_aggregates(player_setbacks, team_as_player_setbacks,
                                                          team_setbacks_over_matches, player_games, players,
                                                          all_actions, normalize=True)

    with pd.ExcelWriter('results/player_aggregates.xlsx') as writer:
        player_aggregates.to_excel(writer, 'Player')
        player_aggregates_normalized.to_excel(writer, 'Player_Norm')

    aggregates_h5 = 'results/aggregates.h5'
    with pd.HDFStore(aggregates_h5) as store:
        store['player_agg'] = player_aggregates
        store['player_agg_norm'] = player_aggregates_normalized


def get_competition_aggregates_and_store():
    atomic = _get_action_aggregates(True, False)
    atomic_normalized = _get_action_aggregates(True, True)
    default = _get_action_aggregates(False, False)
    default_normalized = _get_action_aggregates(False, True)

    setbacks_h5 = 'setback_data/setbacks.h5'
    with pd.HDFStore(setbacks_h5) as setbackstore:
        player_setbacks = setbackstore['player_setbacks']
        team_setbacks = setbackstore['teams_setbacks']
        team_setbacks_over_matches = setbackstore['team_setbacks_over_matches']

    other_aggregates = _get_setback_aggregates(player_setbacks, team_setbacks, team_setbacks_over_matches)

    atomic = pd.merge(atomic, other_aggregates, left_index=True, right_index=True)
    default = default.merge(other_aggregates, left_index=True, right_index=True)

    with pd.ExcelWriter('results/competition_aggregates.xlsx') as writer:
        atomic.to_excel(writer, 'Atomic')
        atomic_normalized.to_excel(writer, 'Atomic', startrow=atomic.shape[0] + 2)
        default.to_excel(writer, 'Default')
        default_normalized.to_excel(writer, 'Default', startrow=default.shape[0] + 2)

    datafolder = 'aggregates'
    aggregates_h5 = os.path.join(datafolder, 'aggregates.h5')
    with pd.HDFStore(aggregates_h5) as aggregatesstore:
        aggregatesstore['competition_agg_atomic'] = atomic
        aggregatesstore['competition_agg_atomic_norm'] = atomic_normalized
        aggregatesstore['competition_agg'] = default
        aggregatesstore['competition_agg_norm'] = default_normalized


def main():
    competitions = utils.test_competitions
    get_competition_aggregates_and_store()
    get_player_aggregates_and_store(competitions=competitions)


if __name__ == '__main__':
    main()
