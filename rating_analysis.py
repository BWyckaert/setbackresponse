import pandas as pd

from tqdm import tqdm

import utils


def map_big_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    actions['score_diff'] = actions.apply(lambda x: -2 if x['score_diff'] < -2 else x['score_diff'], axis=1)
    actions['score_diff'] = actions.apply(lambda x: 2 if x['score_diff'] > 2 else x['score_diff'], axis=1)
    return actions


def get_rating_progression_with_goal_diff_in_game(actions: pd.DataFrame, per_minute: bool) -> pd.DataFrame:
    actions = map_big_goal_diff(utils.add_goal_diff_atomic(actions))
    grouped_by_diff = actions.groupby('score_diff')

    vaep_mean = []
    for goal_diff, actions_by_diff in grouped_by_diff:
        vaep_mean.append(get_rating_progression_during_game(actions_by_diff, per_minute).rename(index={0: goal_diff}))

    return pd.concat(vaep_mean)


def get_rating_progression_with_goal_diff(games: pd.DataFrame, actions: pd.DataFrame, per_minute: bool) -> pd.DataFrame:
    game_ratings = []
    for game_id in tqdm(list(games.game_id), desc="Get rating progression during games: "):
        game_ratings.append(
            get_rating_progression_with_goal_diff_in_game(
                actions[(actions.game_id == game_id) & (actions.period_id != 5)].reset_index(drop=True), per_minute))

    game_ratings = pd.concat(game_ratings)
    grouped_by_index = game_ratings.groupby(game_ratings.index)
    count = grouped_by_index.count().astype(int)[utils.column_order]
    mean = grouped_by_index.mean()[utils.column_order]
    std = grouped_by_index.std()[utils.column_order]
    return count, mean, std


def get_rating_progression_during_game(actions: pd.DataFrame, per_minute: bool) -> pd.DataFrame:
    actions = utils.add_total_seconds_to_game(actions)

    chunck_size = 10
    actions['time_chunck'] = actions.apply(
        lambda x: (x['total_seconds'] // (chunck_size * 60)) if (x['total_seconds'] < 5400) else -1, axis=1)
    group_by_time_chunks = actions.groupby('time_chunck')

    vaep_mean = {}
    for index, time_chunk in group_by_time_chunks:
        # Set the label
        if index == -1:
            label = "90+"
        else:
            label = "{} to {}".format(int(index) * chunck_size, (int(index) + 1) * chunck_size)

        if per_minute:
            # Average VAEP value per minute
            if index == -1:
                # Taking chunk_size would be overestimate if eg only 91 minutes are played
                vaep_mean[label] = [time_chunk['vaep_value'].sum() / (
                        (time_chunk['total_seconds'].iloc[-1] / 60) - 90)]
            else:
                vaep_mean[label] = [time_chunk['vaep_value'].sum() / chunck_size]
        else:
            # Average VAEP value per action
            vaep_mean[label] = [time_chunk['vaep_value'].mean()]

    return pd.DataFrame(vaep_mean)


def get_rating_progression(games: pd.DataFrame, actions: pd.DataFrame, per_minute: bool) -> pd.DataFrame:
    game_ratings = []
    for game_id in tqdm(list(games.game_id), desc="Get rating progression during games: "):
        game_actions = actions[(actions.game_id == game_id) & (actions.period_id != 5)]
        # When calculating player rating progression, game_actions can be empty
        if not game_actions.empty:
            game_ratings.append(get_rating_progression_during_game(game_actions.reset_index(drop=True), per_minute))

    game_ratings = pd.concat(game_ratings).reset_index(drop=True)
    count = game_ratings.count()[utils.column_order].astype(int).to_frame().rename(columns={0: 'Count'})
    mean = game_ratings.mean()[utils.column_order].to_frame().rename(columns={0: 'Mean'})
    std = game_ratings.std()[utils.column_order].to_frame().rename(columns={0: 'Std'})

    return pd.concat([count, mean, std], axis=1)


def store_rating_progression_per_player(actions: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame,
                                        player_games: pd.DataFrame):
    # Only consider player_games where the game_id is also in games
    player_games = player_games[player_games['game_id'].isin(games['game_id'].tolist())]

    player_rating_progression_h5 = "results/pr_progression.h5"
    with pd.HDFStore(player_rating_progression_h5) as store:
        for player_id in tqdm(list(players.player_id), "Getting rating progression per player: "):
            # Take the player_games for the player
            pg_for_player = player_games[player_games['player_id'] == player_id]

            # Only consider players who played at least 10 full games
            if pg_for_player[pg_for_player['minutes_played'] >= 90].shape[0] >= 10:
                # Take the actions of the player in all games
                player_actions = actions[actions['player_id'] == player_id]

                # Only take the games in which the player played himself
                game_ids = pg_for_player['game_id'].tolist()
                games_where_player_played = games[games['game_id'].isin(game_ids)]

                # Get rating progression for the player
                progression_per_action = get_rating_progression(games_where_player_played, player_actions, False)
                progression_per_minute = get_rating_progression(games_where_player_played, player_actions, True)

                # Store rating progression
                store["per_action_{}".format(player_id)] = progression_per_action
                store["per_minute_{}".format(player_id)] = progression_per_minute


def get_rating_analysis_and_store(games: pd.DataFrame, actions: pd.DataFrame):
    rp_per_action = get_rating_progression(games, actions, False)
    rp_per_minute = get_rating_progression(games, actions, True)
    rp_per_action_with_goal_diff = get_rating_progression_with_goal_diff(games, actions, False)
    rp_per_minute_with_goal_diff = get_rating_progression_with_goal_diff(games, actions, True)

    game_rating_progression_h5 = "results/gr_progression_default.h5"
    with pd.HDFStore(game_rating_progression_h5) as store:
        store["per_action"] = rp_per_action
        store["per_minute"] = rp_per_minute

    with pd.ExcelWriter("results/rating_analysis_default.xlsx") as writer:
        rp_per_action.to_excel(writer, "Rating progression")
        rp_per_minute.to_excel(writer, "Rating progression per minute")
        rp_per_action_with_goal_diff[0].to_excel(writer, "With goal diff")
        rp_per_action_with_goal_diff[1].to_excel(writer, "With goal diff", startrow=8)
        rp_per_action_with_goal_diff[2].to_excel(writer, "With goal diff", startrow=15)

        rp_per_minute_with_goal_diff[0].to_excel(writer, "With goal diff per minute")
        rp_per_minute_with_goal_diff[1].to_excel(writer, "With goal diff per minute", startrow=8)
        rp_per_minute_with_goal_diff[2].to_excel(writer, "With goal diff per minute", startrow=15)
