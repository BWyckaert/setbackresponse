import pandas as pd

from tqdm import tqdm

import utils


def add_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    goallike = ['goal', 'owngoal']
    goal_actions = actions[actions['type_name'].isin(goallike)]
    actions['score_diff'] = 0

    for index, goal in goal_actions.iterrows():
        if not goal.equals(actions.iloc[-1].drop(labels=['score_diff'])):
            if goal['type_name'] == 'goal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] + 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] - 1, axis=1)
            if goal['type_name'] == 'owngoal':
                actions['score_diff'].iloc[index + 1:] = actions.iloc[index + 1:].apply(
                    lambda x: x['score_diff'] - 1 if (x['team_id'] == goal['team_id']) else x['score_diff'] + 1, axis=1)

    return actions


def map_big_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    actions['score_diff'] = actions.apply(lambda x: -2 if x['score_diff'] < -2 else x['score_diff'], axis=1)
    actions['score_diff'] = actions.apply(lambda x: 2 if x['score_diff'] > 2 else x['score_diff'], axis=1)
    return actions


def get_rating_progression_with_goal_diff_in_game(actions: pd.DataFrame, per_minute: bool) -> pd.DataFrame:
    actions = map_big_goal_diff(add_goal_diff(actions))
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
    actions = utils.add_total_seconds(actions)

    chunck_size = 5
    group_by_time_chunks = actions.groupby((actions['total_seconds'] // (chunck_size * 60)).astype(int))

    vaep_mean = {}
    for index, time_chunk in group_by_time_chunks:
        label = "{} to {}".format(index * chunck_size, (index + 1) * chunck_size)
        if per_minute:
            # Average VAEP value per actions
            vaep_mean[label] = [time_chunk['vaep_value'].sum() / chunck_size]
        else:
            # Average VAEP value per minute
            vaep_mean[label] = [time_chunk['vaep_value'].mean()]

    return pd.DataFrame(vaep_mean)


def get_rating_progression(games: pd.DataFrame, actions: pd.DataFrame, per_minute: bool) -> pd.DataFrame:
    game_ratings = []
    for game_id in tqdm(list(games.game_id), desc="Get rating progression during games: "):
        game_ratings.append(get_rating_progression_during_game(
            actions[(actions.game_id == game_id) & (actions.period_id != 5)].reset_index(drop=True), per_minute))

    game_ratings = pd.concat(game_ratings).reset_index(drop=True)
    count = game_ratings.count().astype(int).to_frame().rename(columns={0: 'Count'})
    mean = game_ratings.mean().to_frame().rename(columns={0: 'Mean'})
    std = game_ratings.std().to_frame().rename(columns={0: 'Std'})

    # return pd.concat([count, mean, std], axis=1)
    return game_ratings.describe()


def get_rating_analysis_and_store(games: pd.DataFrame, actions: pd.DataFrame):
    rp_per_action = get_rating_progression(games, actions, False)
    rp_per_minute = get_rating_progression(games, actions, True)
    rp_per_action_with_goal_diff = get_rating_progression_with_goal_diff(games, actions, False)
    rp_per_minute_with_goal_diff = get_rating_progression_with_goal_diff(games, actions, True)

    with pd.ExcelWriter("results/rating_analysis.xlsx") as writer:
        rp_per_action.to_excel(writer, "Rating progression")
        rp_per_minute.to_excel(writer, "Rating progression per minute")
        rp_per_action_with_goal_diff[0].to_excel(writer, "With goal diff")
        rp_per_action_with_goal_diff[1].to_excel(writer, "With goal diff", startrow=8)
        rp_per_action_with_goal_diff[2].to_excel(writer, "With goal diff", startrow=15)

        rp_per_minute_with_goal_diff[0].to_excel(writer, "With goal diff per minute")
        rp_per_minute_with_goal_diff[1].to_excel(writer, "With goal diff per minute", startrow=8)
        rp_per_minute_with_goal_diff[2].to_excel(writer, "With goal diff per minute", startrow=15)
