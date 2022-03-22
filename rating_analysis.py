import pandas as pd

from tqdm import tqdm


def add_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    goallike = ['goal', 'owngoal']
    goal_actions = actions[actions['type_name'].isin(goallike)]
    actions['score_diff'] = 0

    for index, goal in goal_actions.iterrows():
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


def rating_with_goal_diff(actions: pd.DataFrame) -> pd.DataFrame:
    actions = map_big_goal_diff(add_goal_diff(actions))
    grouped_by_diff = actions.groupby('score_diff')

    vaep_mean = []
    for _, actions_by_diff in grouped_by_diff:
        vaep_mean.append(rating_during_game(actions_by_diff).set_index('score_diff'))

    vaep_mean = pd.concat(vaep_mean)


def rating_during_game(actions: pd.DataFrame) -> pd.DataFrame:
    actions = actions[actions.period_id != 5]
    group_by_period = actions.groupby("period_id")
    last_action_in_period = []
    for _, period in group_by_period:
        last_action_in_period.append(period.time_seconds.max())

    actions['time_seconds'] = actions.apply(
        lambda x: x['time_seconds'] + sum(last_action_in_period[: x['period_id'] - 1]), axis=1)

    group_by_time_chunks = actions.groupby((actions['time_seconds'] // 300).astype(int))

    vaep_mean = {}
    for index, time_chunk in group_by_time_chunks:
        label = "{} to {}".format(index * 5, (index + 1) * 5)
        vaep_mean[label] = [time_chunk['vaep_value'].mean()]

    return pd.DataFrame(vaep_mean)


def get_average_rating_during_game(games: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    game_ratings = []
    for game_id in tqdm(list(games.game_id), desc="Get rating during games: "):
        game_ratings.append(rating_during_game(actions[(actions.game_id == game_id) & (actions.period_id != 5)]))

    game_ratings = pd.concat(game_ratings).reset_index(drop=True)
    return game_ratings.describe()
