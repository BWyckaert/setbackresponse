import json
import os
import warnings
from typing import List

import matplotsoccer as mps
import pandas as pd
from tqdm import tqdm
import socceraction.spadl as spadl

import utils
import xp_model

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)


def plot_actions(actions: pd.DataFrame, labels: pd.DataFrame, label_title: List[str]):
    mps.actions(location=actions[['start_x', 'start_y', 'end_x', 'end_y']],
                action_type=actions['type_name'],
                color='green',
                team=actions['team_name'],
                figsize=9,
                label=labels,
                labeltitle=label_title,
                legloc='top',
                show=True,
                zoom=22)


def plot_chadli_goal(actions: pd.DataFrame):
    game_id = 2058007
    game_actions = actions[actions['game_id'] == game_id].reset_index(drop=True)
    phase = game_actions.iloc[1384:1391]
    phase.loc[1384, 'type_name'] = 'corner'
    phase.loc[1384, 'end_x'] = 13.65
    phase.loc[1384, 'end_y'] = 29.24
    print(phase)

    labels = phase[['nickname', 'type_name', 'vaep_value']]
    label_title = ['Player', 'Action', 'VAEP-value']

    plot_actions(phase, labels, label_title)


def plot_pass_sequence(actions: pd.DataFrame, games: pd.DataFrame):
    game_id = 2058007
    game_actions = actions[actions['game_id'] == game_id].reset_index(drop=True)
    phase = game_actions.iloc[993:1003]
    exp_pass_acc = xp_model.predict_for_actions(game_actions, games)
    phase = phase.merge(exp_pass_acc, left_on='original_event_id', right_index=True)
    phase['risk'] = 1 - phase['exp_accuracy']
    phase.loc[phase['type_name'] == 'interception', 'risk'] = 0
    phase['accurate'] = phase.apply(lambda x: 'yes' if x['result_name'] == 'success' else 'no', axis=1)
    phase['time_on_ball'] = (phase.shift(-1)['time_seconds'] - phase['time_seconds']) / 2
    print(phase[:-1])

    labels = phase[['nickname', 'type_name', 'risk', 'accurate', 'time_on_ball']]
    label_title = ['Player', 'Action', 'Risk', 'Accurate', 'Time on ball']

    plot_actions(phase[:-1], labels, label_title)


def main():
    spadl_h5 = 'default_data/spadl.h5'
    predictions_h5 = 'default_data/predictions.h5'

    # Get world cup events
    with open(os.path.join('wyscout_data', utils.competition_index.loc[28, 'db_events']), 'rt', encoding='utf-8') as wm:
        events = pd.DataFrame(json.load(wm))

    all_actions = []
    with pd.HDFStore(spadl_h5) as store:
        games = (
            store['games']
                .merge(store['teams'].add_prefix('home_'), how='left')
                .merge(store['teams'].add_prefix('away_'), how='left')
                .merge(store["competitions"], how='left')
        )
        players = store['players']
        teams = store['teams']
        games = games[games['competition_name'].isin(['World Cup'])]

        for game_id in tqdm(games.game_id, "Collecting all actions: "):
            actions = store['actions/game_{}'.format(game_id)]
            actions = actions[actions['period_id'] != 5]
            actions = (
                spadl.add_names(actions).merge(players, how='left').merge(teams, how='left').sort_values(
                    ['game_id', 'period_id', 'action_id'])
            )
            actions = utils.add_goal_diff(actions)
            actions = utils.add_player_diff(actions, game_id, events)
            values = pd.read_hdf(predictions_h5, 'game_{}'.format(game_id))
            all_actions.append(pd.concat([actions, values], axis=1))

    all_actions = pd.concat(all_actions)
    all_actions = utils.add_total_seconds(all_actions, games)

    plot_pass_sequence(all_actions, games)


if __name__ == '__main__':
    main()
