import json
import os
import pandas as pd
import math
import socceraction.atomic.spadl as aspadl
import socceraction.spadl as spadl
import utils

from typing import List
from tqdm import tqdm


def get_data(games: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    actions_grouped_by_game_id = actions.groupby('game_id')
    passlike = ['pass', 'cross', 'freekick_crossed', 'freekick_short', 'corner_crossed', 'corner_short', 'clearance',
                'throw_in']
    passes = actions[actions['type_name'].isin(passlike)]
    passes['result_id'] = passes.apply(lambda x: 1 if (x['type_name'] == 'clearance') else x['result_id'], axis=1)
    passes['result_name'] = passes.apply(lambda x: 'success' if (x['type_name'] == 'clearance') else x['result_name'],
                                         axis=1)

    passes_grouped_by_competition_id = passes.merge(games[['game_id', 'competition_id']], on='game_id').groupby(
        'competition_id')

    root = os.path.join(os.getcwd(), 'wyscout_data')

    all_passes = []
    for competition_id, passes in passes_grouped_by_competition_id:
        with open(os.path.join(root, utils.index.at[competition_id, 'db_events']), 'rt', encoding='utf-8') as we:
            events = pd.DataFrame(json.load(we))

        passes = passes.merge(events[['id', 'subEventName', 'tags']], left_on='original_event_id', right_on='id')
        passes['tags'] = passes.apply(lambda x: [d['id'] for d in x['tags']], axis=1)
        passes['through_ball'] = passes.apply(lambda x: 901 in x['tags'], axis=1)
        passes['degree'] = passes.apply(
            lambda x: abs(math.degrees(math.atan2(x['end_y'] - x['start_y'], x['end_x'] - x['start_x']))), axis=1)
        passes = passes[
            ['id', 'period_id', 'time_seconds', 'start_x', 'start_y', 'end_x', 'end_y', 'degree', 'score_diff',
             'bodypart_name', 'subEventName', 'position', 'through_ball', 'result_id']].set_index('id', drop=True)

        print(passes.head(300))

