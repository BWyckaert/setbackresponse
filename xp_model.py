import json
import os
import pandas as pd
import numpy as np
import math
import xgboost
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

import utils

from typing import List, Optional, Dict, Any
from tqdm import tqdm


def compute_features_and_labels(games: pd.DataFrame, actions: pd.DataFrame,
                                train_competitions: List[str]) -> pd.DataFrame:
    passlike = ['pass', 'cross', 'freekick_crossed', 'freekick_short', 'corner_crossed', 'corner_short', 'clearance',
                'throw_in']
    passes = actions[actions['type_name'].isin(passlike)]
    passes['result_id'] = passes.apply(lambda x: 1 if (x['type_name'] == 'clearance') else x['result_id'], axis=1)
    passes['result_name'] = passes.apply(lambda x: 'success' if (x['type_name'] == 'clearance') else x['result_name'],
                                         axis=1)
    passes['result_id'] = passes.apply(lambda x: 0 if (x['result_id'] == 2) else x['result_id'], axis=1)
    passes['result_name'] = passes.apply(lambda x: 'fail' if (x['result_id'] == 2) else x['result_name'],
                                         axis=1)

    passes_grouped_by_competition_id = passes.merge(games[['game_id', 'competition_id', 'competition_name']],
                                                    on='game_id').groupby('competition_id')

    root = os.path.join(os.getcwd(), 'wyscout_data')

    all_passes = []
    for competition_id, passes in passes_grouped_by_competition_id:
        with open(os.path.join(root, utils.index.at[competition_id, 'db_events']), 'rt', encoding='utf-8') as we:
            events = pd.DataFrame(json.load(we))

        passes = passes.merge(events[['id', 'subEventName', 'tags']], left_on='original_event_id', right_on='id')
        duellike = ['Air duel', 'Ground attacking duel', 'Ground defending duel', 'Ground loose ball duel']
        passes = passes[~passes['subEventName'].isin(duellike)]
        passes['tags'] = passes.apply(lambda x: [d['id'] for d in x['tags']], axis=1)
        passes['through_ball'] = passes.apply(lambda x: 901 in x['tags'], axis=1)
        passes['degree'] = passes.apply(
            lambda x: abs(math.degrees(math.atan2(x['end_y'] - x['start_y'], x['end_x'] - x['start_x']))), axis=1)
        passes = passes[
            ['id', 'competition_name', 'start_x', 'start_y', 'end_x', 'end_y', 'degree', 'total_seconds', 'score_diff',
             'player_diff', 'through_ball', 'bodypart_name', 'subEventName', 'position', 'result_id']
        ].set_index('id', drop=True)

        all_passes.append(passes)

    all_passes = pd.concat(all_passes)
    all_passes = all_passes.dropna()
    all_passes = all_passes.rename(columns={'bodypart_name': 'bodypart', 'subEventName': 'type'})
    X = all_passes.drop(columns={'result_id'})
    y = all_passes[['competition_name', 'result_id']]

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X[['bodypart', 'type', 'position']])
    onehot = pd.DataFrame(data=encoder.transform(X[['bodypart', 'type', 'position']]).toarray(),
                          columns=encoder.get_feature_names_out(['bodypart', 'type', 'position']),
                          index=X.index).astype(bool)
    X = pd.concat([X[['competition_name', 'start_x', 'start_y', 'end_x', 'end_y', 'degree', 'total_seconds',
                      'score_diff', 'player_diff', 'through_ball']], onehot], axis=1)

    X_train = X[X['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})
    X_test = X[~X['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})
    y_train = y[y['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})
    y_test = y[~y['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})

    root = os.path.join(os.getcwd(), 'xp_model')
    data_h5 = os.path.join(root, "data.h5")
    with pd.HDFStore(data_h5) as datastore:
        datastore["X_train"] = X_train
        datastore["X_test"] = X_test
        datastore["y_train"] = y_train
        datastore["y_test"] = y_test


def fit(X: pd.DataFrame, y: pd.DataFrame, val_size=0.20,
        tree_params=dict(n_estimators=150, max_depth=6, objective='binary:logistic'),
        fit_params=dict(eval_metric='auc', verbose=True)) -> xgboost.XGBClassifier:
    nb_states = X.shape[0]
    index = np.random.permutation(nb_states)
    train_index = index[:math.floor(nb_states * (1 - val_size))]
    val_index = index[(math.floor(nb_states * (1 - val_size)) + 1):]

    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]

    eval_set = [(X_val, y_val)] if val_size > 0 else None

    if eval_set is not None:
        val_params = dict(early_stopping_rounds=10, eval_set=eval_set)
        fit_params = {**fit_params, **val_params}

    model = xgboost.XGBClassifier(**tree_params)
    return model.fit(X_train, y_train, **fit_params)


def train_model():
    root = os.path.join(os.getcwd(), 'xp_model')
    data_h5 = os.path.join(root, "data.h5")
    with pd.HDFStore(data_h5) as datastore:
        X_train = datastore["X_train"]
        X_test = datastore["X_test"]
        y_train = datastore["y_train"]
        y_test = datastore["y_test"]

    model = fit(X_train, y_train)
    model.save_model("xp_model/xP_XGBoost.txt")
    evaluate(model, X_test, y_test)


def evaluate(model: xgboost.XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_hat = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print("Confusion matrix", metrics.confusion_matrix(y_test, y_hat))
    print("Accuracy", metrics.accuracy_score(y_test, y_hat))
    print("Recall", metrics.recall_score(y_test, y_hat))
    print("Precision", metrics.precision_score(y_test, y_hat))
    print("ROC_AUC", metrics.roc_auc_score(y_test, y_prob))