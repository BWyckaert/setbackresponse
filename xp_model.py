import json
import os

import catboost
import pandas as pd
import numpy as np
import math
import xgboost
import lightgbm
import sklearn.calibration as cal
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import utils

from typing import List


def fix_passes(actions: pd.DataFrame) -> pd.DataFrame:
    # Only consider passlike actions
    passlike = ['pass', 'cross', 'freekick_crossed', 'freekick_short', 'corner_crossed', 'corner_short', 'clearance',
                'throw_in']
    passes = actions[actions['type_name'].isin(passlike)]

    # Clearance always successful
    passes['result_id'] = passes.apply(lambda x: 1 if (x['type_name'] == 'clearance') else x['result_id'], axis=1)
    passes['result_name'] = passes.apply(lambda x: 'success' if (x['type_name'] == 'clearance') else x['result_name'],
                                         axis=1)

    # Set offside result to fail
    passes['result_id'] = passes.apply(lambda x: 0 if (x['result_id'] == 2) else x['result_id'], axis=1)
    passes['result_name'] = passes.apply(lambda x: 'fail' if (x['result_id'] == 2) else x['result_name'],
                                         axis=1)

    return passes


def add_features(passes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    passes_grouped_by_competition_id = passes.merge(games[['game_id', 'competition_id', 'competition_name']],
                                                    on='game_id').groupby('competition_id')

    all_passes = []
    for competition_id, passes in passes_grouped_by_competition_id:
        # Get wyscout events
        with open(os.path.join("wyscout_data", utils.competition_index.at[competition_id, 'db_events']), 'rt',
                  encoding='utf-8') as we:
            events = pd.DataFrame(json.load(we))

        # Merge passes with wyscout events
        passes = passes.merge(events[['id', 'subEventName', 'tags']], left_on='original_event_id', right_on='id')

        # Remove duellike passes
        duellike = ['Air duel', 'Ground attacking duel', 'Ground defending duel', 'Ground loose ball duel']
        passes = passes[~passes['subEventName'].isin(duellike)]

        # Add through_ball feature
        passes['tags'] = passes.apply(lambda x: [d['id'] for d in x['tags']], axis=1)
        passes['through_ball'] = passes.apply(lambda x: 901 in x['tags'], axis=1)

        # Add distance feature
        passes['distance'] = passes.apply(
            lambda x: math.sqrt(pow(x['end_x'] - x['start_x'], 2) + pow(x['end_y'] - x['start_y'], 2)), axis=1)

        # Add degree to goal feature
        passes['degree'] = passes.apply(
            lambda x: abs(math.degrees(math.atan2(x['end_y'] - x['start_y'], x['end_x'] - x['start_x']))), axis=1)

        # Remove action columns that don't matter
        passes = passes[
            ['id', 'competition_name', 'start_x', 'start_y', 'end_x', 'end_y', 'degree', 'total_seconds', 'score_diff',
             'player_diff', 'through_ball', 'bodypart_name', 'subEventName', 'position', 'result_id']
        ].set_index('id', drop=True)

        all_passes.append(passes)

    all_passes = pd.concat(all_passes)
    all_passes = all_passes.dropna()
    all_passes = all_passes.rename(columns={'bodypart_name': 'bodypart', 'subEventName': 'type'})

    return all_passes


def compute_features_and_labels(games: pd.DataFrame, actions: pd.DataFrame,
                                train_competitions: List[str]) -> pd.DataFrame:
    # Fix passes and add features
    passes = fix_passes(actions)
    passes = add_features(passes, games)

    # Split passes into features and labels
    X = passes.drop(columns={'result_id'})
    y = passes[['competition_name', 'result_id']]

    # Get onehot encoding of categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X[['bodypart', 'type', 'position']])
    onehot = pd.DataFrame(data=encoder.transform(X[['bodypart', 'type', 'position']]).toarray(),
                          columns=encoder.get_feature_names_out(['bodypart', 'type', 'position']),
                          index=X.index).astype(bool)

    # Concat non-categorical (+ through_ball) with categorical features
    X = pd.concat([X[['competition_name', 'start_x', 'start_y', 'end_x', 'end_y', 'degree', 'total_seconds',
                      'score_diff', 'player_diff', 'through_ball']], onehot], axis=1)

    # Split data into train and test set
    X_train = X[X['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})
    X_test = X[~X['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})
    y_train = y[y['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})
    y_test = y[~y['competition_name'].isin(train_competitions)].drop(columns={'competition_name'})

    # Store data
    data_h5 = "xP_data/passes.h5"
    with pd.HDFStore(data_h5) as datastore:
        datastore["X_train"] = X_train
        datastore["X_test"] = X_test
        datastore["y_train"] = y_train
        datastore["y_test"] = y_test


def fit(learner, X: pd.DataFrame, y: pd.DataFrame, val_size=0.2, tree_params=None, fit_params=None):
    # Split into train and validation data, add random_state=0 for testing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

    eval_set = [(X_val, y_val)] if val_size > 0 else None

    # Use given learner to fit a model to the data
    if learner == "xgboost":
        if tree_params is None:
            tree_params = dict(n_estimators=100, max_depth=6, objective='binary:logistic')
        if fit_params is None:
            fit_params = dict(eval_metric='logloss', verbose=True)
        if eval_set is not None:
            val_params = dict(early_stopping_rounds=10, eval_set=eval_set)
            fit_params = {**fit_params, **val_params}
        model = xgboost.XGBClassifier(**tree_params)
        return model.fit(X_train, y_train, **fit_params)

    if learner == "catboost":
        if tree_params is None:
            tree_params = dict(eval_metric='BrierScore', loss_function='Logloss', iterations=100)
        if fit_params is None:
            is_cat_feature = [c.dtype.name == 'category' for (_, c) in X.iteritems()]
            fit_params = dict(cat_features=np.nonzero(is_cat_feature)[0].tolist(), verbose=True)
        if eval_set is not None:
            val_params = dict(early_stopping_rounds=10, eval_set=eval_set)
            fit_params = {**fit_params, **val_params}
        model = catboost.CatBoostClassifier(**tree_params)
        return model.fit(X, y, **fit_params)

    if learner == "lightgbm":
        if tree_params is None:
            tree_params = dict(n_estimators=100, max_depth=6, objective='binary')
        if fit_params is None:
            fit_params = dict(eval_metric='auc', verbose=True)
        if eval_set is not None:
            val_params = dict(early_stopping_rounds=10, eval_set=eval_set)
            fit_params = {**fit_params, **val_params}
        model = lightgbm.LGBMClassifier(**tree_params)
        return model.fit(X, y, **fit_params)


def train_model(filters=[], learner="xgboost", store=False):
    # Get the data
    data_h5 = "xP_data/passes.h5"
    with pd.HDFStore(data_h5) as datastore:
        X_train = datastore["X_train"]
        X_test = datastore["X_test"]
        y_train = datastore["y_train"]
        y_test = datastore["y_test"]

    # Filter unwanted features from the feature set
    for feature in filters:
        for column in X_train.columns:
            if feature in column:
                X_train = X_train.drop(columns={column})
                X_test = X_test.drop(columns={column})

    print("Number of features: ", X_train.shape[1])

    # Fit the model and save
    model = fit(learner, X_train, y_train)
    if learner == "xgboost":
        model.save_model("xP_data/xP_XGBoost.txt")
    if learner == "catboost":
        model.save_model("xP_data/xP_CatBoost.txt")
    if learner == "lightgbm":
        model.booster_.save_model("xP_data/xP_lightgbm.txt")

    # Print some evaluation metrics for the model that has just been fitted
    evaluate(learner, X_test, y_test)

    # Store predictions if needed
    if store:
        store_predictions(learner, X_test, filters)


def evaluate(learner, X_test: pd.DataFrame, y_test: pd.DataFrame, filters):
    for feature in filters:
        for column in X_test.columns:
            if feature in column:
                X_test = X_test.drop(columns={column})

    # Get the model and calculate y_hat and y_prob
    if learner == "xgboost":
        model = xgboost.XGBClassifier()
        model.load_model("xP_data/xP_XGBoost.txt")
        y_hat = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    if learner == "catboost":
        model = catboost.CatBoostClassifier()
        model.load_model("xP_data/xP_CatBoost.txt")
        y_hat = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    if learner == "lightgbm":
        model = lightgbm.Booster(model_file="xP_data/xP_lightgbm.txt")
        y_prob = model.predict(X_test)
        y_hat = np.round(y_prob)

    # Print some metrics
    print()
    print("Majority vote accuracy\n ", y_test.result_id.value_counts(normalize=True))
    print("Confusion matrix \n", metrics.confusion_matrix(y_test, y_hat))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_hat))
    print("Recall: ", metrics.recall_score(y_test, y_hat))
    print("Precision: ", metrics.precision_score(y_test, y_hat))
    print("ROC_AUC: ", metrics.roc_auc_score(y_test, y_prob))


def store_predictions(learner, X_test: pd.DataFrame, filters):
    # Filter unwanted features from the feature set
    for feature in filters:
        for column in X_test.columns:
            if feature in column:
                X_test = X_test.drop(columns={column})

    # Get model and calculate y_prob
    if learner == "xgboost":
        model = xgboost.XGBClassifier()
        model.load_model("xP_data/xP_XGBoost.txt")
        y_prob = model.predict_proba(X_test)[:, 1]
    if learner == "catboost":
        model = catboost.CatBoostClassifier()
        model.load_model("xP_data/xP_CatBoost.txt")
        y_prob = model.predict_proba(X_test)[:, 1]
    if learner == "lightgbm":
        model = lightgbm.Booster(model_file="xP_data/xP_lightgbm.txt")
        y_prob = model.predict(X_test)

    X_test["exp_accuracy"] = y_prob

    # Store exp_accuracy
    predictions_h5 = "xP_data/predictions.h5"
    with pd.HDFStore(predictions_h5) as predictionstore:
        predictionstore["predictions"] = X_test[["exp_accuracy"]]


def plot_calibration(X_test: pd.DataFrame, y_test: pd.DataFrame, filters, learner="xgboost"):
    # Filter unwanted features from the feature set
    for feature in filters:
        for column in X_test.columns:
            if feature in column:
                X_test = X_test.drop(columns={column})

    # Get model and calculate y_prob
    if learner == "xgboost":
        model = xgboost.XGBClassifier()
        model.load_model("xP_data/xP_XGBoost.txt")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_hat = model.predict(X_test)
    if learner == "catboost":
        model = catboost.CatBoostClassifier()
        model.load_model("xP_data/xP_CatBoost.txt")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_hat = model.predict(X_test)
    if learner == "lightgbm":
        model = lightgbm.Booster(model_file="xP_data/xP_lightgbm.txt")
        y_prob = model.predict(X_test)
        y_hat = np.round(y_prob)

    disp = cal.CalibrationDisplay.from_predictions(y_true=y_test, y_prob=y_prob, n_bins=10)

    plt.show()

