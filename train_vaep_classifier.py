import math
import os
import warnings
import pandas as pd
import socceraction.atomic.vaep.features as afs
import socceraction.vaep.features as fs
import numpy as np

from socceraction.atomic.vaep.base import AtomicVAEP
from socceraction.vaep.base import VAEP
from tqdm import tqdm
from typing import List

warnings.filterwarnings('ignore')


def _compute_features(games: pd.DataFrame, spadl_h5: str, features_h5: str, vaep: VAEP):
    """
    Transform the actions in the given games to the feature-based representation of game states and
    stores these representations in the given features_h5 file.

    :param vaep: a vaep object
    :param games: a games dataframe
    :param spadl_h5: location where the spadl.h5 file is stored
    :param features_h5: location to store the features.h5 file
    """
    for game in tqdm(list(games.itertuples()), desc=f"Generating and storing features in {features_h5}"):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        X = vaep.compute_features(game, actions)
        X.to_hdf(features_h5, f"game_{game.game_id}")


def _compute_labels(games: pd.DataFrame, spadl_h5: str, labels_h5: str, vaep: VAEP):
    """
    Computes and stores the labels of the actions in the given games in the given labels_h5 file.

    :param vaep: a vaep object
    :param games: a games dataframe
    :param spadl_h5: location where the spadl.h5 file is stored
    :param labels_h5: location to store the labels.h5 file
    """
    for game in tqdm(list(games.itertuples()), desc=f"Computing and storing labels in {labels_h5}"):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        Y = vaep.compute_labels(game, actions)
        Y.to_hdf(labels_h5, f"game_{game.game_id}")


def _compute_features_and_labels(spadl_h5: str, features_h5: str, labels_h5: str, games: pd.DataFrame, vaep: VAEP):
    """
    Computes and stores the features and labels required to train a VAEP model
    """
    _compute_features(games, spadl_h5, features_h5, vaep)
    _compute_labels(games, spadl_h5, labels_h5, vaep)


def _read_features_and_labels(games: pd.DataFrame, features_h5: str, labels_h5: str, vaep: VAEP, _fs) -> (
        pd.DataFrame, pd.DataFrame):
    """
    Reads in features and labels stored in the features.h5 and labels.h5 files.

    :param _fs: package indication (this is very probably bad practice, but it works for now)
    :param vaep: a vaep object
    :param labels_h5: location where the labels.h5 file is stored
    :param features_h5: location where the features.h5 file is stored
    :param games: the games for which features are stored
    :return: a feature dataframe and a label dataframe
    """
    feature_names = _fs.feature_column_names(vaep.xfns, vaep.nb_prev_actions)

    features = []
    for game_id in tqdm(games.game_id, desc="Selecting features"):
        feature_i = pd.read_hdf(features_h5, f"game_{game_id}")
        features.append(feature_i[feature_names])
    features = pd.concat(features).reset_index(drop=True)

    labels_columns = ["scores", "concedes"]
    labels = []
    for game_id in tqdm(games.game_id, desc="Selecting labels"):
        label_i = pd.read_hdf(labels_h5, f"game_{game_id}")
        labels.append(label_i[labels_columns])
    labels = pd.concat(labels).reset_index(drop=True)
    return features, labels


def _evaluate(vaep: VAEP, test_features: pd.DataFrame, test_labels: pd.DataFrame):
    """
    Print some evaluation metrics.

    :param vaep: a vaep object
    :param test_features: a dataframe representing the features of the testset
    :param test_labels: a dataframe representing the labels of the testset
    """
    print()
    print()
    evaluation = vaep.score(test_features, test_labels)
    for col in evaluation:
        print(f"### Label: {col} ###")
        print(f"Brier score loss:  {evaluation[col]['brier']}")
        print(f"ROC score:  {evaluation[col]['auroc']}")
        print()
    print()


def _rate_actions(games: pd.DataFrame, spadl_h5: str, vaep: VAEP) -> pd.DataFrame:
    """
    Returns a dataframe containing the predicted action ratings for all the actions in the given games.

    :param games: the games for which the action ratings of the actions should be calculated
    :param spadl_h5: location where the spadl.h5 file is stored
    :param vaep: a vaep object
    :return:
    """
    predictions = []
    for game in tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        predictions.append(vaep.rate(game, actions))

    predicted_labels = pd.concat(predictions)
    return predicted_labels.reset_index(drop=True)


def _store_predictions(games: pd.DataFrame, spadl_h5: str, predictions_h5: str, predicted_action_ratings: pd.DataFrame):
    """
    Stores the predicted action ratings for the actions in the given games in the predictions_h5 file.

    :param games: the games for which the predicted action ratings should be stored
    :param spadl_h5: location where the spadl.h5 file is stored
    :param predictions_h5: location to store the precticted_labels
    :param predicted_action_ratings: dataframe containing the prected action ratings for all the actions in the given
           games
    """
    actions = []
    for game_id in tqdm(games.game_id, "Loading game ids: "):
        action_i = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
        actions.append(action_i[["game_id"]])
    actions = pd.concat(actions)
    actions = actions.reset_index(drop=True)

    grouped_predictions = pd.concat([actions, predicted_action_ratings], axis=1).groupby("game_id")
    for k, df in tqdm(grouped_predictions, desc="Saving predictions per game"):
        df = df.reset_index(drop=True)
        df[predicted_action_ratings.columns].to_hdf(predictions_h5, f"game_{int(k)}")


def train_model(train_competitions: List[str], test_competitions: List[str], atomic=True, learners=["xgboost"],
                print_eval=False, compute_features_labels=True, validation_size=0.25) -> VAEP:
    """
    Returns a trained vaep model (trained with the given learner) and stores the action ratings

    :param validation_size: 
    :param test_competitions:
    :param train_competitions:
    :param compute_features_labels: boolean flag indicating whether to recompute the features and labels
    :param atomic: boolean flag indicating whether or not the actions should be atomic
    :param learners: the learner to be used
    :param print_eval: boolean flag indicating whether or not evaluation metrics should be printed
    :return: a trained vaep model
    """
    if atomic:
        vaep = AtomicVAEP()
        datafolder = "atomic_data"
        _fs = afs
    else:
        vaep = VAEP()
        datafolder = "default_data"
        _fs = fs

    spadl_h5 = os.path.join(datafolder, "spadl.h5")
    features_h5 = os.path.join(datafolder, "features.h5")
    labels_h5 = os.path.join(datafolder, "labels.h5")
    predictions_h5 = os.path.join(datafolder, "predictions.h5")
    games = pd.read_hdf(spadl_h5, "games")
    competitions = pd.read_hdf(spadl_h5, "competitions")
    games = games.merge(competitions, how='left')

    if compute_features_labels:
        _compute_features_and_labels(spadl_h5, features_h5, labels_h5, games, vaep)

    train_games = games[games.competition_name.isin(train_competitions)]
    test_games = games[games.competition_name.isin(test_competitions)]

    train_features, train_labels = _read_features_and_labels(train_games, features_h5, labels_h5, vaep, _fs)
    test_features, test_labels = _read_features_and_labels(test_games, features_h5, labels_h5, vaep, _fs)

    for learner in learners:
        vaep.fit(train_features, train_labels, learner=learner, val_size=validation_size)
        if print_eval:
            _evaluate(vaep, test_features, test_labels)

    # _store_predictions(test_games, spadl_h5, predictions_h5, _rate_actions(test_games, spadl_h5, vaep))
    return vaep
