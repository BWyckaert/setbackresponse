import os
import random
import warnings
import pandas as pd
import socceraction.atomic.vaep.features as afs
import socceraction.vaep.features as fs
import sklearn.calibration as cal
import matplotlib.pyplot as plt

from socceraction.atomic.vaep.base import AtomicVAEP
from socceraction.vaep.base import VAEP
from tqdm import tqdm
from typing import List, Dict

import utils

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
    for game in tqdm(list(games.itertuples()), desc="Generating and storing features in {}".format(features_h5)):
        actions = pd.read_hdf(spadl_h5, 'actions/game_{}'.format(game.game_id))
        actions = actions[actions['period_id'] != 5]
        X = vaep.compute_features(game, actions)
        X.to_hdf(features_h5, 'game_{}'.format(game.game_id))


def _compute_labels(games: pd.DataFrame, spadl_h5: str, labels_h5: str, vaep: VAEP):
    """
    Computes and stores the labels of the actions in the given games in the given labels_h5 file.

    :param vaep: a vaep object
    :param games: a games dataframe
    :param spadl_h5: location where the spadl.h5 file is stored
    :param labels_h5: location to store the labels.h5 file
    """
    for game in tqdm(list(games.itertuples()), desc="Computing and storing labels in {}".format(labels_h5)):
        actions = pd.read_hdf(spadl_h5, 'actions/game_{}'.format(game.game_id))
        actions = actions[actions['period_id'] != 5]
        y = vaep.compute_labels(game, actions)
        y.to_hdf(labels_h5, 'game_{}'.format(game.game_id))


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

    :param _fs: package indication
    :param vaep: a vaep object
    :param labels_h5: location where the labels.h5 file is stored
    :param features_h5: location where the features.h5 file is stored
    :param games: the games for which features are stored
    :return: a feature dataframe and a label dataframe
    """
    feature_column_names = _fs.feature_column_names(vaep.xfns, vaep.nb_prev_actions)

    all_features = []
    for game_id in tqdm(games.game_id, desc="Selecting features"):
        features = pd.read_hdf(features_h5, 'game_{}'.format(game_id))
        all_features.append(features[feature_column_names])
    all_features = pd.concat(all_features).reset_index(drop=True)

    label_column_names = ['scores', 'concedes']
    all_labels = []
    for game_id in tqdm(games.game_id, desc="Selecting labels"):
        labels = pd.read_hdf(labels_h5, 'game_{}'.format(game_id))
        all_labels.append(labels[label_column_names])
    all_labels = pd.concat(all_labels).reset_index(drop=True)

    return all_features, all_labels


def _print_eval(predictions: Dict[str, Dict[str, float]]):
    """
    Print brier and ROC score.

    :param predictions: the predictions of the model
    """
    print()
    print()
    for col in predictions:
        print("### Label: {} ###".format(col))
        print("Brier score loss: {}".format(predictions[col]['brier']))
        print("ROC score: {}".format(predictions[col]['auroc']))
        print()
    print()


def _store_eval(predictions: Dict[str, Dict[str, float]], training_set: List[str],
                learner: str, atomic: str, val_size: int):
    """
    Store brier and ROC score for the given arguments.

    :param predictions: the predictions of the model
    :param training_set: list of the competitions used in the training set
    :param learner: the learner used to obtain test_labels
    :param atomic: a boolean indicating whether or not the atomic format was used
    :param val_size: the size of the validation set
    """
    with open('results/vaep_tests.txt', 'a') as f:
        f.write("-------------------------------------------------------------------------------------------------- \n")
        f.write("Atomic: {} \n".format(atomic))
        f.write("Learner: {} \n".format(learner))
        f.write("Training set: {} \n".format(training_set))
        f.write("Validation size: {} \n".format(str(val_size)))
        for col in predictions:
            f.write("\n")
            f.write("### Label: {} ### \n".format(col))
            f.write("Brier score loss:  {} \n".format(predictions[col]['brier']))
            f.write("ROC score:  {} \n".format(predictions[col]['auroc']))
        f.write("\n \n")


def _rate_actions(games: pd.DataFrame, spadl_h5: str, vaep: VAEP) -> pd.DataFrame:
    """
    Rate all the actions in the given games with the vaep object.

    :param games: the games for which the action ratings of the actions should be calculated
    :param spadl_h5: location where the spadl.h5 file is stored
    :param vaep: a vaep object
    :return: a dataframe containing the predictions of the model
    """
    predictions = []
    for game in tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(spadl_h5, 'actions/game_{}'.format(game.game_id))
        actions = actions[actions['period_id'] != 5]
        rating = vaep.rate(game, actions)
        rating['game_id'] = game.game_id
        predictions.append(rating)

    predictions = pd.concat(predictions).reset_index(drop=True)

    return predictions


def _store_predictions(predictions_h5: str, predictions: pd.DataFrame):
    """
    Stores the predicted action ratings for the actions in the given games in the predictions_h5 file.

    :param predictions_h5: location to store the precticted_labels
    :param predictions: dataframe containing the prected action ratings for all the actions in the given
           games
    """
    column_names = ['offensive_value', 'defensive_value', 'vaep_value']
    grouped_predictions = predictions.groupby('game_id')
    for game_id, ratings in tqdm(grouped_predictions, desc="Saving predictions per game"):
        ratings = ratings.reset_index(drop=True)
        ratings[column_names].to_hdf(predictions_h5, 'game_{}'.format(int(game_id)))


def plot_calibration(vaep: VAEP, X: pd.DataFrame, y: pd.DataFrame):
    """
    Plot the calibration curves for both labels

    :param vaep: a vaep object
    :param X: the test features
    :param y: the test labels
    """
    y_hat = vaep._estimate_probabilities(X)

    cal.CalibrationDisplay.from_predictions(y_true=y['scores'], y_prob=y_hat['scores'], n_bins=10,
                                            name="Calibration of scores label")
    plt.show()

    cal.CalibrationDisplay.from_predictions(y_true=y['concedes'], y_prob=y_hat['concedes'], n_bins=10,
                                            name="Calibration of concedes label")
    plt.show()


def train_model(train_competitions: List[str], test_competitions: List[str], atomic=True, learner="xgboost",
                print_eval=False, store_eval=False, store_pred=False, plot_cal=False, compute_features_labels=False,
                validation_size=0.25, tree_params=None, fit_params=None) -> VAEP:
    """
    Returns a trained vaep model trained with the given learner

    :param plot_cal: plot the calibration curves of the labels
    :param fit_params: fit parameters for the learner
    :param tree_params: tree parameters for the learner
    :param store_pred: store predictions of the model
    :param store_eval: store evaluations of the model
    :param validation_size: the size of the validation set
    :param test_competitions: the test competitions
    :param train_competitions: the train competitions
    :param compute_features_labels: recompute the features and labels
    :param atomic: atomic actions or default
    :param learner: the learner to be used
    :param print_eval: print evaluation metrics
    :return: a trained vaep model
    """
    if atomic:
        _fs = afs
        xfns = [
            _fs.actiontype,
            _fs.actiontype_onehot,
            _fs.bodypart,
            _fs.bodypart_onehot,
            # _fs.time,
            _fs.team,
            _fs.time_delta,
            _fs.location,
            _fs.polar,
            _fs.movement_polar,
            _fs.direction,
            _fs.goalscore,
        ]
        vaep = AtomicVAEP(xfns=xfns)
        datafolder = 'atomic_data'
    else:
        _fs = fs
        xfns = [
            _fs.actiontype_onehot,
            _fs.result_onehot,
            _fs.actiontype_result_onehot,
            _fs.bodypart_onehot,
            # _fs.time,
            _fs.startlocation,
            _fs.endlocation,
            _fs.startpolar,
            _fs.endpolar,
            _fs.movement,
            _fs.team,
            _fs.time_delta,
            _fs.space_delta,
            _fs.goalscore,
        ]
        vaep = VAEP(xfns=xfns)
        datafolder = 'default_data'

    spadl_h5 = os.path.join(datafolder, 'spadl.h5')
    features_h5 = os.path.join(datafolder, 'features.h5')
    labels_h5 = os.path.join(datafolder, 'labels.h5')
    predictions_h5 = os.path.join(datafolder, 'predictions.h5')

    with pd.HDFStore(spadl_h5) as store:
        games = store['games']
        competitions = store['competitions']
    games = games.merge(competitions, how='left')

    if compute_features_labels:
        _compute_features_and_labels(spadl_h5, features_h5, labels_h5, games, vaep)

    train_games = games[games.competition_name.isin(train_competitions)]
    test_games = games[games.competition_name.isin(test_competitions)]

    train_features, train_labels = _read_features_and_labels(train_games, features_h5, labels_h5, vaep, _fs)
    test_features, test_labels = _read_features_and_labels(test_games, features_h5, labels_h5, vaep, _fs)

    vaep.fit(train_features, train_labels, learner=learner, val_size=validation_size, tree_params=tree_params,
             fit_params=fit_params)

    if print_eval or store_eval:
        predictions = vaep.score(test_features, test_labels)

        if print_eval:
            _print_eval(predictions)

        if store_eval:
            atomic_str = 'yes' if atomic else 'no'
            _store_eval(predictions=predictions, training_set=train_competitions,
                        learner=learner, atomic=atomic_str, val_size=validation_size)

    if plot_cal:
        plot_calibration(vaep, test_features, test_labels)

    # if store_pred:
    #     _store_predictions(predictions_h5, _rate_actions(test_games, spadl_h5, vaep))

    return vaep


def compare_models():
    learners = ['xgboost', 'catboost', 'lightgbm']

    tc1 = ['World Cup']
    tc2 = ['World Cup', 'European Championship']
    tc3 = ['German first division']
    tc4 = ['English first division']
    tc5 = ['World Cup', 'English first division']
    tc6 = ['English first division', 'German first division']

    tc = [tc1, tc2, tc3, tc4, tc5, tc6]

    test_competitions = ['Spanish first division', 'Italian first division', 'French first division']

    random.seed(0)

    for atomic in [True, False]:
        for learner in learners:
            for train_competitions in tc:
                train_model(train_competitions=train_competitions, test_competitions=test_competitions, atomic=atomic,
                            learner=learner, print_eval=False, store_eval=True, store_pred=False, plot_cal=False,
                            compute_features_labels=False, validation_size=0.25)


def main():
    train_competitions = utils.train_competitions
    test_competitions = utils.test_competitions
    learner = "xgboost"
    validation_size = 0.25
    tree_params = dict(n_estimators=100, max_depth=3)
    fit_params = dict(eval_metric='auc', verbose=True)

    random.seed(0)

    train_model(train_competitions=train_competitions, test_competitions=test_competitions, atomic=False,
                learner=learner, print_eval=True, store_eval=False, store_pred=True, plot_cal=True,
                compute_features_labels=False, validation_size=validation_size, tree_params=tree_params,
                fit_params=fit_params)


if __name__ == '__main__':
    main()
    # compare_models()
