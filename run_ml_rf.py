"""

"""
import argparse
import sys
import logging
import sqlite3
from multiprocessing import Pool
import itertools
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib

from mllib.db import db_store_predicted, db_create_experiments_table, db_model_exists, get_num_folds, \
    get_ids_and_lengths, db_store_model, db_get_loss
from mllib.features.continuous import Continuous
from mllib.features.binary import Binary
from mllib.features.utils import name_to_feature
from mllib.retrievers import SQLRetriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Dataset:
    """
    Dataset container.
    """
    def __init__(self):
        """
        Constructor initializing the properties.
        """
        self.ids = None
        self.lengths = None
        self.feature_names = None
        self.target_names = None
        self.data = None


def parse_args(argv):
    """
    Parse the arguments.

    Args:
        argv: List of command line args.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('db_file', help='Sqlite3 database file.')
    parser.add_argument('dataset_name', help='Name of the dataset in the DB.')
    parser.add_argument('experiment', help='Name of the experiment.')

    parser.add_argument('-p', '--params', default=[], action='append', help='Param list in the form of param:val1,val2,...')

    parser.add_argument('-l', '--features', default=[], action='append', help='Names of features.')

    parser.add_argument('-t', '--target', help='Name of target.')

    parser.add_argument('-i', '--limit', type=int, default=None, help='Number of proteins to use (for testing purposes).')

    return parser.parse_args(argv)


def _calc_feature_values(args):
    """
    Helper function for parallelization of the feature calculation process.

    Args:
        args: List containing db_file, feature (class instance), and the identifier of the protein.

    Returns:
        NumPy Array with the shape (protein_length, feature_values).
    """
    db_file, feature, window, identifier = args
    conn = sqlite3.connect(db_file, timeout=120)
    retriever = SQLRetriever(conn, feature.query)
    result = feature.fit_transform(retriever.transform(identifier))
    conn.close()

    df = pd.DataFrame(result, columns=[f"{type(feature).__qualname__}_{i}" for i in range(result.shape[1])], index=[identifier for _ in range(len(result))])
    if window != 0:
        for column in df.columns:
            for win in range(-int((window - 1) / 2), int((window + 1) / 2)):
                if win != 0:
                    new_column = f"{column}_w{win}"
                    df[new_column] = df[column].shift(win)

    print('.', end='', flush=True)
    return df.ffill().bfill()


def calc_feature_values(db_file, identifiers, feature, window=0):
    """
    Calculate the given feature for a list of identifiers.

    Args:
        db_file: The sqlite3 DB file.
        identifiers: List of identifiers.
        feature: Feature as class instance.
        window: Window length.

    Returns:
        DataFrame array with the shape (num_of_identifiers * length, feature_values).
    """

    # Ugly creating the input for the Pool.map function.
    task_args = zip(
        itertools.repeat(db_file, len(identifiers)),
        itertools.repeat(feature, len(identifiers)),
        itertools.repeat(window, len(identifiers)),
        identifiers
    )

    with Pool() as pool:
        pool_results = pool.map(_calc_feature_values, task_args)

    print()

    return pd.concat(pool_results, axis=0)


def create_dataset(db_file, features, target, dataset_name, kind, fold, window=0, limit=None):
    """
    Create a dataset instance and fill it with the feature values.
    Args:
        db_file: Sqlite3 DB file.
        features: Feature instances.
        target: Target instance.
        dataset_name: Name of the dataset.
        kind: Kind of dataset (train, valid, test).
        fold: Number of the fold.

    Returns:
        Filled dataset instance.
    """
    conn = sqlite3.connect(db_file, timeout=120)
    dataset = Dataset()
    dataset.ids, dataset.lengths = get_ids_and_lengths(conn, dataset_name, kind, fold, limit=limit)
    conn.close()

    feature_dfs = []
    for feature in features:
        log.info(f'Calculating feature {type(feature)}.')
        feature_dfs.append(calc_feature_values(db_file, dataset.ids, feature, window=window))

    feature_df = pd.concat(feature_dfs, axis=1)
    dataset.feature_names = feature_df.columns

    log.info(f'Calculating target {type(target)}.')
    target_df = calc_feature_values(db_file, dataset.ids, target, window=0)
    dataset.target_names = target_df.columns

    dataset.data = pd.concat([feature_df, target_df], axis=1)

    return dataset


def build_model(target, params):
    """
    Build the model.

    Args:
        target: Instance of target class.
        params: RF params.

    Returns:
        Model.
    """
    if isinstance(target, Continuous):
        model_cls = RandomForestRegressor
    else:
        model_cls = RandomForestClassifier

    return model_cls(
        n_estimators=100, #params['n_estimators'],
        max_features=params['max_features'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        n_jobs=params['n_jobs'],
        verbose=params['verbose'],
        warm_start=True
    )


def train_model(max_trees, model, target, train_data, valid_data):
    """
    Train the model.

    Args:
        model: The compiled model.
        target: Target feature instance.
        train_data: Training dataset.
        valid_data: Validation dataset.

    Returns:
        Trained model.
    """

    if isinstance(target, (Binary, Continuous)):
        y = train_data.data[train_data.target_names].values.squeeze()
    else:
        y = np.argmax(train_data.data[train_data.target_names].values, axis=1).squeeze()

    prev_acc = -1.0
    model.fit(train_data.data[train_data.feature_names], y)
    n_trees = 100
    while n_trees <= max_trees:
        curr_acc = eval_model(model, target, valid_data)
        log.info(f'Current acc: {curr_acc} with {n_trees} trees.')
        if curr_acc > prev_acc:
            prev_acc = curr_acc
            n_trees += 100
            model.n_estimators = n_trees
            model.fit(train_data.data[train_data.feature_names], y)
        else:
            log.info('Loss does not improve.')
            break

    return model


def eval_model(model, target, valid_data):
    """
    Evaluate the model.

    Args:
        model: The compiled model.
        target: Target feature instance.
        valid_data: Validation dataset.

    Returns:
        Loss.
    """
    if isinstance(target, (Binary, Continuous)):
        y = valid_data.data[valid_data.target_names].values.squeeze()
    else:
        y = np.argmax(valid_data.data[valid_data.target_names].values, axis=1).squeeze()

    return model.score(valid_data.data[valid_data.feature_names], y)


def test_model(model, test_data):
    """
    Run predictions.

    Args:
        model: Trained model.
        test_data: Test dataset.

    Returns:
        List of NumPy arrays containing predictions.
    """
    if isinstance(model, RandomForestRegressor):
        return model.predict(test_data.data[test_data.feature_names])
    else:
        return model.predict_proba(test_data.data[test_data.feature_names])


def process_predictions(lengths, target, raw_predicted):
    """
    Removes padded positions from the predicted/observed values and
    calls inverse_transform if available.

    Args:
        lengths: Lengths of the entries in the dataset.
        target: Target feature instance.
        raw_predicted: Raw predicted/observed values.

    Returns:
        List of NumPy arrays: [(length, target_vals), ...].
    """
    if isinstance(target, Binary) and raw_predicted.shape[1] > 1:
        raw_predicted = raw_predicted[..., 1]

    if len(raw_predicted.shape) == 1:
        raw_predicted = raw_predicted[..., None]

    predicted = []
    start = 0
    for l in lengths:
        pred = raw_predicted[start:start+l]
        start += l
        if hasattr(target, 'inverse_transform'):
            pred = target.inverse_transform(pred)
        predicted.append(pred)

    return predicted


def main(argv):
    """
    Main method.

    Args:
        argv: Command line arguments.

    Returns:
        Nothing.
    """
    args = parse_args(argv)

    conn = sqlite3.connect(args.db_file, timeout=120)

    # Create the feature instances.
    features = [name_to_feature(x) for x in args.features]
    target = name_to_feature(args.target)

    num_folds = get_num_folds(conn, args.dataset_name)

    log.info('Creating experiments table if not exists.')
    db_create_experiments_table(conn)

    params_map = {
        'window': [21],
        'n_estimators': [2000],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1],
    }

    for p in args.params:
        param, v = p.split(':')
        val_strs = v.split(',')
        vals = []
        for val_str in val_strs:
            if val_str == 'None':
                vals.append[None]
            elif val_str.isdigit():
                vals.append(int(val_str))
            else:
                try:
                    vals.append(float(val_str))
                except ValueError:
                    vals.append(val_str)
        params_map[param] = vals

    random.shuffle(params_map['window'])
    random.shuffle(params_map['max_features'])
    random.shuffle(params_map['min_samples_leaf'])

    for window in params_map['window']:
        for max_features in params_map['max_features']:
                for min_samples_leaf in params_map['min_samples_leaf']:
                    params = {
                        'window': window,
                        'n_estimators': params_map['n_estimators'], #n_estimators,
                        'max_features': max_features,
                        'max_depth': None,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': 2,
                        'n_jobs': 6,
                        'verbose': 2
                    }

                    improvements = []

                    for fold in range(num_folds):
                        log.info(f'Starting with fold {fold}.')

                        prev_loss = None
                        if db_model_exists(conn, args.experiment, fold):
                            prev_loss = db_get_loss(conn, args.experiment, fold)
                            log.info(f'Experiment {args.experiment} fold {fold} already exists with loss {prev_loss}.')

                        log.info('Creating train dataset.')
                        train_data = create_dataset(args.db_file, features, target, args.dataset_name, 'train', fold, window=params['window'], limit=args.limit)

                        log.info('Creating valid dataset.')
                        valid_data = create_dataset(args.db_file, features, target, args.dataset_name, 'valid', fold, window=params['window'], limit=args.limit)

                        log.info('Building model.')
                        model = build_model(target, params)

                        log.info('Training model.')
                        model = train_model(max(params_map['n_estimators']), model, target, train_data, valid_data)
                        params['n_trees'] = len(model.estimators_)

                        importance = []
                        for feat, imp in zip(train_data.feature_names, model.feature_importances_):
                            importance.append([feat, imp])

                        importance = sorted(importance, key=lambda x: x[1])
                        history = {'importance': importance}

                        valid_loss = eval_model(model, target, valid_data)

                        del train_data

                        improvements.append([fold, prev_loss, valid_loss, prev_loss is None or valid_loss > prev_loss])

                        if prev_loss is None or valid_loss > prev_loss:
                            log.info(f'Validation loss improved. Previous loss was {prev_loss}, new loss is {valid_loss}.')

                            log.info('Predicting on valid data.')
                            valid_raw_predicted = test_model(model, valid_data)

                            log.info('Storing valid predictions.')
                            valid_raw_observed = valid_data.data[valid_data.target_names].values

                            valid_observed = process_predictions(valid_data.lengths, target, valid_raw_observed)
                            valid_predicted = process_predictions(valid_data.lengths, target, valid_raw_predicted)

                            db_store_predicted(conn, f'{args.experiment}_VALID_{fold}', target, valid_data.ids, valid_predicted, simulate=False)
                            db_store_predicted(conn, 'observed', target, valid_data.ids, valid_observed, simulate=False)

                            del valid_data

                            log.info('Creating test dataset.')
                            test_data = create_dataset(args.db_file, features, target, args.dataset_name, 'test', fold, window=params['window'], limit=args.limit)

                            log.info('Predicting on test data.')
                            test_raw_predicted = test_model(model, test_data)

                            log.info('Storing test predictions.')
                            test_raw_observed = test_data.data[test_data.target_names].values

                            test_observed = process_predictions(test_data.lengths, target, test_raw_observed)
                            test_predicted = process_predictions(test_data.lengths, target, test_raw_predicted)

                            db_store_predicted(conn, args.experiment, target, test_data.ids, test_predicted, simulate=False)
                            db_store_predicted(conn, 'observed', target, test_data.ids, test_observed, simulate=False)

                            del test_data

                            log.info('Creating independent_test dataset.')
                            independent_test_data = create_dataset(args.db_file, features, target, args.dataset_name, 'independent_test', 0, window=params['window'], limit=args.limit)

                            log.info('Predicting on independent_test data.')
                            independent_test_raw_predicted = test_model(model, independent_test_data)

                            log.info('Storing independent_test predictions.')
                            independent_test_raw_observed = independent_test_data.data[independent_test_data.target_names].values

                            independent_test_observed = process_predictions(independent_test_data.lengths, target, independent_test_raw_observed)
                            independent_test_predicted = process_predictions(independent_test_data.lengths, target, independent_test_raw_predicted)

                            db_store_predicted(conn, f'{args.experiment}_IND_TEST_{fold}', target, independent_test_data.ids, independent_test_predicted, simulate=False)
                            db_store_predicted(conn, 'observed', target, independent_test_data.ids, independent_test_observed, simulate=False)

                            del independent_test_data

                            log.info('Storing model.')
                            db_store_model(conn, args.db_file, (lambda m, f: joblib.dump(m, f, compress=3, protocol=-1)), args.experiment, fold, model, history, params, valid_loss)
                        else:
                            log.info(f'Validation loss did not improve. Previous loss was {prev_loss}, new loss is {valid_loss}.')

                    log.info(params)
                    log.info('Improvements:')
                    avg_prev = sum([x[1] for x in improvements if x[1] is not None]) / len(improvements)
                    avg_now = sum([x[2] for x in improvements]) / len(improvements)
                    avg_delta = avg_prev - avg_now
                    log.info(f'Average improvement:prev:{avg_prev:.3f}:now:{avg_now:.3f}:delta:{avg_delta:.3f}')
                    for imp in improvements:
                        log.info(imp)

    conn.close()

if __name__ == '__main__':
    main(sys.argv[1:])
