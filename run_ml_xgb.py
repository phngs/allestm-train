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

import xgboost as xgb

from mllib.db import db_store_predicted, db_create_experiments_table, db_model_exists, get_num_folds, \
    get_ids_and_lengths, db_store_model, db_get_loss
from mllib.features.continuous import Continuous
from mllib.features.categorical import Categorical
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


def train_model(params, target, train_data, valid_data):
    """
    Train the model.

    Args:
        params: Model parameters.
        target: Target feature instance.
        train_data: Training dataset.
        valid_data: Validation dataset.

    Returns:
        Trained model.
    """
    if isinstance(target, (Binary, Continuous)):
        y = train_data.data[train_data.target_names].values.squeeze()
        y_valid = valid_data.data[valid_data.target_names].values.squeeze()
    else:
        y = np.argmax(train_data.data[train_data.target_names].values, axis=1).squeeze()
        y_valid = np.argmax(valid_data.data[valid_data.target_names].values, axis=1).squeeze()

    history = {}
    model = xgb.train({
        'objective': params['objective'],
        'eta': params['eta'],
        'max_depth': params['max_depth'],
        'min_child_weight': params['min_child_weight'],
        'gamma': params['gamma'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'lambda': params['lambda'],
        'alpha': params['alpha'],
        'nthread': params['nthread'],
        'num_class': params['num_class'],
        'silent': params['silent']
    }, xgb.DMatrix(train_data.data[train_data.feature_names], label=y),
              num_boost_round=10000,
              evals=((xgb.DMatrix(valid_data.data[valid_data.feature_names], label=y_valid), 'valid'),),
              early_stopping_rounds=10, evals_result=history, verbose_eval=True,
              callbacks=None)

    return model, history


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

    return model.eval(xgb.DMatrix(valid_data.data[valid_data.feature_names], label=y), iteration=model.best_iteration)


def test_model(model, test_data):
    """
    Run predictions.

    Args:
        model: Trained model.
        test_data: Test dataset.

    Returns:
        List of NumPy arrays containing predictions.
    """
    return model.predict(xgb.DMatrix(test_data.data[test_data.feature_names]), ntree_limit=model.best_ntree_limit)


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
        'eta': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': [6],
        'min_child_weight': [1],
        'gamma': [0],
        'lambda': [1], # REMOVE
        'alpha': [0] # REMOVE
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

    for it in range(1000):
        log.info(f'Iteration {it}')

        window = random.choice(params_map['window'])
        eta = random.choice(params_map['eta'])
        max_depth = random.choice(params_map['max_depth'])
        min_child_weight = random.choice(params_map['min_child_weight'])
        gamma = random.choice(params_map['gamma'])
        subsample = random.choice(params_map['subsample'])
        colsample_bytree = random.choice(params_map['colsample_bytree'])
        lambd = random.choice(params_map['lambda'])
        alpha = random.choice(params_map['alpha'])

        params = {
            'window': window,
            'eta': eta,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'lambda': lambd,
            'alpha': alpha,
            'nthread': 6,
            'silent': 1
        }

        if isinstance(target, Continuous):
            params['objective'] = 'reg:linear'
            params['num_class'] = 1
        elif isinstance(target, Binary):
            params['objective'] = 'binary:logistic'
            params['num_class'] = 1
        elif isinstance(target, Categorical):
            params['objective'] = 'multi:softprob'
            params['num_class'] = len(target)

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

            log.info('Training model.')
            model, history = train_model(params, target, train_data, valid_data)

            history['importance'] = model.get_score(importance_type='gain')

            valid_loss = model.best_score
            #valid_loss = eval_model(model, target, valid_data)

            del train_data

            improvements.append([fold, prev_loss, valid_loss, prev_loss is None or valid_loss < prev_loss])

            if prev_loss is None or valid_loss < prev_loss:
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
                db_store_model(conn, args.db_file, (lambda m, f: m.save_model(str(f))), args.experiment, fold, model, history, params, valid_loss)
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
