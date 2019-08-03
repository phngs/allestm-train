"""

"""
import argparse
import sys
import logging
import sqlite3
from multiprocessing import Pool
import itertools
import tempfile
import operator

import numpy as np
import pandas as pd

import lightgbm as lgb

from mllib.db import db_store_predicted, db_create_experiments_table, db_model_exists, get_num_folds, \
    get_ids_and_lengths, get_max_length, db_store_model
from mllib.features.categorical import Categorical
from mllib.features.binary import Binary
from mllib.features.utils import name_to_feature
from mllib.retrievers import SQLRetriever

from sklearn.metrics import log_loss, mean_squared_error

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
        self.ids = []
        self.lengths = []
        self.dataset = None


def parse_args(argv):
    """
    Parse the arguments.

    Args:
        argv: List of command line args.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("db_file", help="Sqlite3 database file.")
    parser.add_argument("dataset_name", help="Name of the dataset in the DB.")
    parser.add_argument("experiment", help="Name of the experiment.")

    parser.add_argument("-f", "--features", default=[], action="append", help="Names of input features")
    parser.add_argument("-t", "--target", help="Name of the target")
    parser.add_argument("-w", "--window", type=int, default=5, help="Window size")

    parser.add_argument("-i", "--limit", type=int, default=None, help="Number of proteins to use (for testing purposes).")

    return parser.parse_args(argv)


def _calc_feature_values(args):
    """
    Helper function for parallelization of the feature calculation process.

    Args:
        args: List containing db_file, feature (class instance), window size and the identifier of the protein.

    Returns:
        NumPy Array with the shape (protein_length, feature_values).
    """
    db_file, feature, window, identifier = args
    conn = sqlite3.connect(db_file)
    retriever = SQLRetriever(conn, feature.query)
    result = feature.fit_transform(retriever.transform(identifier))
    df = pd.DataFrame(result, columns=["{}_{}".format(type(feature).__qualname__, i) for i in range(result.shape[1])])

    if window != 0:
        for column in df.columns:
            for win in range(-int((window - 1) / 2), int((window + 1) / 2)):
                if win != 0:
                    new_column = "{}_w{}".format(column, win)
                    df[new_column] = df[column].shift(win)

    conn.close()
    print(".", end="", flush=True)
    return df.ffill().bfill()


def calc_feature_values(db_file, identifiers, feature, window=0):
    """
    Calculate the given feature for a list of identifiers.

    Args:
        db_file: The sqlite3 DB file.
        identifiers: List of identifiers.
        feature: Feature as class instance.
        window: Window size.

    Returns:
        NumPy array with the shape (num_of_identifiers, max_length, feature_values).
    """
    # Init the result array.
    #feature_values = np.zeros((len(identifiers), max_length, len(feature)))

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

    return pool_results


def create_dataset(db_file, features, target, window, dataset_name, kind, fold, limit=None, reference=None):
    """
    Create a dataset instance and fill it with the feature values.
    Args:
        db_file: Sqlite3 DB file.
        features: Feature instances.
        target: Target feature instances.
        window: Window size.
        dataset_name: Name of the dataset.
        kind: Kind of dataset (train, valid, test).
        fold: Number of the fold.

    Returns:
        Filled dataset instance.
    """
    conn = sqlite3.connect(db_file)
    dataset = Dataset()
    dataset.ids, dataset.lengths = get_ids_and_lengths(conn, dataset_name, kind, fold, limit=limit)
    conn.close()

    feature_values = []
    for feature in features:
        log.info("Calculating feature {}.".format(type(feature)))
        feature_values.append(calc_feature_values(db_file, dataset.ids, feature, window=window))

    log.info("Calculating target {}.".format(type(target)))
    target_values = calc_feature_values(db_file, dataset.ids, target)
    target_values = pd.concat(target_values)

    feature_names = []
    categorical_feature_names = []
    for i, feature in enumerate(features):
        if isinstance(feature, Categorical) or isinstance(feature, Binary):
            categorical_feature_names.extend(feature_values[i][0].columns)
        feature_names.extend(feature_values[i][0].columns)

    feature_values_concat = [pd.concat(feature) for feature in feature_values]

    dataset.dataset = lgb.Dataset(
        pd.concat(feature_values_concat, axis=1).values,
        feature_name=feature_names,
        categorical_feature=categorical_feature_names,
        label=target_values.values.squeeze(),
        reference=reference)

    return dataset


def train_model(train_data, valid_data, target, num_leaves, min_data_in_leaf, max_depth, limit=None):
    """
    Train the model.

    Args:
        train_data: Training dataset.
        valid_data: Validation dataset.
        target: Target class instance.
        limit: If true, do only one iteration.

    Returns:
        Trained model.
    """
    params = {"num_threads": 4, "num_leaves": num_leaves, "min_data_in_leaf": min_data_in_leaf, "max_depth": max_depth, "num_iterations":1000000, "objective": target.lightgbm_objective}
    if isinstance(target, Categorical):
        params["num_class"] = target.length
    return lgb.train(params, train_data.dataset, valid_sets=[valid_data.dataset], early_stopping_rounds=50, verbose_eval=100)


def test_model(model, test_data):
    """
    Run predictions.

    Args:
        model: Trained model.
        test_data: Test dataset.

    Returns:
        List of NumPy arrays containing predictions.
    """
    return model.predict(test_data.dataset.data, num_iteration=model.best_iteration)


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
    predicted = []

    start = 0
    for length in lengths:
        pred = raw_predicted[start:start+length]
        start += length
        if hasattr(target, "inverse_transform"):
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

    conn = sqlite3.connect(args.db_file)

    # Create the feature instances deactivating one-hot encoding.

    features = [name_to_feature(x) for x in args.features]
    for feature in features:
        if isinstance(feature, Categorical):
            feature.onehot = False
    target = name_to_feature(args.target)
    if isinstance(target, Categorical):
        target.onehot = False

    num_folds = get_num_folds(conn, args.dataset_name)

    log.info("Creating experiments table if not exists.")
    db_create_experiments_table(conn)

    for fold in range(num_folds):
        log.info("Starting with fold {}.".format(fold))
        best_metric_result = None
        best_model = None
        best_suffix = ""
        best_window = 0
        best_iter = 0

        for window in ([9, 13, 17, 21]):
            for num_leaves in [pow(4, x) for x in range(1, 6)]:
                for min_data_in_leaf in [pow(4, x) for x in range(1, 6)]:
                    max_depth = -1

                    log.info("Testing fold {} {}_{}_{}_{}".format(fold, window, num_leaves, min_data_in_leaf, max_depth))

                    #if db_model_exists(conn, args.experiment, fold):
                        #log.info("Experiment {} fold {} already exists, skipping.".format(args.experiment, fold))
                        #continue

                    log.info("Creating train dataset.")
                    train_data = create_dataset(args.db_file, features, target, window, args.dataset_name, "train", fold, limit=args.limit)

                    log.info("Creating valid dataset.")
                    valid_data = create_dataset(args.db_file, features, target, window, args.dataset_name, "valid", fold, limit=args.limit, reference=train_data.dataset)

                    log.info("Training model.")
                    model = train_model(train_data, valid_data, target, num_leaves, min_data_in_leaf, max_depth, limit=args.limit)

                    # Checking validation score
                    log.info("Predicting on valid data.")
                    valid_data = create_dataset(args.db_file, features, target, window, args.dataset_name, "valid", fold, limit=args.limit)
                    valid_raw_predicted = test_model(model, valid_data)

                    # Convert categorical features to one-hot encoded representation.
                    if isinstance(target, Categorical):
                        valid_raw_observed = np.zeros((len(valid_data.dataset.label), target.length))
                        for i in range(len(valid_data.dataset.label)):
                            valid_raw_observed[i, int(valid_data.dataset.label[i])] = 1
                    else:
                        valid_raw_observed = valid_data.dataset.label.reshape((-1, 1))
                        valid_raw_predicted = valid_raw_predicted.reshape((-1, 1))

                    if target.lightgbm_objective == "regression":
                        metric = mean_squared_error
                        op = operator.lt
                    else:
                        metric = log_loss
                        op = operator.lt

                    metric_result = metric(valid_raw_observed, valid_raw_predicted)
                    if best_metric_result is None or op(metric_result, best_metric_result):
                        log.info("Metric {} improved from {} to {}.".format(metric.__name__, best_metric_result, metric_result))
                        best_metric_result = metric_result
                        best_model = model
                        best_iter = model.best_iteration
                        best_suffix = "_{}_{}_{}_{}_{}".format(
                            window, num_leaves, min_data_in_leaf, max_depth, best_iter
                        )
                        best_window = window
                    else:
                        log.info("Metric {} NOT improved from {} to {}.".format(metric.__name__, best_metric_result, metric_result))
                        continue

                    ###

                    del train_data
                    del valid_data

        log.info("Creating test dataset.")
        test_data = create_dataset(args.db_file, features, target, best_window, args.dataset_name, "test", fold, limit=args.limit)

        log.info("Predicting on test data.")
        raw_predicted = test_model(best_model, test_data)

        log.info("Storing model and predictions.")
        # Convert categorical features to one-hot encoded representation.
        if isinstance(target, Categorical):
            raw_observed = np.zeros((len(test_data.dataset.label), target.length))
            for i in range(len(test_data.dataset.label)):
                raw_observed[i, int(test_data.dataset.label[i])] = 1
        else:
            raw_observed = test_data.dataset.label.reshape((-1, 1))
            raw_predicted = raw_predicted.reshape((-1, 1))

        predicted = process_predictions(test_data.lengths, target, raw_predicted)
        observed = process_predictions(test_data.lengths, target, raw_observed)

        db_store_predicted(conn, args.experiment, target, test_data.ids, predicted, simulate=False)
        db_store_predicted(conn, "observed", target, test_data.ids, observed, simulate=False)

        #db_store_model(conn, args.experiment, fold, best_model, best_suffix)
        db_store_model(conn, lambda m, f: m.save_model(f), args.experiment, fold, best_model, history, params, loss)

    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])

