"""

"""
import argparse
import sys
import logging
import sqlite3

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.externals import joblib

from mllib.db import db_store_predicted, db_create_experiments_table, get_num_folds, \
    db_store_model
from mllib.features.categorical import Categorical
from mllib.features.continuous import Continuous
from mllib.features.binary import Binary
from mllib.features.utils import name_to_feature

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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

    parser.add_argument('-f', '--features', default=[], action='append', help='Prefixes of the experiments to blend.')

    parser.add_argument('-t', '--target', help='Name of target.')

    return parser.parse_args(argv)


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
        #if hasattr(target, 'inverse_transform'):
            #pred = target.inverse_transform(pred)
        predicted.append(pred)

    return predicted


def get_ids_and_lengths(df):
    ids = []
    lengths = []
    lengths_tmp = df.id.value_counts()
    ids_tmp = df.id.unique().tolist()

    for id in ids_tmp:
        ids.append(id)
        lengths.append(lengths_tmp[id])

    return ids, lengths


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
    target = name_to_feature(args.target)
    target_table_name = "{}.{}".format(type(target).__module__, type(target).__qualname__).replace(".", "_")
    target_cols = [f'val_{i}' for i in range(len(target))]

    num_folds = get_num_folds(conn, args.dataset_name)

    log.info('Creating experiments table if not exists.')
    db_create_experiments_table(conn)

    exp_filter_str = ' or '.join([f'experiment glob "{exp}*"' for exp in args.features])
    experiments = [x[0] for x in conn.cursor().execute(
        f'select distinct experiment from {target_table_name} where '
        f'({exp_filter_str}) and '
        f'experiment not like "%VALID%" and experiment not like "%TEST%"'
        f'order by experiment'
    ).fetchall()]

    log.info(f'Target: {target_table_name}')
    log.info(f'Using experiments: {experiments}')

    avg_target_cols = ','.join([f'avg({target_col})' for target_col in target_cols])
    for experiment in experiments:
        ids = []
        preds = []
        qry = f'select id, {avg_target_cols} ' \
              f'from {target_table_name} ' \
              f'where experiment glob("{experiment}_IND_TEST_*") ' \
              f'group by id, resi order by id, resi'

        print(qry)
        for row in conn.cursor().execute(qry):
            if len(ids) == 0 or ids[-1] != row[0]:
                ids.append(row[0])
                preds.append([])
            preds[-1].append(row[1:])

        db_store_predicted(conn, f'{experiment}_IND_TEST', target, ids, [np.array(x) for x in preds], simulate=False)

    conn.close()

if __name__ == '__main__':
    main(sys.argv[1:])
