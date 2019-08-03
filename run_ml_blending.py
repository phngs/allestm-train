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

    feature_cols = []
    for experiment in experiments:
        for target_col in target_cols:
            feature_cols.append(f'{experiment}_{target_col}')

    log.info(f'Using feature columns: {feature_cols}.')

    for fold in range(num_folds):
        obss_valid = pd.read_sql_query(
            f'select id, {",".join(target_cols)} from {target_table_name} join datasets using (id) where name="{args.dataset_name}" and fold={fold} and kind="valid" and experiment="observed" order by id,resi',
            conn)
        obss_test = pd.read_sql_query(
            f'select id, {",".join(target_cols)} from {target_table_name} join datasets using (id) where name="{args.dataset_name}" and fold={fold} and kind="test" and experiment="observed" order by id,resi',
            conn)
        obss_ind = pd.read_sql_query(
            f'select id, {",".join(target_cols)} from {target_table_name} join datasets using (id) where name="{args.dataset_name}" and fold=0 and kind="independent_test" and experiment="observed" order by id,resi',
            conn)

        log.info(f'Observed looks like:\n{obss_valid.head()}')

        preds_valid = []
        preds_test = []
        preds_ind = []
        for experiment in experiments:
            val = pd.read_sql_query(
                f'select {",".join(target_cols)} from {target_table_name} where experiment="{experiment}_VALID_{fold}" order by id,resi',
                conn)
            val.columns = [f'{experiment}_{col}' for col in val.columns]
            preds_valid.append(val)

            ind = pd.read_sql_query(
                f'select {",".join(target_cols)} from {target_table_name} where experiment="{experiment}_IND_TEST_{fold}" order by id,resi',
                conn)
            ind.columns = [f'{experiment}_{col}' for col in ind.columns]
            preds_ind.append(ind)

            tst = pd.read_sql_query(
                f'select {",".join(target_cols)} from {target_table_name} join datasets using (id) where experiment="{experiment}" and kind="test" and fold={fold} order by id,resi',
                conn)
            tst.columns = [f'{experiment}_{col}' for col in tst.columns]
            preds_test.append(tst)

        #print(len(obss_valid))
        #print()
        #for d in preds_valid:
            #print(len(d))

        dfs_valid = pd.concat([obss_valid] + preds_valid, axis=1)
        dfs_test = pd.concat([obss_test] + preds_test, axis=1)
        dfs_ind = pd.concat([obss_ind] + preds_ind, axis=1)

        log.info(f'DataFrame looks like:\n{dfs_valid.head()}')

        #for c in dfs_valid.columns:
            #print(len(dfs_valid[c]))

        #exit()
        #dfs_valid[feature_cols].to_csv('~/feat.csv')
        #print(dfs_valid[target_cols].values.squeeze())

        if isinstance(target, Continuous):
            lrs = LinearRegression().fit(dfs_valid[feature_cols], dfs_valid[target_cols].values.squeeze())
        if isinstance(target, Binary):
            lrs = LogisticRegression().fit(dfs_valid[feature_cols], dfs_valid[target_cols].values.squeeze())
        if isinstance(target, Categorical):
            lrs = LogisticRegression().fit(dfs_valid[feature_cols], np.argmax(dfs_valid[target_cols].values, axis=1).squeeze())

        history = {}
        params = {'features': args.features}

        obs_test = dfs_test[target_cols].values
        obs_ind = dfs_ind[target_cols].values
        log.info(f'Observed test looks like:\n{obs_test[:5]}')

        if isinstance(target, Continuous):
            pred_test = lrs.predict(dfs_test[feature_cols])
            pred_ind = lrs.predict(dfs_ind[feature_cols])
        else:
            pred_test = lrs.predict_proba(dfs_test[feature_cols])
            pred_ind = lrs.predict_proba(dfs_ind[feature_cols])

        log.info(f'Pred test looks like:\n{pred_test[:5]}')

        log.info('Storing test predictions.')
        test_ids, test_lengths = get_ids_and_lengths(dfs_test)
        test_observed = process_predictions(test_lengths, target, obs_test)
        test_predicted = process_predictions(test_lengths, target, pred_test)

        db_store_predicted(conn, args.experiment, target, test_ids, test_predicted, simulate=False)
        db_store_predicted(conn, 'observed', target, test_ids, test_observed, simulate=False)

        log.info('Storing independent test predictions.')
        ind_ids, ind_lengths = get_ids_and_lengths(dfs_ind)
        ind_observed = process_predictions(ind_lengths, target, obs_ind)
        ind_predicted = process_predictions(ind_lengths, target, pred_ind)

        db_store_predicted(conn, f'{args.experiment}_IND_TEST_{fold}', target, ind_ids, ind_predicted, simulate=False)
        db_store_predicted(conn, 'observed', target, ind_ids, ind_observed, simulate=False)

        log.info('Storing model.')
        db_store_model(conn, args.db_file, (lambda m, f: joblib.dump(m, f, compress=3, protocol=-1)), args.experiment, fold, lrs, history, params, 0.0)

    conn.close()

if __name__ == '__main__':
    main(sys.argv[1:])
