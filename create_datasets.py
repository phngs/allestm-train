"""
Takes several filtering parameters and applies them to the initial dataset.
Afterwards, the remaining proteins are redundancy reduced, split into n-fold cross-validation
sets, which are stored in the given sqlite3 db.
"""
import argparse
import sys
import logging
import sqlite3
import tempfile
import subprocess
from multiprocessing import Pool

from operator import attrgetter

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold

from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio.SubsMat import MatrixInfo

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
    parser.add_argument('-m', '--method', default='x-ray', help='Experimental method.')
    parser.add_argument('-t', '--cov_trimmed', type=float, default=100, help='Minimum trimmed coverage.')
    parser.add_argument('-c', '--cov_total', type=float, default=80, help='Minimum total coverage.')
    parser.add_argument('-r', '--resolution', type=float, default=3.5, help='Minimum resolution.')
    parser.add_argument('-l', '--length', type=int, default=30, help='Minimum length.')
    parser.add_argument('-n', '--num_tms', type=int, default=0, help='Minimum number of TMS.')

    parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds.')
    parser.add_argument('-v', '--validation_frac', type=float, default=0.1, help='Fraction of training samples to use as validation set.')
    parser.add_argument('-p', '--test_frac', type=float, default=0.1, help='Fraction of training samples to use as test set.')

    parser.add_argument('-u', '--cdhit_bin', default='/home/users/hoenigschmid/programs/cd-hit-v4.6.4-2015-0603/cd-hit', help='CD-HIT binary.')
    parser.add_argument('-i', '--seq_id', type=float, default=0.4, help='Sequence identity used overall.')
    parser.add_argument('-a', '--word_length', type=int, default=2, help='Word length used for CDHIT.')
    parser.add_argument('-s', '--seq_id_test', type=float, default=0.3, help='Sequence identity used between training and validation/test.')

    parser.add_argument('db_file', help='Sqlite3 db file.')
    parser.add_argument('dataset_name', help='Name of the dataset in the database.')

    return parser.parse_args(argv)


def get_filtered_df(conn, method, cov_trimmed, cov_total, resolution, length, num_tms, limit=None):
    """
    Get a DataFrame containing the proteins which satisfy the filtering criteria.

    Args:
        conn: A sqlite3 connection object.
        method:
        cov_trimmed:
        cov_total:
        resolution:
        length:
        num_tms:
        limit:

    Returns:
        DataFrame containing the filtered proteins.
    """
    if limit:
        return pd.read_sql_query('select * from proteins where method like ? and cov_trimmed=? and cov_total>=? and resolution<=? and length>=? and num_tms>=? ORDER BY id limit ?', conn,
                                 params=(f'%{method}%', cov_trimmed, cov_total, resolution, length, num_tms, limit))
    else:
        return pd.read_sql_query('select * from proteins where method like ? and cov_trimmed=? and cov_total>=? and resolution<=? and length>=? and num_tms>=? ORDER BY id', conn,
                                 params=(f'%{method}%', cov_trimmed, cov_total, resolution, length, num_tms))


def remove_dataset_from_db(name, conn):
    """
    Remove dataset from database.

    Args:
        name: Name of the dataset.
        conn: Sqlite3 connection object.

    Returns:
        Nothing.
    """
    conn.cursor().execute('DELETE FROM datasets WHERE name=?', (name,))

def get_seqrecords_from_df(df):
    """
    Create an SeqRecord iterator from a DataFrame.

    Args:
        df: The DataFrame containing the sequences and their ids.

    Returns:
        SeqRecord generator.
    """
    for row in df.itertuples():
        yield SeqRecord(Seq(row.sequence, IUPAC.protein), id=row.id, description='')


def run_cmd(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    """
    Run a command.

    Args:
        cmd: List of command args.
        stdout: Redirect stdout.
        stderr: Redirect stderr.

    Returns:
        Output of the command.
    """

    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    out, err = proc.communicate()
    if err:
        log.error(f'Error running {" ".join(cmd)}:\n{err.decode()}')
        exit(1)
    return out


def run_cdhit(redundant_sequences, cdhit_bin, seq_id=0.4, word_length=2):
    """
    Takes a list/iterator/generator of SeqRecord objects, runs CDHIT
    and returns a list of non-redundant SeqRecord objects.

    Args:
        redundant_sequences: List/iterator/generator of SeqRecord objects.
        cdhit_bin: CDHIT binary.
        seq_id: Sequence identity threshold.
        word_length: Word length used for CDHIT.

    Returns:
        List of SeqRecord objects.
    """
    with tempfile.NamedTemporaryFile() as tmp_infile, tempfile.NamedTemporaryFile() as tmp_outfile:
        SeqIO.write(redundant_sequences, tmp_infile.name, 'fasta')
        log.info(f'Running CDHIT (input file: {tmp_infile.name}, output file: {tmp_outfile.name}).')
        cmd = [cdhit_bin, '-i', tmp_infile.name, '-o', tmp_outfile.name, '-c', str(seq_id), '-n', str(word_length)]
        run_cmd(cmd)
        log.info('CDHIT finished.')
        return list(SeqIO.parse(tmp_outfile.name, 'fasta'))


def create_tables(conn, cur):
    """
    Deletes and recreates the datasets table.
    Args:
        conn: Connection object.
        cur: Cursor object.

    Returns:
        Nothing.
    """
    datasets_table = ('CREATE TABLE IF NOT EXISTS datasets '
                     '(name VARCHAR(30), fold INT, kind VARCHAR(10), id VARCHAR(6), PRIMARY KEY(name, fold, kind, id))')

    #cur.execute('DROP TABLE IF EXISTS datasets')
    cur.execute(datasets_table)

    tmscore_table = ('CREATE TABLE IF NOT EXISTS tmscores (id_i VARCHAR(6), id_j VARCHAR(6), low FLOAT, avg FLOAT, high FLOAT, PRIMARY KEY(id_i, id_j))')
    cur.execute(tmscore_table)

    ident_table = ('CREATE TABLE IF NOT EXISTS idents (id_i VARCHAR(6), id_j VARCHAR(6), ident FLOAT, PRIMARY KEY(id_i, id_j))')
    cur.execute(ident_table)

    conn.commit()


def db_get_idents(conn):
    """
    Returns the idents table as dict of dicts (the ids in the first dimension are always
    smaller than those of the second.
    Args:
        conn: Sqlite3 connection object.

    Returns:
        Idents table as dictionary.
    """
    cur = conn.cursor()
    result = {}
    for id_i, id_j, ident in cur.execute('SELECT id_i, id_j, ident from idents'):
        result.setdefault(id_i, {})[id_j] = ident
    return result


def get_topo(idx, conn):
    """
    Get the topology for an id.

    Args:
        idx: Protein id.
        conn: Sqlite3 connection object.

    Returns:
        Tuple of (num_tms, num_reentrant).
    """
    return conn.cursor().execute('SELECT num_tms, num_res FROM proteins WHERE id=?', (idx, )).fetchone()


def _calc_pairwise(args):
    """
    Helper function to calculate a pairwise alignment.

    Args:
        args: Tuple of two sequence objects.

    Returns:
        List [1st sequence id, 2nd sequence id, percentage identity].
    """
    seq_i, seq_j = args

    ident = 0
    matrix = MatrixInfo.blosum62
    for a in pairwise2.align.globalds(seq_i.seq, seq_j.seq, matrix, -11, -1, penalize_end_gaps=(False, False)):
        for a_i, a_j in zip(a[0], a[1]):
            if a_i != '-' and a_j != '-' and a_i == a_j:
                ident += 1
        break

    ident /= min(len(seq_i.seq), len(seq_j.seq))

    return [seq_i.id, seq_j.id, ident]


def calc_pairwise(seqs, conn):
    """
    Calculate the pairwise sequence identities for a list of sequence objects and store them in the database.
    The database is first queried, i.e. only non existant pairwise sequence identities have to be calculated.

    Args:
        seqs: List of sequence objects.
        conn: Sqlite3 connection object.

    Returns:
        Idents table as dictionary, i.e. dict of dict, where the id in the first dimension is lexicographically
        smaller than the id in the second dimension.
    """
    log.info('Querying db for idents')
    result = db_get_idents(conn)

    to_align = []
    iter = 0
    iter_db = 0
    for i, seq_i in enumerate(sorted(seqs, key=attrgetter('id'))):
        for j, seq_j in enumerate(sorted(seqs, key=attrgetter('id'))):
            if j > i:
                if seq_i.id in result and seq_j.id in result[seq_i.id]:
                    iter_db += 1
                    continue
                else:
                    iter += 1
                    to_align.append([seq_i, seq_j])

    log.info(f'Found {iter_db} idents in db.')

    log.info('Calculating missing idents.')
    with Pool() as pool:
        pool_results = list(tqdm(pool.imap(_calc_pairwise, to_align), total=iter))

    params = []
    for id_i, id_j, ident in pool_results:
        params.append([id_i, id_j, ident])
    cur = conn.cursor()
    cur.executemany('INSERT INTO idents (id_i, id_j, ident) VALUES (?, ?, ?)', params)
    conn.commit()

    for id_i, id_j, ident in pool_results:
        result.setdefault(id_i, {})[id_j] = ident
    return result


def reduce_by_ident(retain_data, reduce_data, idents, cutoff=0.3):
    """
    Remove the proteins from a list of ids which are sequence similar to the ids in another list of ids.

    Args:
        retain_data: List of ids which should be retained.
        reduce_data: List of ids which should be reduced according to the retained list.
        idents: Dict of dict of identities.
        cutoff: Cutoff, i.e. sequences more similar than cutoff will be removed.

    Returns:
        Resulting list of ids with the redundant sequence ids removed.
    """
    reduce_data_reduced = []
    for reduce in reduce_data:
        valid = True
        for retain in retain_data:
            id_i, id_j = sorted([reduce, retain])
            if idents[id_i][id_j] > cutoff:
                valid = False
                break
        if valid:
            reduce_data_reduced.append(reduce)
    return reduce_data_reduced


def get_stratification(seq_ids, topos, min_size):
    """
    Takes a list of sequence ids and creates groups according to their number of transmembrane segments.
    Merges the two smallest groups until the smallest groups reaches a given minimum size.

    Args:
        seq_ids: List of sequence ids.
        topos: Dictionary of topologies.
        min_size: Minimum size for each group.

    Returns:
        Two lists: The 1st contains the protein ids, the second the group number for each of these proteins.
    """
    bins_dict = {}
    for seq_id in seq_ids:
        bins_dict.setdefault(topos[seq_id][0], []).append(seq_id)

    bins = sorted(bins_dict.values(), key=len, reverse=True)

    while len(bins[-1]) < min_size:
        bins[-2].extend(bins[-1])
        bins.pop()

    strat = []
    data = []
    for s, x in enumerate(bins):
        strat.extend([s for _ in x])
        data.extend(x)

    return np.array(data), np.array(strat)


def main(argv):
    """
    Main method.

    Args:
        argv: Command line arguments.

    Returns:
        Nothing
    """
    args = parse_args(argv)

    conn = sqlite3.connect(args.db_file)
    cur = conn.cursor()

    create_tables(conn, cur)
    remove_dataset_from_db(args.dataset_name, conn)

    # Get filtered dataset from db
    df = get_filtered_df(conn, args.method, args.cov_trimmed, args.cov_total, args.resolution, args.length, args.num_tms, limit=None)
    log.info(f'Initial filtered dataset contains {len(df)} sequences.')

    # Overall redundancy reduction
    redundant_sequences = list(get_seqrecords_from_df(df))
    non_redundant_sequences = run_cdhit(redundant_sequences, args.cdhit_bin, seq_id=args.seq_id, word_length=args.word_length)
    log.info(
        f'Non-redundant (seqid of {args.seq_id}) filtered dataset contains {len(non_redundant_sequences)} sequences.')

    idents = calc_pairwise(non_redundant_sequences, conn)

    topos = {x.id: get_topo(x.id, conn) for x in non_redundant_sequences}

    data, strat = get_stratification([x.id for x in non_redundant_sequences], topos, 2)

    log.info('Separating independent test data.')
    train_data, independent_test = train_test_split(data, stratify=strat, shuffle=True, random_state=1, test_size=args.test_frac)
    log.info(f'Redundant train contains {len(train_data)} sequences, test contains {len(independent_test)}')

    train_data_reduced = reduce_by_ident(independent_test, train_data, idents, cutoff=args.seq_id_test)
    log.info(f'Redundancy reduced train data contains {len(train_data_reduced)} sequences.')

    data_train, strat_train = get_stratification(train_data_reduced, topos, args.folds)

    # Splitting into k folds
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=2)
    for fold, (indices_train, indices_test) in enumerate(kf.split(data_train, strat_train)):
        train_redundant = data_train[indices_train]
        fold_test = data_train[indices_test]

        # Remove proteins from training data which are similar to the folds test set.
        train_all = reduce_by_ident(fold_test, train_redundant, idents, args.seq_id_test)

        fold_data_train, fold_strat_train = get_stratification(train_all, topos, 2)

        # Split remaining training data into redundant test and validation.
        fold_train_redundant, fold_valid = train_test_split(fold_data_train, stratify=fold_strat_train, shuffle=True, random_state=3+fold,
                                                            test_size=max(int(len(fold_data_train)*args.validation_frac), fold_strat_train[-1]+1))

        # Remove proteins from training data which are similar to the folds validation data.
        fold_train_reduced = reduce_by_ident(fold_valid, fold_train_redundant, idents, args.seq_id_test)

        log.info(f'Fold {fold} contains\n'
                 f'{len(train_redundant)} redundant training\n'
                 f'{len(train_all)} non-redundant training (train+valid)\n'
                 f'{len(fold_train_redundant)} redundant training\n'
                 f'{len(fold_train_reduced)} non-redundant training\n'
                 f'{len(fold_valid)} valid\n'
                 f'{len(fold_test)} test examples')

        # Store final datasets into DB.
        for train_id in fold_train_reduced:
            cur.execute('INSERT INTO datasets (name, fold, kind, id) VALUES (?, ?, ?, ?)', (args.dataset_name, fold, 'train', train_id))

        for valid_id in fold_valid:
            cur.execute('INSERT INTO datasets (name, fold, kind, id) VALUES (?, ?, ?, ?)', (args.dataset_name, fold, 'valid', valid_id))

        for test_id in fold_test:
            cur.execute('INSERT INTO datasets (name, fold, kind, id) VALUES (?, ?, ?, ?)', (args.dataset_name, fold, 'test', test_id))

        conn.commit()

    for test_id in independent_test:
        cur.execute('INSERT INTO datasets (name, fold, kind, id) VALUES (?, ?, ?, ?)', (args.dataset_name, 0, 'independent_test', test_id))

    conn.commit()

    conn.close()



if __name__ == '__main__':
    main(sys.argv[1:])

