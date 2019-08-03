"""
Run HHblits on an entry in the protein table and store alignment in the supplied db.
"""
import argparse
import sys
import logging
import subprocess
from tempfile import NamedTemporaryFile
import sqlite3

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio.Align import MultipleSeqAlignment

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
    parser.add_argument('protein_id', help='The id of the protein.')
    parser.add_argument('db_file', help='The sqlite3 db file.')
    parser.add_argument('-b', '--hhblits_bin', default='/home/users/hoenigschmid/programs/hh-suite/bin/hhblits', help='HHblits binary.')
    parser.add_argument('-d', '--hhblits_db', default='/localscratch/uniclust30_2017_10/uniclust30_2017_10', help='Db used by HHblits.')
    parser.add_argument('-c', '--cpus', type=int, default=4, help='Number of CPUs used by HHblits.')

    return parser.parse_args(argv)


def get_protein(cur, protein_id):
    """
    Get a sequence record from the protein table.

    Args:
        cur: Cursor object.
        protein_id: Protein id.

    Returns:
        SeqRecord.
    """
    sequence, = cur.execute('SELECT sequence FROM proteins WHERE id=?', (protein_id,)).fetchone()
    if sequence:
        return SeqRecord(Seq(sequence, IUPAC.protein), id=protein_id, description='')
    else:
        return None


def run_hhblits(query, hhblits_bin, hhblits_db, cpus):
    """
    Run HHblits and return a MultipleSequenceAlignment object.
    Args:
        query: The query protein SeqRecord.
        hhblits_bin: HHblits binary.
        hhblits_db: HHblits db.
        cpus: Number of CPUs.

    Returns:
        MSA.
    """
    with NamedTemporaryFile() as infile, NamedTemporaryFile() as hhr_file, NamedTemporaryFile() as a3m_file:
        SeqIO.write(query, infile.name, 'fasta')
        cmd = [hhblits_bin, '-i', infile.name, '-o', hhr_file.name, '-oa3m', a3m_file.name, '-d', hhblits_db, '-cpu', str(cpus), '-maxfilt', '99999999']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

        if proc.returncode:
            log.error(f'HHblits run failed: {out.decode()}\n{err.decode()}')
            exit(1)

        a3m_file.seek(0)
        msa = a3m_file.read().decode()

        return msa


def create_table(conn, cur):
    """
    Creates the alignment table if it does not exist.

    Args:
        conn: Sqlite3 connection.
        cur: Cursor.

    Returns:
        Nothing.
    """
    cur.execute('CREATE TABLE IF NOT EXISTS alignments (id VARCHAR(6), msa TEXT, PRIMARY KEY(id))')
    conn.commit()


def check_if_exists(cur, protein_id):
    """
    Check if alignment already exists.

    Args:
        cur: Cursor object.
        protein_id: Protein id.

    Returns:
        True if alignment already exists, False otherwise.
    """
    ali_id = cur.execute('SELECT id FROM alignments WHERE id=?', (protein_id,)).fetchone()
    if ali_id:
        return True
    else:
        return False


def write_msa_to_db(conn, cur, protein_id, msa):
    """
    Writes a MSA to the database.

    Args:
        cur: Cursor object.
        protein_id: Protein id of the query.
        msa: MSA.

    Returns:
        Nothing.
    """
    cur.execute('INSERT INTO alignments (id, msa) VALUES (?, ?)', (protein_id, msa))
    conn.commit()


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

    log.info('Creating table.')
    create_table(conn, cur)

    if check_if_exists(cur, args.protein_id):
        log.warning(f'Alignment for protein id {args.protein_id} already exists, skipping.')
    else:
        log.info(f'Fetching sequence for protein id {args.protein_id}.')
        protein = get_protein(cur, args.protein_id)

        log.info('Running HHblits')
        msa = run_hhblits(protein, args.hhblits_bin, args.hhblits_db, args.cpus)

        # Second check if sth. has changed in the meantime.
        if check_if_exists(cur, args.protein_id):
            log.warning(f'Alignment for protein id {args.protein_id} already exists, skipping.')
        else:
            log.info('Writing MSA to db.')
            write_msa_to_db(conn, cur, args.protein_id, msa)

    conn.close()


if __name__ == '__main__':
    main(sys.argv[1:])
