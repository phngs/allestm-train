"""
Save alignment in db.
"""
import argparse
import sys
import logging
import sqlite3

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
    parser.add_argument('db_file', help='Sqlite3 db file.')
    parser.add_argument('ident', help='Identifier (db key).')
    parser.add_argument('infile', help='Alignment in fasta format.')

    return parser.parse_args(argv)


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
    cur.execute('INSERT OR REPLACE INTO alignments (id, msa) VALUES (?, ?)', (protein_id, msa))
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

    log.info(f'Storing {args.ident} from {args.infile}')

    with open(args.infile, 'r') as fh:
        msa = fh.read()
        write_msa_to_db(conn, cur, args.ident, msa)

    conn.close()


if __name__ == '__main__':
    main(sys.argv[1:])
