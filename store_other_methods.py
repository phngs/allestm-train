"""
Parse the other prediction methods output and stores it in the database.
"""
import argparse
import sqlite3
import sys
import logging

from Bio import SeqIO

import mllib.parsers
from mllib.db import db_store_predicted
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
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("db_file", help="Sqlite3 db file.")
    parser.add_argument("format", help="Input format (i.e. which tool was used).")
    parser.add_argument("target", help="Target (to use the right table in the db).")
    parser.add_argument("fasta_file", help="Sequence FASTA file.")
    parser.add_argument("infile", help="Input file")

    return parser.parse_args(argv)


def main(argv):
    """
    Main method.

    Args:
        argv: Command line arguments.

    Returns:
        Nothing
    """
    args = parse_args(argv)

    conn = sqlite3.connect(args.db_file, timeout=120)

    parser = getattr(mllib.parsers, args.format)

    sequence = SeqIO.read(args.fasta_file, "fasta")

    result = parser(args.infile, sequence)

    target = name_to_feature(args.target)

    db_store_predicted(conn, args.format, target, [sequence.id], result, simulate=False)

    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
