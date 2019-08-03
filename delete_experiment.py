"""

"""
import argparse
import sys
import logging
import re
import sqlite3

import mllib.features.continuous
import mllib.features.binary
import mllib.features.categorical

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
    parser.add_argument('db_file', help='Sqlite3 file.')
    parser.add_argument('experiments', nargs='*', help='Experiment which should be deleted file.')

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

    conn = sqlite3.connect(args.db_file)
    tables = ['experiments']
    for module in [mllib.features.categorical, mllib.features.continuous, mllib.features.binary]:
        for entry in filter(lambda x: not x.startswith('_') and x[0].isupper(), dir(module)):
            cls = f'{module.__name__}.{entry}'
            cls = re.sub('\.', '_', cls)
            tables.append(cls)

    for table in tables:
        for experiment in args.experiments:
            log.info(f'Deleting {experiment} from {table}')
            try:
                conn.cursor().execute(f'DELETE FROM {table} WHERE experiment GLOB "*{experiment}*"')
            except sqlite3.OperationalError:
                log.warning(f'Could not delete from {table}. Maybe table does not exist.')
                continue
            log.info('Successful.')
            conn.commit()

    conn.commit()
    conn.close()





if __name__ == '__main__':
    main(sys.argv[1:])
