"""
Takes a file containing multiple lists of PDB ids and a directory of
.pdb and .opm files, extracts the chains, filters them and writes
several properties to a sqlite3 database.
"""
import argparse
import os
import re
import sys
import logging
import itertools
import sqlite3
from operator import itemgetter
from tempfile import NamedTemporaryFile

from tqdm import tqdm

from Bio.PDB import *
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import Polypeptide
import Bio.pairwise2
from Bio import SeqIO
from Bio.PDB.Vector import Vector, calc_dihedral
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

pd.set_option('display.max_rows', 1000)


def parse_args(argv):
    """
    Parse the arguments.

    Args:
        argv: List of command line args.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", help="Directory containing pdb and opm files.")
    parser.add_argument("db_file", help="Output file (sqlite3 db).")
    parser.add_argument("id_files", nargs="+", help="Files containing pdb ids.")
    parser.add_argument("-d", "--dssp", default="mkdssp", help="DSSP binary.")

    return parser.parse_args(argv)


def read_id_files(*filenames):
    """
    Read a list of files, each containing one id per line.

    Args:
        filenames: List of file names.

    Returns:
        Set of ids.
    """
    ids = set()
    for filename in filenames:
        with open(filename, "r") as fh:
            for line in fh:
                ids.add(line.strip())
    return ids


def parse_thickness(filename):
    """
    Parses an .opm file and returns the thickness if any.

    Args:
        filename: .opm file.

    Returns:
        Either half of the bilayer thickness or None if not present.
    """
    thicknesses = []
    with open(filename, "r") as fh:
        for line in fh:
            if line.startswith("REMARK") and re.search("1/2 of bilayer thickness:", line):
                thicknesses.append(float(line.strip().split()[-1]))
    if len(thicknesses) == 0:
        return None
    else:
        return sum(thicknesses)/len(thicknesses)


def find_best_alignment(query, targets):
    """
    Compare a single sequence against a list of sequences. Finds the sequence
    in the list with the highest score which is a perfect match and returns an alignment.

    Args:
        query: Single sequence.
        targets: List of sequences.

    Returns:
        A tuple of aligned sequences. If no perfect match was found, returns None.
    """
    score = 0
    best_ali = None
    for target in targets:
        for strict_ali in Bio.pairwise2.align.globalxx(query, target, one_alignment_only=True):
            if int(strict_ali[2]) == len(query):
                for proper_ali in Bio.pairwise2.align.globalms(query, target, 2, -1, -0.5, -0.1, one_alignment_only=True, penalize_end_gaps=False):
                    if proper_ali[2] > score:
                        score = proper_ali[2]
                        best_ali = (proper_ali[0], proper_ali[1])
                    break
            break
    return best_ali


def create_tables(conn, cur):
    """
    Deletes and recreates the proteins and raw_data tables.
    Args:
        conn: Connection object.
        cur: Cursor object.

    Returns:
        Nothing.
    """
    coord_columns = []
    for atom_type, coord_type in itertools.product(["n", "ca", "c"], "xyz"):
        coord_columns.append("{}_{} FLOAT".format(atom_type, coord_type))

    raw_data_table = ("CREATE TABLE IF NOT EXISTS raw_data "
                      "(id VARCHAR(6), resi INT, pdb_id VARCHAR(4), chain VARCHAR(1), "
                      "atomseq TEXT, seqres TEXT, bfactor FLOAT, phi FLOAT, psi FLOAT, "
                      "sec VARCHAR(1), acc_chain FLOAT, acc_complex FLOAT, "
                      "topo VARCHAR(1), tms_nr INT, re_nr INT, "
                      "{}, PRIMARY KEY(id, resi))").format(", ".join(coord_columns))

    cur.execute("DROP TABLE IF EXISTS raw_data")
    cur.execute(raw_data_table)

    proteins_table = ("CREATE TABLE IF NOT EXISTS proteins "
                          "(id VARCHAR(6), pdb_id VARCHAR(4), chain VARCHAR(1), sequence TEXT, "
                          "length INT, thickness FLOAT, method VARCHAR(50), "
                          "resolution FLOAT, cov_total FLOAT, cov_trimmed FLOAT, "
                          "num_tms INT, num_res INT, "
                          "PRIMARY KEY(id))")

    cur.execute("DROP TABLE IF EXISTS proteins")
    cur.execute(proteins_table)

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

    pdb_ids = read_id_files(*args.id_files)

    # Open db connection and create cursor
    conn = sqlite3.connect(args.db_file)
    cur = conn.cursor()

    create_tables(conn, cur)

    key_error = 0
    for it, pdb_id in enumerate(tqdm(pdb_ids)):
        #if pdb_id != "5b0w":
            #continue
        log.info("{} processing ({}/{}).".format(pdb_id, it, len(pdb_ids)))

        pdb_file = os.path.join(args.data_dir, pdb_id+".pdb")
        opm_file = os.path.join(args.data_dir, pdb_id+".opm")

        # Check if files exist.
        if not os.path.exists(pdb_file) \
                or os.path.getsize(pdb_file) == 0 \
                or not os.path.exists(opm_file) \
                or os.path.getsize(opm_file) == 0:
            log.warning("{} is missing some file, skipping.".format(pdb_id))
            continue

        # Parse the .pdb file.
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        pdb_structure = pdb_parser.get_structure("struc", pdb_file)

        method = pdb_structure.header["structure_method"]
        resolution = pdb_structure.header["resolution"]
        seqress = [record.seq for record in SeqIO.parse(pdb_file, "pdb-seqres")]

        # Parse .opm file.
        opm_parser = Bio.PDB.PDBParser(QUIET=True)
        opm_structure = opm_parser.get_structure("struc", opm_file)

        thickness = parse_thickness(opm_file)
        if not thickness:
            log.warning("{}_{} has no thickness, skipping.".format(pdb_id, chain.id))
            continue

        dssp_complex = None

        # Loop over the chains found in the .opm file.
        for chain in opm_structure[0]:
            log.info("{}_{} processing.".format(pdb_id, chain.id))

            if not chain.id.strip():
                log.info("Empty chain name, skipping")
                continue

            ppb = Bio.PDB.PPBuilder()
            atomseq_aa_list = Polypeptide(list(itertools.chain(*ppb.build_peptides(chain))))
            atomseq = atomseq_aa_list.get_sequence()

            alignment = find_best_alignment(atomseq, seqress)
            if not alignment:
                log.warning("{}_{} has no matching seqres, skipping.".format(pdb_id, chain.id))
                continue

            atomseq_ali, seqres_ali = alignment

            # Build list of columns to track.
            columns = ["id", "resi", "pdb_id", "chain", "atomseq", "seqres", "bfactor", "phi", "psi", "sec", "acc_chain", "acc_complex", "topo", "tms_nr", "re_nr"]
            for atom_type, coord_type in itertools.product(["n", "ca", "c"], "xyz"):
                    columns.append("{}_{}".format(atom_type, coord_type))

            # DataFrame to store all values for this protein chain.
            chain_df = pd.DataFrame({"id": "{}_{}".format(pdb_id, chain.id), "pdb_id": pdb_id, "chain": chain.id, "atomseq": list(atomseq_ali), "seqres": list(seqres_ali)}, index=np.arange(len(atomseq_ali)), columns=columns)

            # Add atom coords according to the alignment.
            if all([all([atom_type in res for res in atomseq_aa_list]) for atom_type in ["N", "CA", "C"]]):
                for atom_type in ["n", "ca", "c"]:
                    #if all([atom_type.upper() in res for res in atomseq_aa_list]):
                    chain_df.loc[chain_df.atomseq != "-", [atom_type+"_x", atom_type+"_y", atom_type+"_z"]] = [res[atom_type.upper()].get_coord() for res in atomseq_aa_list]
            else:
                log.warning("{}_{} has not all N, CA or C atoms, skipping.".format(pdb_id, chain.id))
                continue

            # Add ca bfactors.
            chain_df.loc[chain_df.atomseq != "-", "bfactor"] = [res["CA"].get_bfactor() for res in atomseq_aa_list]

            #
            # Filtering.
            #
            # Convert columns to numeric.
            chain_df = chain_df.apply(pd.to_numeric, errors="ignore")

            if not chain_df.ca_z.min() <= -thickness or not chain_df.ca_z.max() >= thickness:
                log.warning("{}_{} has no atoms on both sides of the membrane, skipping.".format(pdb_id, chain.id))
                continue

            if len(chain_df[chain_df.atomseq != "-"].bfactor.unique()) == 1:
                log.warning("{}_{} contains only one bfactor value, skipping.".format(pdb_id, chain.id))
                continue

            if len(chain_df[chain_df.atomseq != "-"].atomseq.unique()) == 1:
                log.warning("{}_{} contains only one type of amino acid, skipping.".format(pdb_id, chain.id))
                continue

            # DSSP
            if not dssp_complex:
                try:
                    dssp_complex = DSSP(opm_structure[0], opm_file, dssp=args.dssp)
                except (KeyError, PDBException, Exception):
                    log.warning("Incomplete chain, skipping...")
                    key_error += 1
                    continue

            with NamedTemporaryFile() as tmpfile:
                io = PDBIO()

                io.set_structure(chain)
                io.save(tmpfile.name)

                try:
                    dssp_chain = DSSP(opm_structure[0], tmpfile.name, dssp=args.dssp)

                    resids = [(chain.id, aa.id) for aa in atomseq_aa_list]
                    dssp = [(dssp_chain[idx][2], dssp_chain[idx][3], dssp_complex[idx][3]) for idx in resids]

                    chain_df.loc[chain_df.atomseq != "-", ["sec", "acc_chain", "acc_complex"]] = dssp
                except (KeyError, PDBException, Exception):
                    log.warning("Incomplete residue, skipping...")
                    key_error += 1
                    continue

            # Super ugly TMS and RE calculation
            zcoords = chain_df[chain_df.atomseq != "-"].ca_z.values

            smooth_win = 5
            mask = np.ones(smooth_win) / smooth_win
            zcoords_smoothed = np.convolve(zcoords, mask, 'same')

            top_mem = thickness
            bot_mem = -thickness

            top_tm = thickness - 10 #thickness * 2 / 10 * 2.5
            bot_tm = -thickness + 10 #-thickness * 2 / 10 * 2.5

            top_re = thickness - 3 #thickness * 2 / 10 * 4
            bot_re = -thickness + 3 #-thickness * 2 / 10 * 4

            top_re2 = thickness - 5 #thickness * 2 / 10 * 4
            bot_re2 = -thickness + 5 #-thickness * 2 / 10 * 4

            transitions = []
            if bot_mem <= zcoords[0] < 0:
                transitions.append([0, "^", "red", "start"])
                transitions.append([0, "^", "green", "up"])
            elif 0 < zcoords[0] <= top_mem:
                transitions.append([0, "v", "red", "start"])
                transitions.append([0, "v", "green", "down"])

            for pos in range(1, len(zcoords) - 1):
                prev_zcoord = zcoords[pos - 1]
                cur_zcoord = zcoords[pos]

                if prev_zcoord <= bot_mem <= cur_zcoord:
                    transitions.append([pos, "^", "red", "start"])
                    transitions.append([pos, "^", "green", "up"])
                elif prev_zcoord <= top_mem <= cur_zcoord:
                    if transitions[-1][3] == "up":
                        transitions.pop()
                    transitions.append([pos-1, "^", "red", "stop"])

                elif prev_zcoord <= bot_re <= cur_zcoord:
                    if zcoords_smoothed[pos - 1] < zcoords_smoothed[pos] < zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "up":
                            transitions.pop()
                        transitions.append([pos, "^", "green", "up"])
                elif prev_zcoord <= bot_re2 <= cur_zcoord:
                    if zcoords_smoothed[pos - 1] < zcoords_smoothed[pos] < zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "up":
                            transitions.pop()
                        transitions.append([pos, "^", "green", "up"])
                elif prev_zcoord <= bot_tm <= cur_zcoord:
                    if zcoords_smoothed[pos - 1] < zcoords_smoothed[pos] < zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "up":
                            transitions.pop()
                        transitions.append([pos, "^", "green", "up"])
                elif prev_zcoord <= top_tm <= cur_zcoord: # and zcoords[transitions[-1][0]] < 0:
                    if zcoords_smoothed[pos - 1] < zcoords_smoothed[pos] < zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "up":
                            transitions.pop()
                        #elif transitions[-1][3] == "down" and zcoords[transitions[-1][0]] >= zcoords[pos]:
                        #continue
                        transitions.append([pos, "^", "blue", "up"])
                elif prev_zcoord <= top_re <= cur_zcoord: # and zcoords[transitions[-1][0]] < 0:
                    if zcoords_smoothed[pos - 1] < zcoords_smoothed[pos] < zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "up":
                            transitions.pop()
                        #elif transitions[-1][3] == "down" and zcoords[transitions[-1][0]] >= zcoords[pos]:
                        #continue
                        transitions.append([pos, "^", "blue", "up"])
                elif prev_zcoord <= top_re2 <= cur_zcoord: # and zcoords[transitions[-1][0]] < 0:
                    if zcoords_smoothed[pos - 1] < zcoords_smoothed[pos] < zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "up":
                            transitions.pop()
                        #elif transitions[-1][3] == "down" and zcoords[transitions[-1][0]] >= zcoords[pos]:
                        #continue
                        transitions.append([pos, "^", "blue", "up"])

                elif prev_zcoord >= top_mem >= cur_zcoord:
                    transitions.append([pos, "v", "red", "start"])
                    transitions.append([pos, "v", "green", "down"])
                elif prev_zcoord >= bot_mem >= cur_zcoord:
                    if transitions[-1][3] == "down":
                        transitions.pop()
                    transitions.append([pos-1, "v", "red", "stop"])

                elif prev_zcoord >= top_re >= cur_zcoord:
                    if zcoords_smoothed[pos - 1] > zcoords_smoothed[pos] > zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "down":
                            transitions.pop()
                        transitions.append([pos, "v", "green", "down"])
                elif prev_zcoord >= top_re2 >= cur_zcoord:
                    if zcoords_smoothed[pos - 1] > zcoords_smoothed[pos] > zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "down":
                            transitions.pop()
                        transitions.append([pos, "v", "green", "down"])
                elif prev_zcoord >= top_tm >= cur_zcoord:
                    if zcoords_smoothed[pos - 1] > zcoords_smoothed[pos] > zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "down":
                            transitions.pop()
                        transitions.append([pos, "v", "green", "down"])
                elif prev_zcoord >= bot_tm >= cur_zcoord: # and zcoords[transitions[-1][0]] > 0:
                    if zcoords_smoothed[pos - 1] > zcoords_smoothed[pos] > zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "down":
                            transitions.pop()
                        #elif transitions[-1][3] == "up" and zcoords[transitions[-1][0]] <= zcoords[pos]:
                        #continue
                        transitions.append([pos, "v", "blue", "down"])
                elif prev_zcoord >= bot_re >= cur_zcoord: # and zcoords[transitions[-1][0]] > 0:
                    if zcoords_smoothed[pos - 1] > zcoords_smoothed[pos] > zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "down":
                            transitions.pop()
                        #elif transitions[-1][3] == "up" and zcoords[transitions[-1][0]] <= zcoords[pos]:
                        #continue
                        transitions.append([pos, "v", "blue", "down"])
                elif prev_zcoord >= bot_re2 >= cur_zcoord: # and zcoords[transitions[-1][0]] > 0:
                    if zcoords_smoothed[pos - 1] > zcoords_smoothed[pos] > zcoords_smoothed[pos +1 ]:
                        if transitions[-1][3] == "down":
                            transitions.pop()
                        #elif transitions[-1][3] == "up" and zcoords[transitions[-1][0]] <= zcoords[pos]:
                        #continue
                        transitions.append([pos, "v", "blue", "down"])

            if bot_mem <= zcoords[-1] < 0:
                if transitions[-1][3] == "down":
                    transitions.pop()
                transitions.append([len(zcoords)-1, "v", "red", "stop"])
            elif bot_mem <= zcoords[-2] < 0:
                if transitions[-1][3] == "down":
                    transitions.pop()
                transitions.append([len(zcoords)-2, "v", "red", "stop"])
            elif 0 < zcoords[-1] <= top_mem:
                if transitions[-1][3] == "up":
                    transitions.pop()
                transitions.append([len(zcoords)-1, "^", "red", "stop"])
            elif 0 < zcoords[-2] <= top_mem:
                if transitions[-1][3] == "up":
                    transitions.pop()
                transitions.append([len(zcoords)-2, "^", "red", "stop"])

            for t in range(len(transitions) - 1):

                if transitions[t][3] == "up":
                    prev_pos = transitions[t - 1][0]
                    next_pos = transitions[t + 1][0]
                    if next_pos - prev_pos > 1:
                        shift = np.argmax(zcoords[prev_pos + 1:next_pos])
                        transitions[t][0] = prev_pos + 1 + shift
                elif transitions[t][3] == "down":
                    prev_pos = transitions[t - 1][0]
                    next_pos = transitions[t + 1][0]
                    if next_pos - prev_pos > 1:
                        shift = np.argmin(zcoords[prev_pos + 1:next_pos])
                        transitions[t][0] = prev_pos + 1 + shift

            for t in range(len(transitions) - 1, -1, -1):
                if transitions[t][2] == "green":
                    if transitions[t][3] == "up":
                        if zcoords[transitions[t][0]] < bot_re:
                            transitions.pop(t)
                    elif transitions[t][3] == "down":
                        if zcoords[transitions[t][0]] > top_re:
                            transitions.pop(t)

            # 4a2n_B

            min_length = 3
            tms = []
            res = []
            start = 0
            kind = "tms"
            for t in range(len(transitions)):
                curr = transitions[t]
                if curr[3] == "start":
                    start = curr[0]
                    kind = "tms"
                elif curr[2] == "green":
                    kind = "re"
                elif curr[2] == "blue" or curr[3] == "stop":
                    end = curr[0]
                    if end - start >= min_length - 1:
                        mm = sorted([zcoords[start], zcoords[end]])
                        if kind == "tms" and mm[0] < 0 and mm[1] > 0:
                            tms.append([start, end])
                        elif kind == "re" and ((mm[0] < 0 and mm[1] < 0) or (mm[0] > 0 and mm[1] > 0)):
                            res.append([start, end])
                    kind = "tms"
                    start = curr[0] + 1


            topo = []
            tms_nr = [-1 for _ in range(len(zcoords))]
            re_nr = [-1 for _ in range(len(zcoords))]
            for i, zcoord in enumerate(zcoords):
                if zcoord < 0:
                    topo.append("I")
                else:
                    topo.append("O")

            for num, tm in enumerate(tms):
                for t in range(tm[0], tm[1] + 1):
                    topo[t] = "T"
                    tms_nr[t] = num

            for num, re in enumerate(res):
                for r in range(re[0], re[1] + 1):
                    topo[r] = "R"
                    re_nr[r] = num

            chain_df.loc[chain_df.atomseq != "-", "topo"] = topo
            chain_df.loc[chain_df.atomseq != "-", "tms_nr"] = tms_nr
            chain_df.loc[chain_df.atomseq != "-", "re_nr"] = re_nr

            #
            # Interpolation, calculation of coverage and % of missing residues
            # in the seqres.
            #
            cov_total = len(chain_df[chain_df.atomseq != "-"]) / len(chain_df) * 100

            # Trimming.
            first, last = chain_df[chain_df.atomseq != "-"].iloc[[0, -1]].index
            chain_df = chain_df.loc[first:last]
            cov_trimmed = len(chain_df[chain_df.atomseq != "-"]) / len(chain_df) * 100

            # Interpolation.
            chain_df.interpolate(inplace=True)

            #
            # Phi/psi angle calculation
            #
            def calc_phi_psi(row):
                iloc = chain_df.index.get_loc(row.name)
                iloc_before = iloc - 1
                iloc_after = iloc + 1

                phi = 0
                psi = 0

                n = Vector(*row[["n_x", "n_y", "n_z"]])
                ca = Vector(*row[["ca_x", "ca_y", "ca_z"]])
                c = Vector(*row[["c_x", "c_y", "c_z"]])

                if iloc_before < 0:
                    pass
                else:
                    cp = Vector(*chain_df.iloc[iloc_before][["c_x", "c_y", "c_z"]])
                    phi = calc_dihedral(cp, n, ca, c)

                if iloc_after >= len(chain_df):
                    pass
                else:
                    nn = Vector(*chain_df.iloc[iloc_after][["n_x", "n_y", "n_z"]])
                    psi = calc_dihedral(n, ca, c, nn)

                row[["phi", "psi"]] = [phi, psi]
                return row

            chain_df = chain_df.apply(calc_phi_psi, axis=1)

            # Set resi.
            chain_df.resi = np.arange(len(chain_df))

            # Write to db.
            #log.info("{}_{} looks good, adding to database.".format(pdb_id, chain.id))

            chain_df.to_sql("raw_data", conn, if_exists="append", index=False)
            cur.execute("INSERT INTO proteins (id, pdb_id, chain, sequence, length, thickness, "
                        "method, resolution, cov_total, cov_trimmed, num_tms, num_res) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        ("{}_{}".format(pdb_id, chain.id), pdb_id, chain.id, chain_df.seqres.str.cat(), len(chain_df), thickness,
                        method, resolution, cov_total, cov_trimmed, len(tms), len(res)))

            conn.commit()

    conn.close()

    log.info("{} key errors".format(key_error))


if __name__ == "__main__":
    main(sys.argv[1:])
