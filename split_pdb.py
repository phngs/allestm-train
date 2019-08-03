#!/usr/bin/env python3.6
import sys
import argparse
import os

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_file")
    parser.add_argument("out_dir")
    parser.add_argument("kf_bin")
    return parser.parse_args(argv)

def read_pdb(pdb_file):
    chains = {}
    pdb_id = ""

    with open(pdb_file) as pdb_fh:
        for line in pdb_fh:
            if line.startswith("HEADER"):
                pdb_id = line.strip().split()[-1].lower()
            elif line.startswith("ATOM"):
                chain_id = "{}_{}".format(pdb_id, line[21])
                res_num = int(line[22:26].strip())

                if chain_id not in chains:
                    chains[chain_id] = []

                chains[chain_id].append([res_num, line.strip()])

    return chains

def write_pdb(chain_data, chain_file):
    with open(chain_file, "w") as fh:
        for (_, line) in chain_data:
            print(line, file=fh)

def write_bin(kf_bin, chain_file, chain_dir, chain_data, kf_file):
    first = chain_data[0][0]
    last = chain_data[-1][0]

    cmd = "python {} -f {} -o {} -l '{}-{}'".format(kf_bin, chain_file, chain_dir, first, last)
    with open(kf_file, "w") as fh:
        print(cmd, file=fh)

def main(argv):
    args = parse_args(argv)

    chains = read_pdb(args.pdb_file)

    for (chain_id, chain_data) in chains.items():
        chain_dir = os.path.join(args.out_dir, chain_id)
        chain_file = os.path.join(chain_dir, "{}.pdb".format(chain_id))
        os.makedirs(chain_dir, exist_ok = True)
        write_pdb(chain_data, chain_file)

        kf_file = os.path.join(chain_dir, "{}.sh".format(chain_id))
        write_bin(os.path.abspath(args.kf_bin), os.path.abspath(chain_file), os.path.abspath(chain_dir), chain_data, os.path.abspath(kf_file))

if __name__ == "__main__":
    main(sys.argv[1:])

