import os
import torch
import warnings
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Chain import Chain
from Bio.PDB import MMCIFParser, PDBParser

warnings.filterwarnings("ignore", module="Bio")

OUTPUT_FILE = "./data/biolip.pt"
STRUCTURE_FOLDER = ""# Folder contains all the complex file from biolip with naming format of: {pdb_id}.pdb or {pdb_id}.cif, in lower case
BIOLIP_META_FILE = "./BioLiP.txt"
THREAD_NUM = 16 # Adjust this number accordingly

BIOLIP_META_HEADER = [
    "pdb_id",
    "receptor_chain",
    "resolution",
    "binding_site",
    "ligand_ccd_id",
    "ligand_chain",
    "ligand_serial_num",
    "binding_site_pdb", # pocket
    "binding_site_reorder",
    "catalyst_site_pdb",
    "catalyst_site_reorder",
    "enzyme_class_id",
    "go_term_id",
    "binding_affinity_literature",
    "binding_affinity_binding_moad",
    "binding_affinity_pdbind_cn",
    "binding_affinity_binding_db",
    "uniprot_db",
    "pubmed_id",
    "ligand_res_num",
    "receptor_seq"
]

# Files known to fail when processing due to either dssp or biopython.
KNOWN_FAIL_RECORDS =  [
    # {'pdb_id': '1ai0', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '1aiy', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '1aw8', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '1epl', 'receptor_chain': 'E', 'ligand_chain': 'I'},
    # {'pdb_id': '1g65', 'receptor_chain': 'K', 'ligand_chain': '3'},
    # {'pdb_id': '1jd2', 'receptor_chain': 'H', 'ligand_chain': '8'},
    # {'pdb_id': '1jpl', 'receptor_chain': 'A', 'ligand_chain': 'E'},
    # {'pdb_id': '1juq', 'receptor_chain': 'A', 'ligand_chain': 'E'},
    # {'pdb_id': '1n6j', 'receptor_chain': 'A', 'ligand_chain': 'G'},
    # {'pdb_id': '1orh', 'receptor_chain': 'A', 'ligand_chain': 'B'},
    # {'pdb_id': '1yk0', 'receptor_chain': 'A', 'ligand_chain': 'E'},
    # {'pdb_id': '2cnn', 'receptor_chain': 'A', 'ligand_chain': 'I'},
    # {'pdb_id': '2hld', 'receptor_chain': 'Y', 'ligand_chain': '1'},
    # {'pdb_id': '2je4', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '2o8m', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '2qas', 'receptor_chain': 'A', 'ligand_chain': 'B'},
    # {'pdb_id': '2qiy', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '2zjp', 'receptor_chain': 'F', 'ligand_chain': '5'},
    # {'pdb_id': '3a0b', 'receptor_chain': 'E', 'ligand_chain': 'N'},
    # {'pdb_id': '3a0h', 'receptor_chain': 'E', 'ligand_chain': 'N'},
    # {'pdb_id': '3brh', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '3bu8', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '3bua', 'receptor_chain': 'A', 'ligand_chain': 'E'},
    # {'pdb_id': '3cf5', 'receptor_chain': 'F', 'ligand_chain': '5'},
    # {'pdb_id': '3e3q', 'receptor_chain': 'a', 'ligand_chain': 'b'},
    # {'pdb_id': '3j92', 'receptor_chain': 'v', 'ligand_chain': 'w'},
    # {'pdb_id': '3j9m', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '3jag', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '3jah', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '3jai', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '3jbu', 'receptor_chain': 's', 'ligand_chain': 'z'},
    # {'pdb_id': '3jbv', 'receptor_chain': 'e', 'ligand_chain': 'z'},
    # {'pdb_id': '3jcu', 'receptor_chain': 'b', 'ligand_chain': 'u'},
    # {'pdb_id': '3lk4', 'receptor_chain': '1', 'ligand_chain': '3'},
    # {'pdb_id': '3lu9', 'receptor_chain': 'B', 'ligand_chain': 'C'},
    # {'pdb_id': '3m5n', 'receptor_chain': 'B', 'ligand_chain': 'F'},
    # {'pdb_id': '3mlt', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '3mn5', 'receptor_chain': 'A', 'ligand_chain': 'S'},
    # {'pdb_id': '3mnw', 'receptor_chain': 'A', 'ligand_chain': 'P'},
    # {'pdb_id': '3mnz', 'receptor_chain': 'A', 'ligand_chain': 'P'},
    # {'pdb_id': '3mt6', 'receptor_chain': 'A', 'ligand_chain': '1'},
    # {'pdb_id': '3nzj', 'receptor_chain': 'K', 'ligand_chain': '3'},
    # {'pdb_id': '3nzw', 'receptor_chain': 'K', 'ligand_chain': '3'},
    # {'pdb_id': '3nzx', 'receptor_chain': 'K', 'ligand_chain': '3'},
    # {'pdb_id': '3oe7', 'receptor_chain': 'Y', 'ligand_chain': '1'},
    # {'pdb_id': '3oee', 'receptor_chain': 'Y', 'ligand_chain': '1'},
    # {'pdb_id': '3oeh', 'receptor_chain': 'Y', 'ligand_chain': '1'},
    # {'pdb_id': '3pl7', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '3pma', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '3pmb', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '3r3g', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '3s7h', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '4ewz', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '4ex0', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '4ex1', 'receptor_chain': 'B', 'ligand_chain': 'A'},
    # {'pdb_id': '4fzc', 'receptor_chain': 'H', 'ligand_chain': 'c'},
    # {'pdb_id': '4fzg', 'receptor_chain': 'H', 'ligand_chain': 'c'},
    # {'pdb_id': '4gk7', 'receptor_chain': 'b', 'ligand_chain': '6'},
    # {'pdb_id': '4jsq', 'receptor_chain': 'K', 'ligand_chain': 'c'},
    # {'pdb_id': '4jsu', 'receptor_chain': 'H', 'ligand_chain': 'e'},
    # {'pdb_id': '4jt0', 'receptor_chain': 'K', 'ligand_chain': 'c'},
    # {'pdb_id': '4l29', 'receptor_chain': 'A', 'ligand_chain': 'm'},
    # {'pdb_id': '4l3c', 'receptor_chain': 'A', 'ligand_chain': 'm'},
    # {'pdb_id': '4pj0', 'receptor_chain': 'a', 'ligand_chain': 't'},
    # {'pdb_id': '4qby', 'receptor_chain': 'H', 'ligand_chain': '1'},
    # {'pdb_id': '4rmi', 'receptor_chain': 'A', 'ligand_chain': 'B'},
    # {'pdb_id': '4tnh', 'receptor_chain': 'e', 'ligand_chain': 'G'},
    # {'pdb_id': '4tni', 'receptor_chain': 'e', 'ligand_chain': 'G'},
    # {'pdb_id': '4tnj', 'receptor_chain': 'e', 'ligand_chain': 'G'},
    # {'pdb_id': '4twt', 'receptor_chain': 'A', 'ligand_chain': 'E'},
    # {'pdb_id': '4u0g', 'receptor_chain': 'A', 'ligand_chain': 'c'},
    # {'pdb_id': '4ub8', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '4uy8', 'receptor_chain': 'S', 'ligand_chain': '7'},
    # {'pdb_id': '4v46', 'receptor_chain': 'A0', 'ligand_chain': 'B0'},
    # {'pdb_id': '4v4j', 'receptor_chain': 'n', 'ligand_chain': 'v'},
    # {'pdb_id': '4v5f', 'receptor_chain': 'BL', 'ligand_chain': 'Bm'},
    # {'pdb_id': '4v5k', 'receptor_chain': 'CL', 'ligand_chain': 'CY'},
    # {'pdb_id': '4v62', 'receptor_chain': 'AE', 'ligand_chain': 'AY'},
    # {'pdb_id': '4v82', 'receptor_chain': 'AE', 'ligand_chain': 'AY'},
    # {'pdb_id': '4v98', 'receptor_chain': 'A2', 'ligand_chain': 'A1'},
    # {'pdb_id': '4v9r', 'receptor_chain': 'AL', 'ligand_chain': 'AW'},
    # {'pdb_id': '4v9s', 'receptor_chain': 'AL', 'ligand_chain': 'AW'},
    # {'pdb_id': '4wqu', 'receptor_chain': 'BL', 'ligand_chain': 'BX'},
    # {'pdb_id': '4wz7', 'receptor_chain': '5', 'ligand_chain': 'AG'},
    # {'pdb_id': '4x6z', 'receptor_chain': 'F', 'ligand_chain': 'e'},
    # {'pdb_id': '4y69', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4y6a', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4y6v', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4y6z', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y70', 'receptor_chain': 'H', 'ligand_chain': 'e'},
    # {'pdb_id': '4y74', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y75', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4y77', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y78', 'receptor_chain': 'b', 'ligand_chain': '6'},
    # {'pdb_id': '4y7w', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y7x', 'receptor_chain': 'K', 'ligand_chain': 'c'},
    # {'pdb_id': '4y7y', 'receptor_chain': 'H', 'ligand_chain': 'c'},
    # {'pdb_id': '4y80', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y81', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4y82', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y84', 'receptor_chain': 'b', 'ligand_chain': 'j'},
    # {'pdb_id': '4y8g', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y8h', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y8i', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y8j', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y8k', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4y8l', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4y8n', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4y8o', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4y8p', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y8q', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4y8s', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4y8t', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4y8u', 'receptor_chain': 'b', 'ligand_chain': '2'},
    # {'pdb_id': '4y9z', 'receptor_chain': 'b', 'ligand_chain': '6'},
    # {'pdb_id': '4ya0', 'receptor_chain': 'b', 'ligand_chain': '2'},
    # {'pdb_id': '4ya2', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4ya3', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4ya5', 'receptor_chain': 'b', 'ligand_chain': 'd'},
    # {'pdb_id': '4ya7', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '4ya9', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '4yuu', 'receptor_chain': 'a1', 'ligand_chain': 'f1'},
    # {'pdb_id': '5aj4', 'receptor_chain': 'Ah', 'ligand_chain': 'Az'},
    # {'pdb_id': '5cdw', 'receptor_chain': 'A', 'ligand_chain': 'I'},
    # {'pdb_id': '5cgg', 'receptor_chain': 'K', 'ligand_chain': 'g'},
    # {'pdb_id': '5cgh', 'receptor_chain': 'K', 'ligand_chain': 'e'},
    # {'pdb_id': '5dkp', 'receptor_chain': 'A', 'ligand_chain': 'O'},
    # {'pdb_id': '5e79', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '5gad', 'receptor_chain': 'i', 'ligand_chain': 'k'},
    # {'pdb_id': '5gae', 'receptor_chain': 'g', 'ligand_chain': 'i'},
    # {'pdb_id': '5gaf', 'receptor_chain': 'i', 'ligand_chain': 'k'},
    # {'pdb_id': '5gag', 'receptor_chain': 'i', 'ligand_chain': 'k'},
    # {'pdb_id': '5gah', 'receptor_chain': 'i', 'ligand_chain': 'k'},
    # {'pdb_id': '5gm6', 'receptor_chain': 'A', 'ligand_chain': 'X'},
    # {'pdb_id': '5gti', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '5hau', 'receptor_chain': '1F', 'ligand_chain': '1x'},
    # {'pdb_id': '5htc', 'receptor_chain': 'A', 'ligand_chain': 'B'},
    # {'pdb_id': '5ley', 'receptor_chain': 'K', 'ligand_chain': 'c'},
    # {'pdb_id': '5lez', 'receptor_chain': 'b', 'ligand_chain': 'f'},
    # {'pdb_id': '5lf0', 'receptor_chain': 'b', 'ligand_chain': 'h'},
    # {'pdb_id': '5lzp', 'receptor_chain': '0', 'ligand_chain': 'X'},
    # {'pdb_id': '5lzt', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '5lzv', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '5lzw', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '5mx7', 'receptor_chain': 'A1', 'ligand_chain': 'B1'},
    # {'pdb_id': '5nco', 'receptor_chain': 'i', 'ligand_chain': 'k'},
    # {'pdb_id': '5nif', 'receptor_chain': 'E', 'ligand_chain': '4'},
    # {'pdb_id': '5nmb', 'receptor_chain': 'A2', 'ligand_chain': 'B2'},
    # {'pdb_id': '5okz', 'receptor_chain': 'a', 'ligand_chain': 'd'},
    # {'pdb_id': '5ool', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '5oom', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '5twh', 'receptor_chain': 'A', 'ligand_chain': 'E'},
    # {'pdb_id': '5ws5', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '5ws6', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '5wvk', 'receptor_chain': 'R', 'ligand_chain': 'Y'},
    # {'pdb_id': '5xnm', 'receptor_chain': 'b', 'ligand_chain': 'u'},
    # {'pdb_id': '5yq7', 'receptor_chain': 'L', 'ligand_chain': 'Y'},
    # {'pdb_id': '5zf0', 'receptor_chain': 'B1', 'ligand_chain': 'X1'},
    # {'pdb_id': '5zwm', 'receptor_chain': 'X', 'ligand_chain': 'Z'},
    # {'pdb_id': '5zzn', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '6bgl', 'receptor_chain': 'A', 'ligand_chain': 'l'},
    # {'pdb_id': '6bgo', 'receptor_chain': 'C', 'ligand_chain': 'i'},
    # {'pdb_id': '6c0f', 'receptor_chain': 'b', 'ligand_chain': 'x'},
    # {'pdb_id': '6cae', 'receptor_chain': '1O', 'ligand_chain': 'B'},
    # {'pdb_id': '6cb1', 'receptor_chain': 'b', 'ligand_chain': 'x'},
    # {'pdb_id': '6f1t', 'receptor_chain': 'F', 'ligand_chain': 'd'},
    # {'pdb_id': '6frk', 'receptor_chain': 'x', 'ligand_chain': 't'},
    # {'pdb_id': '6fti', 'receptor_chain': '3', 'ligand_chain': '0'},
    # {'pdb_id': '6ftj', 'receptor_chain': 'x', 'ligand_chain': 'z'},
    # {'pdb_id': '6g5k', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '6gaw', 'receptor_chain': 'Ah', 'ligand_chain': 'AZ'},
    # {'pdb_id': '6gaz', 'receptor_chain': 'Ae', 'ligand_chain': 'BT'},
    # {'pdb_id': '6gb2', 'receptor_chain': 'BJ', 'ligand_chain': 'DL'},
    # {'pdb_id': '6giq', 'receptor_chain': 'a', 'ligand_chain': 'g'},
    # {'pdb_id': '6gwt', 'receptor_chain': 'E', 'ligand_chain': 'z'},
    # {'pdb_id': '6gxm', 'receptor_chain': 'E', 'ligand_chain': 'z'},
    # {'pdb_id': '6gxn', 'receptor_chain': 'E', 'ligand_chain': 'z'},
    # {'pdb_id': '6hiw', 'receptor_chain': 'Cv', 'ligand_chain': 'UR'},
    # {'pdb_id': '6hix', 'receptor_chain': 'A1', 'ligand_chain': 'UI'},
    # {'pdb_id': '6hiy', 'receptor_chain': 'Cv', 'ligand_chain': 'UR'},
    # {'pdb_id': '6hiz', 'receptor_chain': 'DB', 'ligand_chain': 'UP'},
    # {'pdb_id': '6j2q', 'receptor_chain': 'R', 'ligand_chain': 'Y'},
    # {'pdb_id': '6j2x', 'receptor_chain': 'R', 'ligand_chain': 'Y'},
    # {'pdb_id': '6j30', 'receptor_chain': 'R', 'ligand_chain': 'Y'},
    # {'pdb_id': '6j3y', 'receptor_chain': 'a', 'ligand_chain': 'f'},
    # {'pdb_id': '6j3z', 'receptor_chain': 'a', 'ligand_chain': 'f'},
    # {'pdb_id': '6j40', 'receptor_chain': 'a', 'ligand_chain': 'f'},
    # {'pdb_id': '6jeo', 'receptor_chain': 'aB', 'ligand_chain': 'aX'},
    # {'pdb_id': '6jlj', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '6jlk', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '6jll', 'receptor_chain': 'j', 'ligand_chain': 'y'},
    # {'pdb_id': '6jln', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '6jlo', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '6jlp', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '6k33', 'receptor_chain': 'aB', 'ligand_chain': 'aX'},
    # {'pdb_id': '6kac', 'receptor_chain': 'b', 'ligand_chain': 'u'},
    # {'pdb_id': '6lkq', 'receptor_chain': 'K', 'ligand_chain': 'z'},
    # {'pdb_id': '6lqv', 'receptor_chain': 'B1', 'ligand_chain': 'X1'},
    # {'pdb_id': '6ly5', 'receptor_chain': 'b', 'ligand_chain': 'm'},
    # {'pdb_id': '6mtd', 'receptor_chain': 'KK', 'ligand_chain': 'w'},
    # {'pdb_id': '6n1d', 'receptor_chain': 'BS13', 'ligand_chain': 'BTHX'},
    # {'pdb_id': '6nah', 'receptor_chain': 'a', 'ligand_chain': '3'},
    # {'pdb_id': '6nu2', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '6nu3', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '6olz', 'receptor_chain': 'AC', 'ligand_chain': 'A'},
    # {'pdb_id': '6om0', 'receptor_chain': 'C', 'ligand_chain': 'y'},
    # {'pdb_id': '6om7', 'receptor_chain': 'C', 'ligand_chain': 'y'},
    # {'pdb_id': '6pag', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '6pep', 'receptor_chain': 'A', 'ligand_chain': 'X'},
    # {'pdb_id': '6pnj', 'receptor_chain': 'b', 'ligand_chain': 'x'},
    # {'pdb_id': '6q38', 'receptor_chain': 'A', 'ligand_chain': 'C'},
    # {'pdb_id': '6q3g', 'receptor_chain': 'a1', 'ligand_chain': 'd1'},
    # {'pdb_id': '6q9d', 'receptor_chain': 'A7', 'ligand_chain': 'AM'},
    # {'pdb_id': '6qbx', 'receptor_chain': 'a2', 'ligand_chain': 'x1'},
    # {'pdb_id': '6qc3', 'receptor_chain': 'a1', 'ligand_chain': 'x1'},
    # {'pdb_id': '6qdw', 'receptor_chain': 'e', 'ligand_chain': 'z'},
    # {'pdb_id': '6qza', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '6qzc', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '6qzd', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '6r0e', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '6r6g', 'receptor_chain': 'AB', 'ligand_chain': 'AG'},
    # {'pdb_id': '6r6p', 'receptor_chain': 'C', 'ligand_chain': '1'},
    # {'pdb_id': '6r7q', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '6sg9', 'receptor_chain': 'CC', 'ligand_chain': 'Uf'},
    # {'pdb_id': '6sga', 'receptor_chain': 'Ca', 'ligand_chain': 'UU'},
    # {'pdb_id': '6sgb', 'receptor_chain': 'Cb', 'ligand_chain': 'UA'},
    # {'pdb_id': '6sh2', 'receptor_chain': 'AAA', 'ligand_chain': 'DDD'},
    # {'pdb_id': '6t59', 'receptor_chain': 'R3', 'ligand_chain': 'NI'},
    # {'pdb_id': '6t9m', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6tb9', 'receptor_chain': 'B2', 'ligand_chain': 'F2'},
    # {'pdb_id': '6tba', 'receptor_chain': 'B2', 'ligand_chain': 'F2'},
    # {'pdb_id': '6tf9', 'receptor_chain': 'eP1', 'ligand_chain': 'NP1'},
    # {'pdb_id': '6tg8', 'receptor_chain': 'AAA', 'ligand_chain': 'PPP'},
    # {'pdb_id': '6tka', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6toq', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '6trc', 'receptor_chain': '2', 'ligand_chain': 'z'},
    # {'pdb_id': '6trd', 'receptor_chain': '2', 'ligand_chain': 'z'},
    # {'pdb_id': '6tsl', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6tsm', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6tsu', 'receptor_chain': 'A2', 'ligand_chain': 'F2'},
    # {'pdb_id': '6tvq', 'receptor_chain': 'AaA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6tvu', 'receptor_chain': 'AaA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6tvw', 'receptor_chain': 'CCC', 'ligand_chain': 'DDD'},
    # {'pdb_id': '6twz', 'receptor_chain': 'A', 'ligand_chain': 'D000'},
    # {'pdb_id': '6txs', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6u42', 'receptor_chain': '5H', 'ligand_chain': '5R'},
    # {'pdb_id': '6u48', 'receptor_chain': 'CE', 'ligand_chain': 'A'},
    # {'pdb_id': '6v39', 'receptor_chain': 'k', 'ligand_chain': 'u'},
    # {'pdb_id': '6v3a', 'receptor_chain': 'k', 'ligand_chain': 'u'},
    # {'pdb_id': '6v3b', 'receptor_chain': 'k', 'ligand_chain': 'u'},
    # {'pdb_id': '6v41', 'receptor_chain': 'AAA', 'ligand_chain': 'QQQ'},
    # {'pdb_id': '6v8i', 'receptor_chain': 'AE', 'ligand_chain': 'CF'},
    # {'pdb_id': '6v8w', 'receptor_chain': 'A', 'ligand_chain': 'AA'},
    # {'pdb_id': '6vlz', 'receptor_chain': 'I', 'ligand_chain': 'TB'},
    # {'pdb_id': '6vmi', 'receptor_chain': 'I', 'ligand_chain': 'TB'},
    # {'pdb_id': '6w1s', 'receptor_chain': 'G', 'ligand_chain': 'N'},
    # {'pdb_id': '6w6l', 'receptor_chain': '1', 'ligand_chain': '3'},
    # {'pdb_id': '6wat', 'receptor_chain': 'AA', 'ligand_chain': 'A'},
    # {'pdb_id': '6ws0', 'receptor_chain': 'CCC', 'ligand_chain': 'ZZZ'},
    # {'pdb_id': '6ws5', 'receptor_chain': 'CCC', 'ligand_chain': 'ZZZ'},
    # {'pdb_id': '6x6h', 'receptor_chain': 'A1', 'ligand_chain': 'P'},
    # {'pdb_id': '6x89', 'receptor_chain': 'A7', 'ligand_chain': 'A'},
    # {'pdb_id': '6xa1', 'receptor_chain': 'LC', 'ligand_chain': 'NC'},
    # {'pdb_id': '6xir', 'receptor_chain': 'AQ', 'ligand_chain': 'n'},
    # {'pdb_id': '6xxf', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6xxx', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6xy3', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6xyw', 'receptor_chain': 'Bw', 'ligand_chain': 'Bt'},
    # {'pdb_id': '6y8k', 'receptor_chain': 'AAA', 'ligand_chain': 'PPP'},
    # {'pdb_id': '6yd2', 'receptor_chain': 'A', 'ligand_chain': '611'},
    # {'pdb_id': '6yd3', 'receptor_chain': 'A', 'ligand_chain': '611'},
    # {'pdb_id': '6ydp', 'receptor_chain': 'Ah', 'ligand_chain': 'AZ'},
    # {'pdb_id': '6ydw', 'receptor_chain': 'Ah', 'ligand_chain': 'AZ'},
    # {'pdb_id': '6yh0', 'receptor_chain': 'AAA', 'ligand_chain': 'EEE'},
    # {'pdb_id': '6yn1', 'receptor_chain': 'a', 'ligand_chain': 'Y'},
    # {'pdb_id': '6ynx', 'receptor_chain': 'i2', 'ligand_chain': 'i1'},
    # {'pdb_id': '6yss', 'receptor_chain': 'E', 'ligand_chain': 'v'},
    # {'pdb_id': '6yst', 'receptor_chain': 'E', 'ligand_chain': 'v'},
    # {'pdb_id': '6yvr', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '6yxm', 'receptor_chain': 'HHH', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6yxx', 'receptor_chain': 'BJ', 'ligand_chain': 'Ur'},
    # {'pdb_id': '6yxy', 'receptor_chain': 'A8', 'ligand_chain': 'UM'},
    # {'pdb_id': '6yzf', 'receptor_chain': 'CCC', 'ligand_chain': 'FFF'},
    # {'pdb_id': '6z4v', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '6zjg', 'receptor_chain': 'HHH', 'ligand_chain': 'B'},
    # {'pdb_id': '6zm7', 'receptor_chain': 'SC', 'ligand_chain': 'CF'},
    # {'pdb_id': '6zme', 'receptor_chain': 'SC', 'ligand_chain': 'CF'},
    # {'pdb_id': '6zmt', 'receptor_chain': 'D', 'ligand_chain': 'i'},
    # {'pdb_id': '6zn5', 'receptor_chain': 'D', 'ligand_chain': 'i'},
    # {'pdb_id': '6zs9', 'receptor_chain': 't3', 'ligand_chain': 't4'},
    # {'pdb_id': '6zsa', 'receptor_chain': 't1', 'ligand_chain': 't4'},
    # {'pdb_id': '6zsb', 'receptor_chain': 't1', 'ligand_chain': 't4'},
    # {'pdb_id': '6zsc', 'receptor_chain': 't1', 'ligand_chain': 't4'},
    # {'pdb_id': '6zsd', 'receptor_chain': 't1', 'ligand_chain': 't4'},
    # {'pdb_id': '6zsg', 'receptor_chain': 't1', 'ligand_chain': 't4'},
    # {'pdb_id': '6zvn', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '7a23', 'receptor_chain': 'a', 'ligand_chain': 'h'},
    # {'pdb_id': '7a24', 'receptor_chain': 'c', 'ligand_chain': 'n'},
    # {'pdb_id': '7a5f', 'receptor_chain': 'I3', 'ligand_chain': 'l3'},
    # {'pdb_id': '7a5g', 'receptor_chain': 'I3', 'ligand_chain': 'l3'},
    # {'pdb_id': '7a5h', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7a5i', 'receptor_chain': 'I3', 'ligand_chain': 'l3'},
    # {'pdb_id': '7a5j', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7a5k', 'receptor_chain': 'I3', 'ligand_chain': 'l3'},
    # {'pdb_id': '7a8y', 'receptor_chain': 'BBB', 'ligand_chain': 'AAA'},
    # {'pdb_id': '7aew', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7agx', 'receptor_chain': '2F', 'ligand_chain': '2Q'},
    # {'pdb_id': '7aks', 'receptor_chain': 'AAA', 'ligand_chain': 'BaB'},
    # {'pdb_id': '7ane', 'receptor_chain': 'o', 'ligand_chain': 'ba'},
    # {'pdb_id': '7anm', 'receptor_chain': 'A', 'ligand_chain': 'aa'},
    # {'pdb_id': '7aoi', 'receptor_chain': 'A2', 'ligand_chain': 'UB'},
    # {'pdb_id': '7ar4', 'receptor_chain': 'AAA', 'ligand_chain': 'PaP'},
    # {'pdb_id': '7ar7', 'receptor_chain': 'V', 'ligand_chain': 'r'},
    # {'pdb_id': '7ar8', 'receptor_chain': 'V', 'ligand_chain': 'r'},
    # {'pdb_id': '7arb', 'receptor_chain': 'V', 'ligand_chain': 'r'},
    # {'pdb_id': '7ard', 'receptor_chain': 'G', 'ligand_chain': 'q'},
    # {'pdb_id': '7bbp', 'receptor_chain': 'AAA', 'ligand_chain': 'GGG'},
    # {'pdb_id': '7bf1', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7bf2', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7bim', 'receptor_chain': 'a', 'ligand_chain': 'b'},
    # {'pdb_id': '7bn2', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7bns', 'receptor_chain': 'A2', 'ligand_chain': 'B2'},
    # {'pdb_id': '7bnu', 'receptor_chain': 'A2', 'ligand_chain': 'B2'},
    # {'pdb_id': '7cgo', 'receptor_chain': 'a', 'ligand_chain': '5'},
    # {'pdb_id': '7coy', 'receptor_chain': 'aB', 'ligand_chain': 'aM'},
    # {'pdb_id': '7d1t', 'receptor_chain': 'j', 'ligand_chain': 'y'},
    # {'pdb_id': '7d1u', 'receptor_chain': 'j', 'ligand_chain': 'y'},
    # {'pdb_id': '7dco', 'receptor_chain': 'A', 'ligand_chain': 'U'},
    # {'pdb_id': '7dr2', 'receptor_chain': 'aB', 'ligand_chain': 'aM'},
    # {'pdb_id': '7dvq', 'receptor_chain': '3', 'ligand_chain': 'y'},
    # {'pdb_id': '7e80', 'receptor_chain': 'a', 'ligand_chain': '5'},
    # {'pdb_id': '7e81', 'receptor_chain': 'Ca', 'ligand_chain': 'GE'},
    # {'pdb_id': '7e82', 'receptor_chain': 'a', 'ligand_chain': '5'},
    # {'pdb_id': '7f4v', 'receptor_chain': 'aB', 'ligand_chain': 'aM'},
    # {'pdb_id': '7fix', 'receptor_chain': 'B1', 'ligand_chain': 'X1'},
    # {'pdb_id': '7l08', 'receptor_chain': 'I', 'ligand_chain': 'TB'},
    # {'pdb_id': '7l20', 'receptor_chain': 'I', 'ligand_chain': 'TB'},
    # {'pdb_id': '7lki', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7miz', 'receptor_chain': 'A4', 'ligand_chain': '0'},
    # {'pdb_id': '7muc', 'receptor_chain': 'AL', 'ligand_chain': 'MU'},
    # {'pdb_id': '7mud', 'receptor_chain': 'AL', 'ligand_chain': 'MU'},
    # {'pdb_id': '7muq', 'receptor_chain': 'AL', 'ligand_chain': 'MU'},
    # {'pdb_id': '7mus', 'receptor_chain': 'AK', 'ligand_chain': 'AU'},
    # {'pdb_id': '7muv', 'receptor_chain': 'AL', 'ligand_chain': 'MU'},
    # {'pdb_id': '7muw', 'receptor_chain': 'AL', 'ligand_chain': 'MU'},
    # {'pdb_id': '7muy', 'receptor_chain': 'AK', 'ligand_chain': 'AU'},
    # {'pdb_id': '7n6g', 'receptor_chain': '5b', 'ligand_chain': '2M'},
    # {'pdb_id': '7ndq', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7ndt', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7ndu', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7nfx', 'receptor_chain': 'B', 'ligand_chain': 'z'},
    # {'pdb_id': '7nmz', 'receptor_chain': 'AA', 'ligand_chain': 'C'},
    # {'pdb_id': '7nqh', 'receptor_chain': 'Ai', 'ligand_chain': 'AZ'},
    # {'pdb_id': '7nql', 'receptor_chain': 'Ah', 'ligand_chain': 'AZ'},
    # {'pdb_id': '7ns3', 'receptor_chain': '4', 'ligand_chain': 'Fb'},
    # {'pdb_id': '7nsh', 'receptor_chain': 'BJ', 'ligand_chain': 'DL'},
    # {'pdb_id': '7nsi', 'receptor_chain': 'Ah', 'ligand_chain': 'AZ'},
    # {'pdb_id': '7nsj', 'receptor_chain': 'Ah', 'ligand_chain': 'AZ'},
    # {'pdb_id': '7nvr', 'receptor_chain': '7', 'ligand_chain': 'Y'},
    # {'pdb_id': '7nw1', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7nwg', 'receptor_chain': 'P3', 'ligand_chain': '1'},
    # {'pdb_id': '7nww', 'receptor_chain': 'E', 'ligand_chain': 'B'},
    # {'pdb_id': '7nze', 'receptor_chain': 'AAA', 'ligand_chain': 'FFF'},
    # {'pdb_id': '7nzf', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7nzh', 'receptor_chain': 'AAA', 'ligand_chain': 'EEE'},
    # {'pdb_id': '7o00', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7o50', 'receptor_chain': 'A', 'ligand_chain': 'H'},
    # {'pdb_id': '7o9k', 'receptor_chain': 'I', 'ligand_chain': 't4'},
    # {'pdb_id': '7o9m', 'receptor_chain': 'I', 'ligand_chain': 't6'},
    # {'pdb_id': '7odv', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7of0', 'receptor_chain': '8', 'ligand_chain': 'm'},
    # {'pdb_id': '7of2', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7of3', 'receptor_chain': '8', 'ligand_chain': 'm'},
    # {'pdb_id': '7of5', 'receptor_chain': '8', 'ligand_chain': 'm'},
    # {'pdb_id': '7of6', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7of7', 'receptor_chain': '8', 'ligand_chain': 'm'},
    # {'pdb_id': '7og1', 'receptor_chain': 'MMM', 'ligand_chain': 'PPP'},
    # {'pdb_id': '7og4', 'receptor_chain': 't1', 'ligand_chain': 't4'},
    # {'pdb_id': '7ogo', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7ogq', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7ogu', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7ogz', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7ohx', 'receptor_chain': 'm', 'ligand_chain': 'i'},
    # {'pdb_id': '7oi7', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oi8', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oi9', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oia', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oib', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oic', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oid', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oie', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '7oif', 'receptor_chain': 'R', 'ligand_chain': 'B'},
    # {'pdb_id': '7oig', 'receptor_chain': 'E', 'ligand_chain': 'B'},
    # {'pdb_id': '7oiq', 'receptor_chain': 'AAA', 'ligand_chain': 'DDD'},
    # {'pdb_id': '7oit', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '7ok6', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7opc', 'receptor_chain': 'e', 'ligand_chain': 'f'},
    # {'pdb_id': '7opd', 'receptor_chain': 'e', 'ligand_chain': 'f'},
    # {'pdb_id': '7oq8', 'receptor_chain': 'A', 'ligand_chain': 'B'},
    # {'pdb_id': '7oui', 'receptor_chain': 'a', 'ligand_chain': 't'},
    # {'pdb_id': '7oya', 'receptor_chain': 'C1', 'ligand_chain': 's1'},
    # {'pdb_id': '7oyd', 'receptor_chain': 'C', 'ligand_chain': 's1'},
    # {'pdb_id': '7oyn', 'receptor_chain': 'A', 'ligand_chain': 'B'},
    # {'pdb_id': '7p5u', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7p81', 'receptor_chain': 'A', 'ligand_chain': 'c'},
    # {'pdb_id': '7pbc', 'receptor_chain': 'AAA', 'ligand_chain': 'EEE'},
    # {'pdb_id': '7pd3', 'receptor_chain': 'G', 'ligand_chain': 'l'},
    # {'pdb_id': '7pdw', 'receptor_chain': 'AAA', 'ligand_chain': 'EEE'},
    # {'pdb_id': '7pi5', 'receptor_chain': 'b', 'ligand_chain': 'u'},
    # {'pdb_id': '7pin', 'receptor_chain': 'b1', 'ligand_chain': 'u1'},
    # {'pdb_id': '7piw', 'receptor_chain': 'b1', 'ligand_chain': 'u1'},
    # {'pdb_id': '7pkq', 'receptor_chain': 'B', 'ligand_chain': 'u'},
    # {'pdb_id': '7pnk', 'receptor_chain': 'b', 'ligand_chain': 'u'},
    # {'pdb_id': '7po4', 'receptor_chain': 'I', 'ligand_chain': 'v'},
    # {'pdb_id': '7pua', 'receptor_chain': 'Ca', 'ligand_chain': 'DZ'},
    # {'pdb_id': '7pub', 'receptor_chain': 'Cn', 'ligand_chain': 'UG'},
    # {'pdb_id': '7pzn', 'receptor_chain': 'C', 'ligand_chain': 'M'},
    # {'pdb_id': '7q4t', 'receptor_chain': 'AAA', 'ligand_chain': 'LbL'},
    # {'pdb_id': '7q5t', 'receptor_chain': 'AAA', 'ligand_chain': 'JJJ'},
    # {'pdb_id': '7q5u', 'receptor_chain': 'AAA', 'ligand_chain': 'LLL'},
    # {'pdb_id': '7q5w', 'receptor_chain': 'AAA', 'ligand_chain': 'JJJ'},
    # {'pdb_id': '7q6i', 'receptor_chain': 'A', 'ligand_chain': 'X'},
    # {'pdb_id': '7q8d', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8f', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8g', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8h', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8i', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8j', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8k', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8l', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8m', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8n', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8o', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8p', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q8q', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7q9b', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7q9c', 'receptor_chain': 'AA', 'ligand_chain': 'PAA'},
    # {'pdb_id': '7q9h', 'receptor_chain': 'AA', 'ligand_chain': 'PAA'},
    # {'pdb_id': '7q9s', 'receptor_chain': 'AAA', 'ligand_chain': 'CCC'},
    # {'pdb_id': '7qcq', 'receptor_chain': 'AAA', 'ligand_chain': 'BBB'},
    # {'pdb_id': '7qf9', 'receptor_chain': 'AAA', 'ligand_chain': 'EEE'},
    # {'pdb_id': '7qff', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7qfh', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7qh7', 'receptor_chain': '6', 'ligand_chain': 'f'},
    # {'pdb_id': '7qhj', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7qhk', 'receptor_chain': 'AA', 'ligand_chain': 'PA'},
    # {'pdb_id': '7qns', 'receptor_chain': 'AA', 'ligand_chain': 'PC'},
    # {'pdb_id': '7qo2', 'receptor_chain': 'AA', 'ligand_chain': 'PAA'},
    # {'pdb_id': '7r1o', 'receptor_chain': 'AAA', 'ligand_chain': 'FFF'},
    # {'pdb_id': '7r4g', 'receptor_chain': 'h', 'ligand_chain': 'Y'},
    # {'pdb_id': '7rf4', 'receptor_chain': 'e', 'ligand_chain': 'r'},
    # {'pdb_id': '7rf6', 'receptor_chain': 'e', 'ligand_chain': 'r'},
    # {'pdb_id': '7rro', 'receptor_chain': 'D6', 'ligand_chain': 'D5'},
    # {'pdb_id': '7spb', 'receptor_chain': 'C10', 'ligand_chain': 'E10'},
    # {'pdb_id': '7spi', 'receptor_chain': 'C10', 'ligand_chain': 'E10'},
    # {'pdb_id': '7tgh', 'receptor_chain': '3c', 'ligand_chain': '3M'},
    # {'pdb_id': '7tm3', 'receptor_chain': '1', 'ligand_chain': '2'},
    # {'pdb_id': '7too', 'receptor_chain': 'AL17', 'ligand_chain': 'AGR'},
    # {'pdb_id': '7top', 'receptor_chain': 'AL17', 'ligand_chain': 'PR'},
    # {'pdb_id': '7tor', 'receptor_chain': 'AL24', 'ligand_chain': 'GR2'},
    # {'pdb_id': '7tut', 'receptor_chain': '1', 'ligand_chain': '2'},
    # {'pdb_id': '7u0h', 'receptor_chain': 'b', 'ligand_chain': 'm'},
    # {'pdb_id': '7u8c', 'receptor_chain': 'H', 'ligand_chain': 'BA2'},
    # {'pdb_id': '7ug7', 'receptor_chain': 'EF', 'ligand_chain': 'B'},
    # {'pdb_id': '7uif', 'receptor_chain': 'D', 'ligand_chain': 'z'},
    # {'pdb_id': '7uio', 'receptor_chain': 'AD', 'ligand_chain': 'Az'},
    # {'pdb_id': '7ung', 'receptor_chain': '0', 'ligand_chain': 'O'},
    # {'pdb_id': '7vd5', 'receptor_chain': 'a', 'ligand_chain': 'f'},
    # {'pdb_id': '7w3b', 'receptor_chain': 'A', 'ligand_chain': 'v'},
    # {'pdb_id': '7wtm', 'receptor_chain': 'SJ', 'ligand_chain': 'Se'},
    # {'pdb_id': '7x8x', 'receptor_chain': 'a', 'ligand_chain': '2'},
    # {'pdb_id': '7xxf', 'receptor_chain': '0', 'ligand_chain': 'a'},
    # {'pdb_id': '7y5e', 'receptor_chain': 'a6', 'ligand_chain': 'N6'},
    # {'pdb_id': '7y7a', 'receptor_chain': 'a9', 'ligand_chain': 'N9'},
    # {'pdb_id': '7y8r', 'receptor_chain': 'D', 'ligand_chain': 'W'},
    # {'pdb_id': '7yk5', 'receptor_chain': 'A', 'ligand_chain': 'b'},
    # {'pdb_id': '7ymi', 'receptor_chain': 'a', 'ligand_chain': 't'},
    # {'pdb_id': '7ymm', 'receptor_chain': '1A', 'ligand_chain': '1T'},
    # {'pdb_id': '7z34', 'receptor_chain': 'm', 'ligand_chain': '0'},
    # {'pdb_id': '7z43', 'receptor_chain': 'AAA', 'ligand_chain': 'XXX'},
    # {'pdb_id': '7z4o', 'receptor_chain': 'AAA', 'ligand_chain': 'KKK'},
    # {'pdb_id': '8a22', 'receptor_chain': 'Aq', 'ligand_chain': 'Xi'},
    # {'pdb_id': '8agu', 'receptor_chain': 'l', 'ligand_chain': '1'},
    # {'pdb_id': '8agv', 'receptor_chain': 'l', 'ligand_chain': '1'},
    # {'pdb_id': '8agw', 'receptor_chain': 'l', 'ligand_chain': '1'},
    # {'pdb_id': '8agx', 'receptor_chain': 'l', 'ligand_chain': '1'},
    # {'pdb_id': '8agz', 'receptor_chain': 'l', 'ligand_chain': '1'},
    # {'pdb_id': '8akn', 'receptor_chain': 'e', 'ligand_chain': 'A'},
    # {'pdb_id': '8am9', 'receptor_chain': 'e', 'ligand_chain': 'A'},
    # {'pdb_id': '8apn', 'receptor_chain': 'Aq', 'ligand_chain': 'Xi'},
    # {'pdb_id': '8apo', 'receptor_chain': 'Aq', 'ligand_chain': 'Xi'},
    # {'pdb_id': '8axk', 'receptor_chain': 'H', 'ligand_chain': 'k0'},
    # {'pdb_id': '8b3o', 'receptor_chain': 'KKK', 'ligand_chain': 'PPP'},
    # {'pdb_id': '8b3p', 'receptor_chain': 'FFF', 'ligand_chain': 'AAA'},
    # {'pdb_id': '8b7y', 'receptor_chain': 'a', 'ligand_chain': 'z'},
    # {'pdb_id': '8bpo', 'receptor_chain': 'T2', 'ligand_chain': 'D1'},
    # {'pdb_id': '8bpx', 'receptor_chain': 'AB', 'ligand_chain': 'AJ'},
    # {'pdb_id': '8bq5', 'receptor_chain': 'AB', 'ligand_chain': 'AJ'},
    # {'pdb_id': '8bq6', 'receptor_chain': 'AB', 'ligand_chain': 'AJ'},
    # {'pdb_id': '8btk', 'receptor_chain': 'SX', 'ligand_chain': 'SZ'},
    # {'pdb_id': '8bvq', 'receptor_chain': 'A', 'ligand_chain': 'G'},
    # {'pdb_id': '8bvw', 'receptor_chain': '0', 'ligand_chain': 'Y'},
    # {'pdb_id': '8bw1', 'receptor_chain': 'H', 'ligand_chain': 'e'},
    # {'pdb_id': '8bzl', 'receptor_chain': 'A', 'ligand_chain': 'h'},
    # {'pdb_id': '8c28', 'receptor_chain': 'AAA', 'ligand_chain': 'DDD'},
    # {'pdb_id': '8c29', 'receptor_chain': 'b', 'ligand_chain': 'u'},
    # {'pdb_id': '8c2d', 'receptor_chain': 'AAA', 'ligand_chain': 'PPP'},
    # {'pdb_id': '8c30', 'receptor_chain': 'AAA', 'ligand_chain': 'PPP'},
    # {'pdb_id': '8c3d', 'receptor_chain': 'A', 'ligand_chain': 'LIG'},
    # {'pdb_id': '8ceu', 'receptor_chain': 'c', 'ligand_chain': 'I'},
    # {'pdb_id': '8ckb', 'receptor_chain': 'A050', 'ligand_chain': 'A499'},
    # {'pdb_id': '8cmn', 'receptor_chain': 'AA', 'ligand_chain': 'B'},
    # {'pdb_id': '8d6v', 'receptor_chain': 'C', 'ligand_chain': 'i'},
    # {'pdb_id': '8d6w', 'receptor_chain': 'C', 'ligand_chain': 'i'},
    # {'pdb_id': '8d6x', 'receptor_chain': 'G', 'ligand_chain': 'f'},
    # {'pdb_id': '8e73', 'receptor_chain': 'A', 'ligand_chain': 'K'},
    # {'pdb_id': '8ekf', 'receptor_chain': 'HHH', 'ligand_chain': 'CCC'},
    # {'pdb_id': '8esr', 'receptor_chain': 'F', 'ligand_chain': 'T'},
    # {'pdb_id': '8esw', 'receptor_chain': 'S4', 'ligand_chain': 'V3'},
    # {'pdb_id': '8esz', 'receptor_chain': 'V1', 'ligand_chain': 'V3'},
    # {'pdb_id': '8etc', 'receptor_chain': 'C', 'ligand_chain': 'T'},
    # {'pdb_id': '8etg', 'receptor_chain': 'F', 'ligand_chain': 'T'},
    # {'pdb_id': '8eth', 'receptor_chain': 'F', 'ligand_chain': 'T'},
    # {'pdb_id': '8eti', 'receptor_chain': 'F', 'ligand_chain': 'T'},
    # {'pdb_id': '8etj', 'receptor_chain': 'C', 'ligand_chain': 'T'},
    # {'pdb_id': '8eup', 'receptor_chain': 'F', 'ligand_chain': 'T'},
    # {'pdb_id': '8euy', 'receptor_chain': 'C', 'ligand_chain': 'T'},
    # {'pdb_id': '8ev3', 'receptor_chain': 'F', 'ligand_chain': 'T'},
    # {'pdb_id': '8fkp', 'receptor_chain': 'SH', 'ligand_chain': 'LH'},
    # {'pdb_id': '8fkq', 'receptor_chain': 'SH', 'ligand_chain': 'LH'},
    # {'pdb_id': '8fl2', 'receptor_chain': 'NU', 'ligand_chain': 'BD'},
    # {'pdb_id': '8fl3', 'receptor_chain': 'NU', 'ligand_chain': 'BD'},
    # {'pdb_id': '8fl4', 'receptor_chain': 'NU', 'ligand_chain': 'BD'},
    # {'pdb_id': '8fwg', 'receptor_chain': 'c', 'ligand_chain': 'f5'},
    # {'pdb_id': '8fxp', 'receptor_chain': '0G', 'ligand_chain': 'AT'},
    # {'pdb_id': '8fxr', 'receptor_chain': 'c', 'ligand_chain': 'f5'},
    # {'pdb_id': '8glv', 'receptor_chain': 'Da', 'ligand_chain': 'zi'},
    # {'pdb_id': '8gn1', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '8gn2', 'receptor_chain': 'k', 'ligand_chain': 'y'},
    # {'pdb_id': '8gra', 'receptor_chain': 'G', 'ligand_chain': 'J2'},
    # {'pdb_id': '8gym', 'receptor_chain': 'qC', 'ligand_chain': 'qm'},
    # {'pdb_id': '8gzu', 'receptor_chain': '0H', 'ligand_chain': '0J'},
    # {'pdb_id': '8h2i', 'receptor_chain': 'cX', 'ligand_chain': 'dc'},
    # {'pdb_id': '8hju', 'receptor_chain': 'W', 'ligand_chain': 'X'},
    # {'pdb_id': '8i7o', 'receptor_chain': 'Fb', 'ligand_chain': 'Fc'},
    # {'pdb_id': '8i9p', 'receptor_chain': 'CI', 'ligand_chain': 'LX'},
    # {'pdb_id': '8ir1', 'receptor_chain': 'u', 'ligand_chain': 'r'},
    # {'pdb_id': '8ir3', 'receptor_chain': 'u', 'ligand_chain': 'r'},
    # {'pdb_id': '8iug', 'receptor_chain': 'h', 'ligand_chain': 'Z'},
    # {'pdb_id': '8iun', 'receptor_chain': 'h', 'ligand_chain': 'Z'},
    # {'pdb_id': '8iyj', 'receptor_chain': 'B5', 'ligand_chain': 'Q5'},
    # {'pdb_id': '8j07', 'receptor_chain': '8P', 'ligand_chain': '8Q'},
    # {'pdb_id': '8j5k', 'receptor_chain': 'a', 'ligand_chain': 't'},
    # {'pdb_id': '8j5p', 'receptor_chain': 'V', 'ligand_chain': 'X'},
    # {'pdb_id': '8oin', 'receptor_chain': 'B1', 'ligand_chain': 'B2'},
    # {'pdb_id': '8oip', 'receptor_chain': 'AB', 'ligand_chain': 'Bd'},
    # {'pdb_id': '8oiq', 'receptor_chain': 'B1', 'ligand_chain': 'B2'},
    # {'pdb_id': '8ois', 'receptor_chain': 'AB', 'ligand_chain': 'Bd'},
    # {'pdb_id': '8oj0', 'receptor_chain': '1', 'ligand_chain': '3'},
    # {'pdb_id': '8oj8', 'receptor_chain': '1', 'ligand_chain': '3'},
    # {'pdb_id': '8otz', 'receptor_chain': 'E2', 'ligand_chain': 'B3'},
    # {'pdb_id': '8p10', 'receptor_chain': 'L', 'ligand_chain': 'c'},
    # {'pdb_id': '8p2k', 'receptor_chain': 'BC', 'ligand_chain': 'BK'},
    # {'pdb_id': '8p6j', 'receptor_chain': 'BBB', 'ligand_chain': 'AAA'},
    # {'pdb_id': '8phs', 'receptor_chain': 'AC', 'ligand_chain': 'AI'},
    # {'pdb_id': '8pkh', 'receptor_chain': 'BC', 'ligand_chain': 'HM'},
    # {'pdb_id': '8ppl', 'receptor_chain': 'AC', 'ligand_chain': 'Aj'},
    # {'pdb_id': '8q2m', 'receptor_chain': 'AA', 'ligand_chain': 'A'},
    # {'pdb_id': '8qpc', 'receptor_chain': 'AA', 'ligand_chain': 'C'},
    # {'pdb_id': '8qsj', 'receptor_chain': 'I', 'ligand_chain': 'l'},
    # {'pdb_id': '8rbx', 'receptor_chain': 'b', 'ligand_chain': '1'},
    # {'pdb_id': '8scb', 'receptor_chain': 'P', 'ligand_chain': '1'},
    # {'pdb_id': '8snb', 'receptor_chain': '1y', 'ligand_chain': '1r'},
    # {'pdb_id': '8t4s', 'receptor_chain': 'C', 'ligand_chain': 'n'}
]

def calc_angle(p1,p2,p3):
    # if (p1==None or p2==None or p3==None):
    #     return None
    v1 = p2 - p1
    v2 = p2 - p3
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    if mag_v1 == 0 or mag_v2 == 0:
        raise ValueError("One of the vectors has zero magnitude, leading to an undefined angle.")
    cos_theta = dot_product / (mag_v1 * mag_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # avoid float precision problem
    theta = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta)
    return theta_degrees

def calc_dihedral(p1, p2, p3, p4):
    # if (p1==None or p2==None or p3==None or p4==None):
    #     return None
    # Convert points to numpy arrays
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    # Vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    # Normal vectors
    n1 = np.cross(v1, v2)  # Normal to the plane formed by p1, p2, p3
    n2 = np.cross(v2, v3)  # Normal to the plane formed by p2, p3, p4
    # Normalize normal vectors
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    # Calculate the cosine of the angle between n1 and n2
    cos_theta = np.dot(n1, n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # avoid float precision problem
    # To get the correct sign, use the scalar triple product
    sign = np.sign(np.dot(np.cross(n1, n2), v2))
    # Compute the angle in radians
    theta = np.arccos(cos_theta) * sign
    # Convert to degrees
    theta_degrees = np.degrees(theta)
    return theta_degrees

def calculate_oxygen_angle(c, o):
    """
    Calculate the azimuthal angle (theta) and polar angle (phi) 
    given points c and o in 3D space.
    
    Parameters:
    c (np.ndarray): Coordinates of point c (shape: (3,)).
    o (np.ndarray): Coordinates of point o (shape: (3,)).
    
    Returns:
    tuple: (theta, phi) angles in radians.
    """
    # Calculate the vector from c to o
    vector_co = o - c
    # Calculate the length of co
    r = np.linalg.norm(vector_co)
    # Calculate theta (azimuthal angle)
    theta1 = np.arctan2(vector_co[1], vector_co[0])
    theta1 = np.degrees(theta1)
    # Calculate phi (polar angle)
    theta2 = np.arccos(vector_co[2] / r) if r != 0 else 0  # Handle case where r = 0
    theta2 = np.degrees(theta2)
    
    return float(theta1), float(theta2)

def extract_angle_dihedrals(residues):
    # (-1CA, -1C, N, CA)  # omega
    # (-1C, N, CA, C)   # phi
    # (N, CA, C, 1N)   # psi  
    # (N, CA, C)       # tau
    # (CA, C, 1N)  
    # (C, 1N, 1CA)
    angle_dihedrals = []
    for i in range(1, len(residues) - 1):
        prev_res = residues[i - 1]
        res = residues[i]
        next_res = residues[i + 1]
        prev_C = prev_res['C'].get_coord()
        prev_CA = prev_res['CA'].get_coord()
        res_N = res['N'].get_coord()
        res_CA = res['CA'].get_coord()
        res_C = res['C'].get_coord()
        res_O = res['O'].get_coord()
        next_N = next_res['N'].get_coord()
        # Something is wrong here, or I'm just stupid
        # next_CA = next_res['CA'].get_coord()
        # theta_o1, theta_o2 = calculate_oxygen_angle(res_C, res_O)
        # angle_dihedrals.append({
        #     "omega":calc_dihedral(prev_CA,prev_C,res_N,res_CA),
        #     "phi":calc_dihedral(prev_C,res_N,res_CA,res_C),
        #     "psi":calc_dihedral(res_N,res_CA,res_C,next_N),
        #     "theta1":calc_angle(res_N, res_CA, res_C),
        #     "theta2":calc_angle(res_CA, res_C, next_N),
        #     "theta3":calc_angle(res_C, next_N, next_CA),
        #     "theta_o1": theta_o1,
        #     "theta_o2": theta_o2,
        # })
        angle_dihedrals.append({
            "omega":calc_dihedral(prev_CA,prev_C,res_N,res_CA),
            "phi":calc_dihedral(prev_C,res_N,res_CA,res_C),
            "psi":calc_dihedral(res_N,res_CA,res_C,next_N),
            "dihedral_o": calc_dihedral(res_N, res_CA, res_C, res_O),
            "theta1":calc_angle(res_N, res_CA, res_C),
            "theta2":calc_angle(res_CA, res_C, next_N),
            "theta3":calc_angle(prev_C,res_N,res_CA),
            # "theta3":calc_angle(res_C, next_N, next_CA),
            "theta_o": calc_angle(res_CA, res_C, res_O),
        })
    return angle_dihedrals

# Create resid to res map mannually
def create_res_id_map(c:Chain):
    id_map = {}
    for res in c.get_residues():
        res_id = str(res.get_id()[1])
        res_icode = res.get_id()[2]
        full_id = res_id+res_icode
        id_map[full_id.strip()] = res
        if( res_id != full_id and
            res_id not in id_map.keys()):
            id_map[res_id] = res
    return id_map

def extract_dssp_features(structure, file_path):
    dssp = DSSP(structure, file_path)
    # calculate dssp features
    chain_id_map = {}
    dssp_features = {}
    for k in dssp.keys():
        chain_id = k[0]
        residue_id = str(k[1][1])+str(k[1][2]).strip()
        if(chain_id not in chain_id_map.keys()):
            chain_id_map[chain_id] = create_res_id_map(structure[chain_id])
        if(chain_id not in dssp_features.keys()):
            dssp_features[chain_id] = []
        dssp_features[chain_id].append({
            "res":chain_id_map[chain_id][residue_id],
            "alpha_carbon_coord":list(chain_id_map[chain_id][residue_id]["CA"].get_coord().astype(float)),
            "amino_acid":dssp[k][1],
            "secondary_structure":dssp[k][2], 
            "relative_ASA":dssp[k][3],
            "NH_O_1_relidx":dssp[k][6], "NH_O_1_energy":dssp[k][7], 
            "O_NH_1_relidx":dssp[k][8], "O_NH_1_energy":dssp[k][9],
            "NH_O_2_relidx":dssp[k][10], "NH_O_2_energy":dssp[k][11],
            "O_NH_2_relidx":dssp[k][12], "O_NH_2_energy":dssp[k][13],
        })
    return dssp_features

def drop_res_ojb(residue_features:list):
    for res in residue_features:
        del res["res"]
    return residue_features

def parse_by_record(record):
    msg=False
    receptor_chain_id = record.receptor_chain
    ligand_chain_id = record.ligand_chain
    structure_ids = {'pdb_id': record.pdb_id, 'receptor_chain': receptor_chain_id, 'ligand_chain': ligand_chain_id}
    try:
        if structure_ids in KNOWN_FAIL_RECORDS:
            raise Exception("Known Fail Record")
        # Parse file
        if os.path.exists(os.path.join(STRUCTURE_FOLDER, f"{record.pdb_id}.pdb")):
            file_path = os.path.join(STRUCTURE_FOLDER, f"{record.pdb_id}.pdb")
            parser = PDBParser(QUIET=True)
        elif os.path.exists(os.path.join(STRUCTURE_FOLDER, f"{record.pdb_id}.cif")):
            file_path = os.path.join(STRUCTURE_FOLDER, f"{record.pdb_id}.cif")
            parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("87", file_path)[0]
        # Calculate dssp features
        features = extract_dssp_features(structure, file_path)
        # Calculate angle and dihedral features
        for chain_id in [receptor_chain_id, ligand_chain_id]:
            chain = features[chain_id]
            residues = [res["res"] for res in chain]
            angle_dihedrals = extract_angle_dihedrals(residues)
            for idx, angle_dihedral in enumerate(angle_dihedrals):
                features[chain_id][idx+1].update(angle_dihedral)
        # include pocket info.
        pocket_ids = [res_id[1:] for res_id in record.binding_site_pdb.split()]
        pocket_idx = [
            i for i, r in enumerate(features[receptor_chain_id]) 
            if str(r["res"].get_id()[1]) in pocket_ids
        ]
        pocket_idx = []
        for id in pocket_ids:
            idx = -1
            for i, r in enumerate(features[receptor_chain_id]):
                full_id = (str(r["res"].get_id()[1])+r["res"].get_id()[2]).strip()
                if(id == full_id):
                    idx = i
            if(idx==-1):
                for i, r in enumerate(features[receptor_chain_id]):
                    res_id = str(r["res"].get_id()[1]).strip()
                    if(id == res_id):
                        idx = i
            if(idx!=-1):
                pocket_idx.append(idx)
            else:
                msg = f"{id} not found."
            
        return [structure_ids, {
            "receptor":drop_res_ojb(features[receptor_chain_id]),
            "ligand":drop_res_ojb(features[ligand_chain_id]),
            "pocket_idx":pocket_idx,
            "msg":msg
        }]
    except Exception as e:
        return [structure_ids,{"msg": str(e)}]

def parse_by_record_mp_wrapper(row):
    print(row.pdb_id)
    parse_res = parse_by_record(row)
    return parse_res

def create_data(complex_feature):
    # [1:-1]: drop first and last residue
    receptor = complex_feature[1]["receptor"][1:-1]
    ligand = complex_feature[1]["ligand"][1:-1]

    pos = [
        r["alpha_carbon_coord"] for r in receptor
    ] + [
        r["alpha_carbon_coord"] for r in ligand
    ]

    amino_acid = [
        r["amino_acid"] for r in receptor
    ] + [
        r["amino_acid"] for r in ligand
    ]

    secondary_structure = [
        r["secondary_structure"] for r in receptor
    ] + [
        r["secondary_structure"] for r in ligand
    ]
    secondary_structure = ['-' if char == 'P' else char for char in secondary_structure]
    
    numerical_features = [
        list(r.values())[3:-8:2] for r in receptor
    ] + [
        list(r.values())[3:-8:2] for r in ligand
    ]
    
    angle_features = [
        list(r.values())[-8:] for r in receptor
    ] + [
        list(r.values())[-8:] for r in ligand
    ]
    
    ligand_idx = list(range(len(receptor), len(receptor)+len(ligand)))
    pocket_idx = complex_feature[1]["pocket_idx"]
    edge_idx = [list(i) for i in itertools.product(ligand_idx, pocket_idx)]
    pocket_mask = torch.zeros(len(receptor)+len(ligand), dtype=torch.bool)
    pocket_mask[pocket_idx] = True
    
    graph = {
        "structure_ids":complex_feature[0],
        "coors":torch.tensor(pos),
        "amino_acid":amino_acid,
        "secondary_structure":secondary_structure,
        "numerical_features":torch.tensor(numerical_features),
        "angle_features":torch.deg2rad(torch.tensor(angle_features)),
        "edge_index":torch.tensor(edge_idx).T.contiguous(),
        "ligand_mask":torch.Tensor([False]*len(receptor)+[True]*len(ligand)).bool(),
        "ligand_idx":torch.tensor(ligand_idx, dtype=torch.int),
        "pocket_mask":pocket_mask,
        "pocket_idx":torch.tensor(pocket_idx, dtype=torch.int)
    }
    return graph

def res_to_dataset(ori_data):
    # remove error data
    data = [r for r in ori_data if not r[1]["msg"]]
    x_idxes = []
    for i, complex_feature in enumerate(data):
        receptor_seq = [res["amino_acid"] for res in complex_feature[1]["receptor"]]
        ligand_seq = [res["amino_acid"] for res in complex_feature[1]["ligand"]]
        if("X" in receptor_seq or "X" in ligand_seq):
            x_idxes.append(i)
    data = [r for i,r in enumerate(data) if i not in x_idxes]
    data = [r for r in data if len(r[1]["ligand"])>=5]
    result = [create_data(r) for r in data]
    return result

if __name__ == "__main__":
    complexes = pd.read_csv(BIOLIP_META_FILE, sep="\t", names=BIOLIP_META_HEADER)
    complexes.drop_duplicates(subset="pdb_id", inplace=True)
    complexes.reset_index(drop=True, inplace=True)
    complexes = complexes.loc[complexes.resolution<5]
    rows = [complexes.iloc[i] for i in range(len(complexes))]
    
    # Get all information from complexfile
    with mp.Pool(THREAD_NUM) as pool:
        result = pool.map(parse_by_record_mp_wrapper, rows)
    # Format the result
    result = res_to_dataset(result)
    # Save the result
    torch.save(result, OUTPUT_FILE)