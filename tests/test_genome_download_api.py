import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kmerml.download.ncbidownload import *
from Bio import Entrez

Entrez.email = "matheuspedroncassol@gmail.com"
Entrez.tool = "ufrgslbcmleveduras"

filepath = "data/ncbi_dataset.tsv"
accessions = read_accession_list(filepath)
print(f"Found {len(accessions)} accessions in {filepath}")
download_genome_bioentrez(accessions, "data/raw/")
