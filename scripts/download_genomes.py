from kmerml.utils.validation import get_valid_email, get_valid_tool
from kmerml.download.ncbidownload import read_accession_list, download_genome_bioentrez
from Bio import Entrez
import argparse


def main():
    Entrez.email = get_valid_email()
    Entrez.tool = get_valid_tool()
    parser = argparse.ArgumentParser(description="Download genomes from NCBI by assembly accession list.")
    parser.add_argument(
        "--accession-list",
        type=str,
        default="data/ncbi_dataset.tsv",
        help="Path to the file containing assembly accession numbers (default: data/ncbi_dataset.tsv)"
    )
    args = parser.parse_args()

    accession_file = args.accession_list
    print(f"Using accession file: {accession_file}")
    print("Do you want to specify a custom .csv/.tsv column name for the accession numbers? (y/[n])")
    custom_column = input().strip().lower() == 'y'
    if custom_column:
        column_name = input("Enter the column name (NCBI default is 'Assembly Accession'): ").strip() or "Assembly Accession"
    else:
        column_name = None
    acessions = read_accession_list(accession_file, column=column_name)
    print(f"Found {len(acessions)} accessions in {accession_file}")
    print("Any not found accessions can be seen in the log file (data/logs/entrez.log).")
    download_genome_bioentrez(acessions, "data/raw/")
    print("All genomes downloaded and decompressed successfully.")
    print("Downloaded genomes can be found in the 'data/raw/' directory.")

if __name__ == "__main__":
    main()