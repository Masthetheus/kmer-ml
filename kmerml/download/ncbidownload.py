import os
import csv
import requests
from Bio import Entrez
import gzip
import time
from kmerml.utils.progress import progress_bar
from kmerml.utils.logger import setup_logging

logger = setup_logging()

def read_accession_txt(filepath):
    """
    Reads a text file containing a list of NCBI accession numbers.
    Each line should contain a valid accession number, starting with "GCF_" or "GCA_".
    Ignores empty lines and comments (lines starting with "#").

    Args:
        filepath (str): Path to the text file.

    Returns:
        list: List of valid accession numbers found in the file.
    """
    accessions = set()
    with open(filepath) as f:
        for line in f:
            acc = line.strip() # Remove espaços em branco
            if not acc or acc.startswith("#"): # Ignora linhas vazias ou comentários
                continue
            if not acc.startswith(("GCF_", "GCA_")): # Checa integridade do accession
                print(f"Aviso: accession suspeito '{acc}' ignorado.")
                continue
            accessions.add(acc)
    return list(accessions)

def read_accession_list(filepath, column=None):
    """
    Reads a list of NCBI accession numbers from a file.
    Supports .txt, .csv, and .tsv files.
    If the column is not specified, tries to automatically detect the column
    containing the accession numbers.

    Args:
        filepath (str): Path to the file containing accession numbers.
        column (int or str, optional): Index or name of the column containing the accession numbers.
                                       If None, tries to detect automatically.

    Returns:
        list: List of accession numbers found in the file.
    """
    ext = os.path.splitext(filepath)[1].lower()
    accessions = []
    if ext == ".txt":
        accessions = read_accession_txt(filepath)
    elif ext in [".csv", ".tsv"]:
        delimiter = ',' if ext == ".csv" else '\t'
        with open(filepath, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            # Detect column index
            if column is None:
                for i, col in enumerate(header):
                    if "assembly" in col.lower() and ("accession" in col.lower()):
                        column = i
                        break
                else:
                    column = 0  # fallback: primeira coluna
            elif isinstance(column, str):
                column = header.index(column)
            for row in reader:
                if row and len(row) > column:
                    accessions.append(row[column].strip())
    else:
        raise ValueError("Unsupported file format. Use .txt, .csv, or .tsv files.")
    return accessions

def get_genome_entrez_links(accessions):
    """
    Returns a list of genome Entrez links for one or more assembly accessions.
    Args:
        accessions (str or list): Single accession or list of accessions.
    Returns:
        list: List of Entrez links.
    """
    log = "data/logs/entrez.log"
    if isinstance(accessions, str):
        accessions = [accessions]
    links = {}
    try:
        for acc in accessions:
            # Search the ID on the DB
            print(f"Searching ID for Assembly Accession: {acc}")
            handle = Entrez.esearch(db="assembly", term=acc, retmode="xml")
            record = Entrez.read(handle)
            handle.close()

            if not record["IdList"]:
                print(f"No result found for {acc}")
                with open(log, 'a') as log:
                    log.write(f"ID for Assembly Accession {acc} not found.\n")
                return None

            # Get's the first ID
            assembly_id = record["IdList"][0]
            print(f"Assembly ID found: {assembly_id}")

            # Get assembly summary
            print(f"Searching summary for Assembly ID: {assembly_id}")
            handle = Entrez.esummary(db="assembly", id=assembly_id, retmode="xml")
            summary = Entrez.read(handle)
            handle.close()

            # Obtain FTP Assembly link
            ftp_path = summary['DocumentSummarySet']['DocumentSummary'][0].get('FtpPath_RefSeq')

            if not ftp_path:
                ftp_path = summary['DocumentSummarySet']['DocumentSummary'][0].get('FtpPath_GenBank')
            if ftp_path:
                link = ftp_path + "/" + ftp_path.split("/")[-1] + "_genomic.fna.gz"
                link = link[3:]
                link = 'https' + link
                links[acc] = link
            else:
                print(f"Link FTP not found for {acc}")
                with open(log, 'a') as log:
                    log.write(f"Link FTP not found for {acc}.\n")
                return None
        return links
    except Exception as e:
        print(f"Error during {acc}: {e}")
        return None

def convert_fasta_to_uppercase(file_path, file_uncompressed):
    """
    Converts a gzipped FASTA file to uppercase and writes to output_path.
    Decompress all .gz files in the specified directory and remove the original compressed files.
    Args:
        file_path (str): Path to the gzipped FASTA file.
        file_uncompressed (str): Path to the output uncompressed FASTA file.
    Returns:
        list: List of decompressed file paths.
    """

    # Detect if gzipped
    open_func = gzip.open if file_path.endswith('.gz') else open
    with open_func(file_path, 'rt') as fin, open(file_uncompressed, 'w') as fout:
        for line in fin:
            if line.startswith('>'):
                fout.write(line)
            else:
                fout.write(line.upper())
    os.remove(file_path)  # Remove the original gzipped file

def download_genome_bioentrez(accessions, out):
    """
    Downloads a genome file from NCBI using Bio.Entrez.
    Args:
        assembly_accession (str): Assembly accession number.
        out (str): Path to the output directory where the genome will be saved.
        organism (str): Name of the organism, used for naming the file.
    Returns:
        None
    """
    # Obtain genome links from Entrez
    links = get_genome_entrez_links(accessions)
    cont = 1 # For the progress bar
    start = time.time() # For the progress bar
    if len(links) != len(accessions):
        if not links:
            print("No valid links found for the provided accessions.")
            return
        else:
            print(f"Some accessions could not be found. Check the log file (data/logs/entrez.log) for details.")

    # Makes sure output directory exists
    os.makedirs(out, exist_ok=True)
    for organism, link in links.items():
        # Sets file name
        file_name = organism.replace('.', '_')
        # Initiate download
        response = requests.get(link, stream=True)
        if response.status_code == 200:
            file_path = out+file_name+'.gz'
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            file_uncompressed = file_path[:-3] + '.fa'  # Remove .gz extension    
            convert_fasta_to_uppercase(file_path, file_uncompressed )
        else:
            print(f"Error during download of {organism}: {response.status_code}")
        
        start = progress_bar(cont, len(links), start_time=start)
        cont+=1
