import os
from Bio import SeqIO
from kmerml.utils.progress import progress_bar
from kmerml.utils.logger import setup_logging

logger = setup_logging('kmer_generator')

def kmer_stats(kmer):
    gc = (kmer.count('G') + kmer.count('C')) / len(kmer) * 100
    cpg = sum(1 for i in range(len(kmer)-1) if kmer[i:i+2] == 'CG')
    # Bit array: [A, T, C, G] presença/ausência
    presence = [int(base in kmer) for base in 'ATCG']
    # Contagem: [A, T, C, G]
    counts = [kmer.count(base) for base in 'ATCG']
    return gc, cpg, presence, counts

def process_genome(fasta_path, k, out_dir):
    org_name = os.path.splitext(os.path.basename(fasta_path))[0]
    out_path = os.path.join(out_dir, f"{org_name}/kmers_k{k}.tsv")
    # Check if output directory exists, if not create it
    out_dir_path = os.path.dirname(out_path)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path, exist_ok=True)
    with open(out_path, "w") as out:
        out.write("kmer\tgc_percent\tcpg_count\tA_present\tT_present\tC_present\tG_present\tA_count\tT_count\tC_count\tG_count\n")
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if "N" in kmer:
                    continue
                gc, cpg, presence, counts = kmer_stats(kmer)
                out.write(
                    f"{kmer}\t{gc:.2f}\t{cpg}\t"
                    f"{presence[0]}\t{presence[1]}\t{presence[2]}\t{presence[3]}\t"
                    f"{counts[0]}\t{counts[1]}\t{counts[2]}\t{counts[3]}\n"
                )