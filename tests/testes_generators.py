import os
from kmerml.kmers.generators import kmer_stats, process_genome

def main():
    fasta_dir = "data/raw/"
    out_dir = "data/processed/kmers/"
    os.makedirs(out_dir, exist_ok=True)
    ks = [8, 10, 12]  # Exemplo, substitua pelos valores desejados
    fasta_files = [os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if f.endswith(".fasta") or f.endswith(".fa")]
    for fasta in fasta_files:
        for k in ks:
            print(f"Processing {fasta} with k={k}")
            process_genome(fasta, k, out_dir)

if __name__ == "__main__":
    main()