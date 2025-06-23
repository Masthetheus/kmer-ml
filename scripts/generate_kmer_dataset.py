import os
import gzip
from collections import defaultdict
from pathlib import Path

def main():
    extractor = KmerExtractor(compress=True)
    k_values = [3, 4, 5, 6]  # Multiple k values
    
    # Your sequences would come from FASTA files
    sequences = [("organism_1", "ATCGATCG"), ("organism_2", "GCTAGCTA")]
    
    extractor.batch_extract(sequences, k_values)

if __name__ == "__main__":
    main()