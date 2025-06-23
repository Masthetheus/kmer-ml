from kmerml.in_test_kmer.generate import KmerExtractor
from kmerml.in_test_kmer.statistics import KmerStatistics

# Initialize extractor
extractor = KmerExtractor(output_dir="data/processed/", compress=False)

# Define k values to extract
k_values = [5, 6, 7, 8, 9, 10]

# Process in parallel (recommended for multiple organisms)
extractor.extract_kmers_from_fasta(fasta_file="data/raw/teste.fasta", k_values=k_values, organism_id="teste")

def main():
    parser = argparse.ArgumentParser(description="Generate statistics from k-mer files")
    parser.add_argument("--input_dir", default="data/processed", help="Directory containing k-mer files")
    parser.add_argument("--output", default="kmer_statistics.csv", help="Output CSV file")
    parser.add_argument("--pattern", default="*_k*.txt*", help="File pattern to match")
    args = parser.parse_args()
    
    try:
        stats = KmerStatistics(input_dir=args.input_dir)
        stats.save_stats_to_csv(args.output, args.pattern)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())