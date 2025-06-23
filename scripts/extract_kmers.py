import argparse
import sys
from pathlib import Path
from kmerml.kmers.generate import KmerExtractor
from kmerml.utils.path_utils import find_files

def main():
    parser = argparse.ArgumentParser(description="Extract k-mers from genome files")
    
    # Input options
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory or file(s) with genome sequences")
    parser.add_argument("--pattern", "-p", default="*.fa,*.fasta",
                       help="Comma-separated patterns to match genome files")
    
    # Output options
    parser.add_argument("--output-dir", "-o", default="data/processed/kmers",
                       help="Output directory for k-mer files")
    parser.add_argument("--compress", "-c", action="store_true",
                       help="Compress output files")
    
    # K-mer options
    parser.add_argument("--k-values", "-k", default="8,9,10,11,12",
                       help="Comma-separated list of k values to extract")
    
    # Processing options
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Search input directory recursively")
    
    args = parser.parse_args()
    
    # Parse k values
    try:
        k_values = [int(k) for k in args.k_values.split(",")]
    except ValueError:
        print("Error: k values must be integers")
        return 1
        
    # Find input files
    patterns = args.pattern.split(",")
    input_path = Path(args.input)
    
    if input_path.is_file():
        genome_files = [input_path]
    else:
        genome_files = find_files(args.input, patterns=patterns, recursive=args.recursive)
    
    if not genome_files:
        print(f"No genome files found matching patterns: {args.pattern}")
        return 1
        
    print(f"Found {len(genome_files)} genome files")
    
    # Initialize extractor
    extractor = KmerExtractor(output_dir=args.output_dir, compress=args.compress)
    
    # Process files
    for i, genome in enumerate(genome_files):
        organism_id = genome.stem
        print(f"Processing {organism_id} ({i+1}/{len(genome_files)})")
        extractor.extract_kmers_from_fasta(genome, k_values, organism_id=organism_id)
    
    print("K-mer extraction completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())