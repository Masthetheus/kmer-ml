import argparse
import sys
from pathlib import Path
from kmerml.kmers.statistics import KmerFeatureExtractor
from kmerml.utils.path_utils import find_files

def main():
    parser = argparse.ArgumentParser(description="Generate ML features from k-mer files")
    
    # Input options
    parser.add_argument("--input", "-i", default = 'data/processed/kmers/', 
                       help="Input directory with k-mer files")
    
    # Output options
    parser.add_argument("--output-dir", "-o", default="data/processed/features",
                       help="Output directory for feature files")
    
    # Feature options
    parser.add_argument("--feature-set", "-f", default="all",
                       help="Feature set to generate (all, basic, advanced)")
    
    # Metadata option
    parser.add_argument("--metadata", "-m", default="data/metadata/genome_metadata.json",
                       help="Path to genome metadata file")
                       
    # K-mer options
    parser.add_argument("--k-values", "-k", default="all",
                       help="Comma-separated list of k values to process (or 'all')")
    
    args = parser.parse_args()
    
    # Find k-mer files
    if args.k_values.lower() == "all":
        patterns = ["k*.txt"]
    else:
        try:
            k_list = [int(k) for k in args.k_values.split(",")]
            patterns = [f"k{k}.txt" for k in k_list]
        except ValueError:
            print("Error: k values must be integers")
            return 1
    
    kmer_files = find_files(args.input, patterns=patterns, recursive=True)
    
    if not kmer_files:
        print(f"No k-mer files found in {args.input} matching patterns: {patterns}")
        return 1
    
    print(f"Found {len(kmer_files)} k-mer files")
    
    # Determine feature set
    if args.feature_set.lower() == "all":
        feature_set = None  # Use default (all features)
    elif args.feature_set.lower() == "basic":
        feature_set = ["gc_content", "base_counts"]
    elif args.feature_set.lower() == "advanced":
        feature_set = ["gc_content", "base_counts", "entropy", "cpg_sites", "repeats"]
    else:
        print(f"Unknown feature set: {args.feature_set}")
        return 1
    
    # Initialize feature extractor
    feature_extractor = KmerFeatureExtractor(
        input_paths=kmer_files,
        output_dir=args.output_dir,
        metadata_file=args.metadata
    )
    
    # Generate features
    output_files = feature_extractor.extract_features(required_features=feature_set)
    
    print(f"Generated {len(output_files)} feature files")
    return 0

if __name__ == "__main__":
    sys.exit(main())