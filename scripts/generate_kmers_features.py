import argparse
import sys
import os
from pathlib import Path
from kmerml.kmers.statistics import KmerFeatureExtractor
from kmerml.utils.path_utils import find_files
from kmerml.utils.path_utils import get_accession_codes_from_tsv
import concurrent.futures
import time

def process_organism(kmer_files, output_dir, feature_set, metadata_file):
    """Process a single organism's k-mer files"""
    try:
        # Get organism name from the first file
        organism = kmer_files[0].parent.name
        
        # Initialize extractor for just this organism
        extractor = KmerFeatureExtractor(
            input_paths=kmer_files,
            output_dir=output_dir,
            metadata_file=metadata_file
        )
        
        # Extract features
        output_file = extractor.extract_features(required_features=feature_set)
        return (organism, True, output_file)
    except Exception as e:
        return (organism, False, str(e))

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
    
    # Add parallel processing option
    parser.add_argument("--processes", "-j", type=int, default=None,
                       help="Number of parallel processes to use (default: number of CPU cores)")
    
    parser.add_argument("--accession-tsv", type=str, default=None,
                    help="TSV file with accession codes to filter organisms")


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
    print(kmer_files)
    if not kmer_files:
        print(f"No k-mer files found in {args.input} matching patterns: {patterns}")
        return 1
    
    print(f"Found {len(kmer_files)} k-mer files")
    
    # Determine feature set
    if args.feature_set.lower() == "all":
        feature_set = None  # Use default (all features)
    else:
        print(f"Unknown feature set: {args.feature_set}")
        return 1
    
    files_by_organism = {}

    for file_path in kmer_files:
        organism = file_path.parent.name
        if organism not in files_by_organism:
            files_by_organism[organism] = []
        files_by_organism[organism].append(file_path)

    if args.accession_tsv:
        accessions = set(get_accession_codes_from_tsv(args.accession_tsv))
        files_by_organism = {org: files for org, files in files_by_organism.items() if org in accessions}
        
    organisms = list(files_by_organism.keys())
    print(f"Found {len(organisms)} organisms to process")
    
    # Determine number of processes
    n_processes = args.processes
    if n_processes is None:
        n_processes = os.cpu_count()
    n_processes = min(n_processes, len(organisms))
    
    # Process in parallel if multiple organisms and processes
    if n_processes > 1 and len(organisms) > 1:
        print(f"Processing {len(organisms)} organisms using {n_processes} parallel processes")
        
        results = []
        start_time = time.time()
        completed = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit tasks
            future_to_org = {}
            for organism, org_files in files_by_organism.items():
                future = executor.submit(
                    process_organism, 
                    org_files, 
                    args.output_dir, 
                    feature_set, 
                    args.metadata
                )
                future_to_org[future] = organism
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_org):
                organism = future_to_org[future]
                completed += 1
                
                try:
                    result = future.result()
                    success = result[1]
                    
                    # Calculate progress stats
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(organisms) - completed) / rate if rate > 0 else 0
                    
                    if success:
                        print(f"[{completed}/{len(organisms)}] Processed {organism} "
                              f"({rate:.2f} genomes/sec, ~{remaining:.1f}s remaining)")
                        results.append(result)
                    else:
                        print(f"[{completed}/{len(organisms)}] Error processing {organism}: {result[2]}")
                
                except Exception as e:
                    print(f"[{completed}/{len(organisms)}] Error processing {organism}: {str(e)}")
        
        successful = [r for r in results if r[1]]
        print(f"Generated {len(successful)} feature files out of {len(organisms)} organisms")
    
    else:
        # Sequential processing
        print(f"Processing {len(organisms)} organisms sequentially")
        
        # Initialize feature extractor with all files
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