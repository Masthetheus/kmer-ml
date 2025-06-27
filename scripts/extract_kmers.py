import argparse
import sys
from pathlib import Path
import os
from kmerml.kmers.generate import KmerExtractor
from kmerml.utils.path_utils import find_files
from kmerml.utils.path_utils import get_accession_codes_from_tsv

def main():
    parser = argparse.ArgumentParser(description="Extract k-mers from genome files")
    
    # Input options
    parser.add_argument("--input", "-i", default="data/raw/", 
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
    parser.add_argument("--processes", "-j", type=int, default=None,
                       help="Number of parallel processes to use (default: number of CPU cores)")
    
    parser.add_argument("--accession-tsv", type=str, default=None,
                    help="TSV file with accession codes to filter genomes")

    
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

    if args.accession_tsv:
        accessions = set(get_accession_codes_from_tsv(args.accession_tsv))
        # Accept both with and without version
        def file_matches_accession(genome):
            stem = genome.stem
            # Try exact match
            if stem in accessions:
                return True
            # Try without version (e.g., GCF_000005845.2 -> GCF_000005845)
            if "." in stem and stem.split(".")[0] in accessions:
                return True
            return False
        genome_files = [g for g in genome_files if file_matches_accession(g)]

    if not genome_files:
        print(f"No genome files found matching patterns: {args.pattern}")
        return 1
        
    print(f"Found {len(genome_files)} genome files")
    
    # Initialize extractor
    extractor = KmerExtractor(output_dir=args.output_dir, compress=args.compress)
    
    # Determine number of processes
    n_processes = args.processes
    if n_processes is None:
        n_processes = os.cpu_count()
        print(f"Using {n_processes} CPU cores for parallel processing")
    else:
        print(f"Using {n_processes} processes as specified")
    
    # Extract organism IDs from filenames
    organism_ids = [genome.stem for genome in genome_files]
    
    # Process files in parallel
    if n_processes > 1 and len(genome_files) > 1:
        print(f"Processing {len(genome_files)} genomes in parallel")
        processed_ids = extractor.extract_from_genome_list(
            genome_files, 
            k_values, 
            organism_ids=organism_ids,
            n_processes=n_processes
        )
        print(f"Successfully processed {len(processed_ids)} out of {len(genome_files)} genomes")
    else:
        # Fall back to sequential processing for single file or if parallel disabled
        print("Processing genomes sequentially")
        for i, genome in enumerate(genome_files):
            organism_id = organism_ids[i]
            print(f"Processing {organism_id} ({i+1}/{len(genome_files)})")
            try:
                extractor.extract_kmers_from_fasta(genome, k_values, organism_id=organism_id)
                print(f"Completed {organism_id}")
            except Exception as e:
                print(f"Error processing {organism_id}: {str(e)}")
    
    print("K-mer extraction completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())