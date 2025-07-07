from kmerml.utils.genome_metadata import GenomeMetadataManager
from kmerml.utils.path_utils import find_files
from kmerml.utils.kmer_metadata import KmerMetadataManager
import argparse
import sys
from pathlib import Path

def gather_metadata(option, genome_input_dir=None, kmer_input_dir=None, output_file=None):
    """
    Gather metadata based on user selection.
    
    Args:
        option (str): 'genome', 'kmer', or 'both'
        genome_input_dir (str): Directory containing genome files
        kmer_input_dir (str): Directory containing k-mer files
        output_file (str): Path to output metadata file
    """
    option = option.lower()
    
    if option == "genome":
        if not genome_input_dir:
            print("Error: genome input directory is required for 'genome' option")
            return False
            
        print(f"Gathering genome metadata from {genome_input_dir}...")
        manager = GenomeMetadataManager(output_file)
        manager.collect_metadata(genome_input_dir)
        print(f"Genome metadata saved to {output_file}")
        
    elif option == "kmer":
        if not kmer_input_dir:
            print("Error: k-mer input directory is required for 'kmer' option")
            return False
            
        print(f"Gathering k-mer metadata from {kmer_input_dir}...")
        kmer_files = find_files(kmer_input_dir, patterns=["k*.txt"], recursive=True)
        manager = KmerMetadataManager(output_file)
        manager.add_kmer_metadata(kmer_files)
        print(f"K-mer metadata saved to {output_file}")
        
    elif option == "both":
        if not genome_input_dir or not kmer_input_dir:
            print("Error: both genome and k-mer input directories are required for 'both' option")
            return False
            
        print(f"Gathering genome metadata from {genome_input_dir}...")
        genome_manager = GenomeMetadataManager(output_file)
        genome_manager.collect_metadata(genome_input_dir)
        
        print(f"Gathering k-mer metadata from {kmer_input_dir}...")
        kmer_files = find_files(kmer_input_dir, patterns=["k*.txt"], recursive=True)
        kmer_manager = KmerMetadataManager(output_file)
        kmer_manager.add_kmer_metadata(kmer_files)
        print(f"Combined metadata saved to {output_file}")
        
    else:
        print(f"Error: Unknown option '{option}'")
        print("Valid options are: genome, kmer, both")
        return False
    
    return True

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate metadata for genomes and/or k-mers')
    parser.add_argument('option', choices=['genome', 'kmer', 'both'], 
                        help='Type of metadata to gather')
    parser.add_argument('--genome-input', '-g', default = 'data/raw/',
                        help='Input directory containing genome files')
    parser.add_argument('--kmer-input', '-k', default='data/processed/kmers/',
                        help='Input directory containing k-mer files')
    parser.add_argument('--output', '-o', default='data/metadata/genome_metadata.json', 
                        help='Output metadata file path')
    
    args = parser.parse_args()
    
    # Validate arguments based on option
    if args.option == 'genome' and not args.genome_input:
        parser.error("--genome-input is required when option is 'genome'")
    elif args.option == 'kmer' and not args.kmer_input:
        parser.error("--kmer-input is required when option is 'kmer'")
    elif args.option == 'both' and (not args.genome_input or not args.kmer_input):
        parser.error("Both --genome-input and --kmer-input are required when option is 'both'")
    
    success = gather_metadata(args.option, 
                              args.genome_input, 
                              args.kmer_input,
                              args.output)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()