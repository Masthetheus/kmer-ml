from kmerml.utils.genome_metadata import GenomeMetadataManager
from kmerml.utils.path_utils import find_files
from kmerml.utils.kmer_metadata import KmerMetadataManager

def main():
    print("Starting metadata generation...")
    
    # Collect genome metadata

    metadata_manager = GenomeMetadataManager()
    metadata_manager.collect_metadata("data/raw/")
    
    # Collect k-mer metadata
    kmer_metadata_manager = KmerMetadataManager()
    kmer_files = find_files("data/processed/kmers", patterns=["*.txt"], recursive=True)
    kmer_metadata_manager.add_kmer_metadata(kmer_files)
