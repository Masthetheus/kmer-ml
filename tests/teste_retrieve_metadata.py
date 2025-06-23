from kmerml.utils.genome_metadata import GenomeMetadataManager
from kmerml.utils.path_utils import find_files
from kmerml.utils.kmer_metadata import KmerMetadataManager

genome_files = find_files("data/raw/", patterns=["*.fa", "*.fasta", "*.fna"], recursive=True)
metadata_manager = GenomeMetadataManager()
metadata_manager.collect_metadata("data/raw/")
metadata_manager = KmerMetadataManager()
kmer_files = find_files("data/processed/kmers", patterns=["*.txt"], recursive=True)
metadata_manager.add_kmer_metadata(kmer_files)

