from kmerml.kmers.generate import KmerExtractor
from kmerml.kmers.statistics import KmerFeatureExtractor
from kmerml.utils.path_utils import find_files
from kmerml.utils.genome_metadata import GenomeMetadataManager
# Initialize extractor
extractor = KmerExtractor(output_dir="data/processed/kmers/", compress=False)
genomes = find_files("data/raw/", patterns=["*.fa", "*.fasta"], recursive=True)
# Define k values to extract
k_values = [8, 9, 10, 11, 12]

# Process in parallel (recommended for multiple organisms)
#extractor.extract_from_genome_list(genomes, k_values)
kmer_files = find_files("data/processed/kmers", patterns=["*.txt"], recursive=True)

feature_extractor = KmerFeatureExtractor(
    input_paths=kmer_files,
    output_dir="data/processed/features",
    metadata_file="data/metadata/genome_metadata.json"
)

output_files = feature_extractor.extract_features()
print(f"Extracted features saved to: {output_files}")