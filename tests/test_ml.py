from kmerml.ml.features import KmerFeatureBuilder

# Initialize builder
builder = KmerFeatureBuilder("data/features/")

# Build matrices with different metrics
count_matrix = builder.build_from_statistics_files(metric="count")
gc_matrix = builder.build_from_statistics_files(metric="gc_percent")
entropy_matrix = builder.build_from_statistics_files(metric="shannon_entropy")

# Show results
print("Count matrix:")
print(count_matrix)
print("\nGC content matrix:")
print(gc_matrix)
print("\nEntropy matrix:")
print(entropy_matrix)

# Access organism and k-mer lists
print(f"\nOrganisms: {builder.organisms}")
print(f"K-mers: {builder.kmers}")