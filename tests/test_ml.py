from kmerml.ml.features import KmerFeatureBuilder

# Build feature matrix from k-mer files
feature_builder = KmerFeatureBuilder("data/processed/features")
feature_matrix = feature_builder.build_from_statistics_files(k_value=8)

# Normalize and filter features
normalized = feature_builder.normalize(method="frequency")
filtered = feature_builder.filter_features(min_prevalence=0.2, min_variance=0.001)

# Get top features for clustering
top_features = feature_builder.get_top_features(n_features=500, method="variance")

print(f"Original features: {feature_matrix.shape[1]}")
print(f"After filtering: {filtered.shape[1]}")
print(f"Top features: {top_features.shape[1]}")