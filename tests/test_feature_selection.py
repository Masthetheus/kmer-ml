from kmerml.ml.features import KmerFeatureBuilder
from kmerml.ml.feature_selection import PhylogeneticFeatureSelector, select_features_by_variance
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

# 1. Load the feature matrix (with filtering for k≥8)
print("Loading feature matrix...")
builder = KmerFeatureBuilder(stats_dir="data/processed/features/")
feature_matrix = builder.build_from_statistics_files(
    metric="shannon_entropy"
)
print(f"Feature matrix shape: {feature_matrix.shape}")

# Limit to a smaller number of features for testing
if feature_matrix.shape[1] > 10000:
    print(f"Limiting to 10,000 features for faster testing")
    # Select features by variance
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    selector.fit(feature_matrix)
    # Get top 10,000 by variance
    variances = selector.variances_
    top_indices = np.argsort(variances)[-10000:]
    feature_matrix = feature_matrix.iloc[:, top_indices]
    print(f"Reduced feature matrix shape: {feature_matrix.shape}")

# 2. Generate reference distances (using a proxy since we don't have real phylogenetic distances)
print("\nGenerating reference distances...")
# Option 1: Use Euclidean distance on the feature matrix as a proxy
dist_matrix = squareform(pdist(feature_matrix.values, metric='euclidean'))
print(f"Distance matrix shape: {dist_matrix.shape}")

# 3. Test variance-based selection (simple method first)
print("\nTesting variance-based selection...")
start_time = time.time()
variance_features = select_features_by_variance(feature_matrix, threshold=0.01, n_features=100)
variance_time = time.time() - start_time
print(f"Selected {len(variance_features)} features by variance in {variance_time:.2f} seconds")
print(f"Sample features: {variance_features[:5]}")

# 4. Test phylogenetic feature selection methods
n_features = 100  # Number of features to select

# Random Forest method
print("\nTesting Random Forest feature selection...")
start_time = time.time()
rf_selector = PhylogeneticFeatureSelector(n_features=n_features, method='random_forest')
rf_selector.fit(feature_matrix, dist_matrix)
rf_features = rf_selector.selected_features
rf_time = time.time() - start_time
print(f"Selected {len(rf_features)} features using Random Forest in {rf_time:.2f} seconds")
print(f"Sample features: {rf_features[:5]}")

# Mutual Information method
print("\nTesting Mutual Information feature selection...")
start_time = time.time()
mi_selector = PhylogeneticFeatureSelector(n_features=n_features, method='mutual_info')
mi_selector.fit(feature_matrix, dist_matrix)
mi_features = mi_selector.selected_features
mi_time = time.time() - start_time
print(f"Selected {len(mi_features)} features using Mutual Information in {mi_time:.2f} seconds")
print(f"Sample features: {mi_features[:5]}")

# Correlation method
print("\nTesting Correlation feature selection...")
start_time = time.time()
corr_selector = PhylogeneticFeatureSelector(n_features=n_features, method='correlation')
corr_selector.fit(feature_matrix, dist_matrix)
corr_features = corr_selector.selected_features
corr_time = time.time() - start_time
print(f"Selected {len(corr_features)} features using Correlation in {corr_time:.2f} seconds")
print(f"Sample features: {corr_features[:5]}")

# 5. Compare feature overlap between methods
print("\nComparing feature overlap between methods...")
rf_set = set(rf_features)
mi_set = set(mi_features)
corr_set = set(corr_features)
var_set = set(variance_features)

print(f"RF ∩ MI: {len(rf_set & mi_set)} features")
print(f"RF ∩ Correlation: {len(rf_set & corr_set)} features")
print(f"MI ∩ Correlation: {len(mi_set & corr_set)} features")
print(f"All methods: {len(rf_set & mi_set & corr_set)} features")
print(f"Variance ∩ (RF ∪ MI ∪ Corr): {len(var_set & (rf_set | mi_set | corr_set))} features")

# 6. Analyze k-mer lengths in selected features
print("\nAnalyzing k-mer lengths in selected features...")
def analyze_kmer_lengths(features, name):
    lengths = [len(f) for f in features]
    length_counts = pd.Series(lengths).value_counts().sort_index()
    print(f"\n{name} selection k-mer lengths:")
    print(length_counts)
    return length_counts

rf_lengths = analyze_kmer_lengths(rf_features, "Random Forest")
mi_lengths = analyze_kmer_lengths(mi_features, "Mutual Information")
corr_lengths = analyze_kmer_lengths(corr_features, "Correlation")
var_lengths = analyze_kmer_lengths(variance_features, "Variance")

# 7. Save selected features to file
print("\nSaving selected features to files...")
# Create a DataFrame with feature importances from each method
all_features = set(rf_features) | set(mi_features) | set(corr_features) | set(variance_features)
feature_df = pd.DataFrame(index=sorted(all_features))

# Add importances from each method
for selector, name in zip(
    [rf_selector, mi_selector, corr_selector], 
    ["random_forest", "mutual_info", "correlation"]
):
    feature_df[f"{name}_importance"] = pd.Series(selector.feature_importances)

# Add length information
feature_df["length"] = feature_df.index.map(len)

# Save to CSV
output_file = "data/processed/selected_features.csv"
feature_df.to_csv(output_file)
print(f"Saved feature information to {output_file}")

print("\nFeature selection testing complete!")