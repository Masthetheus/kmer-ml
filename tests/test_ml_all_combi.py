import itertools
import time
import os
from kmerml.ml.features import KmerFeatureBuilderAgg
from kmerml.ml.dimensionality import reduce_dimensions, plot_reduced
from scipy.cluster.hierarchy import dendrogram, to_tree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import scipy.io
import fastcluster
import seaborn as sns
import pandas as pd

plt.ioff()  # Disable interactive mode to avoid blocking

stats_dir = "data/processed/features"
output_dir = "data/results/auto_combinations_all_org_best_silhouette"
os.makedirs(output_dir, exist_ok=True)

all_metrics = ['relative_freq', 'gc_skew', 'at_skew',
                'shannon_entropy', 'is_palindrome', 'unique_kmer_ratio',
                'palindrome_ratio', 'noncanonical_ratio']

all_agg_funcs = ['mean']

# Generate all non-empty combinations of metrics and agg_funcs
# ...existing code...

# Generate only combos with at_skew + others OR relative_freq + others
metric_combos = []
for r in range(2, len(all_metrics)+1):  # combos of size >=2
    for combo in itertools.combinations(all_metrics, r):
        if ("at_skew" in combo and "relative_freq" not in combo) or ("relative_freq" in combo and "at_skew" not in combo):
            metric_combos.append(combo)

agg_func_combos = []
for r in range(1, len(all_agg_funcs)+1):
    agg_func_combos.extend(itertools.combinations(all_agg_funcs, r))

total_combinations = len(metric_combos) * len(agg_func_combos)
print(f"Total combinations: {total_combinations}")

# ...rest of the code remains unchanged...

# Estimate running time (very rough, based on a single run)
start_time = time.time()
# Run a single quick test to estimate

builder = KmerFeatureBuilderAgg(stats_dir=stats_dir)
feature_matrix = builder.build_aggregated_features(
    metrics=list(all_metrics[:2]), agg_funcs=list(all_agg_funcs[:1]), n_jobs=4
)
feature_matrix_std = builder.standardize_features()
X_pca = reduce_dimensions(feature_matrix_std, method="pca", n_components=min(2, feature_matrix_std.shape[1]))
elapsed = time.time() - start_time
estimated_total = elapsed * total_combinations
print(f"Estimated total running time: {estimated_total/60:.1f} minutes")

import concurrent.futures

def process_combination(args):
    metrics, agg_funcs, combo_id, total_combinations, all_metrics, all_agg_funcs, output_dir, stats_dir = args
    combo_name = f"metrics_{'_'.join(metrics)}__agg_{'_'.join(agg_funcs)}"
    print(f"Running combination {combo_id}/{total_combinations}: {combo_name}")

    builder = KmerFeatureBuilderAgg(stats_dir=stats_dir)
    feature_matrix = builder.build_aggregated_features(
        metrics=list(metrics), agg_funcs=list(agg_funcs), n_jobs=4
    )
    feature_matrix_std = builder.standardize_features()

    feature_matrix_std.to_csv(f"{output_dir}/{combo_name}_features.csv")

    n_pca = min(2, feature_matrix_std.shape[1])
    X_pca = reduce_dimensions(feature_matrix_std, method="pca", n_components=n_pca)
    np.savetxt(f"{output_dir}/{combo_name}_X_pca.csv", X_pca, delimiter=",")

    # KMeans clustering and metrics
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(10, len(feature_matrix_std)))
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
        inertias.append(kmeans.inertia_)
        labels = kmeans.labels_
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(X_pca, labels))
        else:
            silhouette_scores.append(float('nan'))

    plt.figure()
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel("n_clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for KMeans")
    plt.savefig(f"{output_dir}/{combo_name}_elbow.png")
    plt.close()

    plt.figure()
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for KMeans")
    plt.savefig(f"{output_dir}/{combo_name}_silhouette.png")
    plt.close()

    min_samples = 5
    if X_pca.shape[0] > min_samples:
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(X_pca)
        distances, indices = nbrs.kneighbors(X_pca)
        distances = np.sort(distances[:, min_samples-1])
        plt.figure()
        plt.plot(distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{min_samples}th NN distance")
        plt.title("k-distance plot for DBSCAN")
        plt.savefig(f"{output_dir}/{combo_name}_dbscan_kdist.png")
        plt.close()

    selector = VarianceThreshold(threshold=0.01)
    selector.fit(feature_matrix_std)
    selected_features = feature_matrix_std.columns[selector.get_support()]
    with open(f"{output_dir}/{combo_name}_selected_features.txt", "w") as f:
        f.write("Selected features by variance:\n")
        f.write("\n".join(selected_features))

    corr = feature_matrix_std.corr()
    corr.to_csv(f"{output_dir}/{combo_name}_corr.csv")
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{combo_name}_corr.png")
    plt.close()

    # UMAP
    try:
        X_umap = reduce_dimensions(feature_matrix_std, method="umap", n_components=n_pca)
        np.savetxt(f"{output_dir}/{combo_name}_X_umap.csv", X_umap, delimiter=",")
    except Exception as e:
        print(f"UMAP failed for {combo_name}: {e}")
        X_umap = None

    # --- KMeans ---
    kmeans_pca = KMeans(n_clusters=3, random_state=42).fit(X_pca)
    clusters_pca_kmeans = kmeans_pca.labels_
    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters_pca_kmeans, cmap='tab10')
    plt.title("KMeans Clustering (PCA)")
    plt.savefig(f"{output_dir}/{combo_name}_kmeans_scatter.png")
    plt.close()

    if X_umap is not None:
        kmeans_umap = KMeans(n_clusters=3, random_state=42).fit(X_umap)
        clusters_umap_kmeans = kmeans_umap.labels_
        plt.figure()
        plt.scatter(X_umap[:,0], X_umap[:,1], c=clusters_umap_kmeans, cmap='tab10')
        plt.title("KMeans Clustering (UMAP)")
        plt.savefig(f"{output_dir}/{combo_name}_kmeans_umap_scatter.png")
        plt.close()

    # --- DBSCAN ---
    dbscan_pca = DBSCAN(eps=4, min_samples=5).fit(X_pca)
    clusters_pca_dbscan = dbscan_pca.labels_
    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters_pca_dbscan, cmap='tab10')
    plt.title("DBSCAN Clustering (PCA)")
    plt.savefig(f"{output_dir}/{combo_name}_dbscan_scatter.png")
    plt.close()

    if X_umap is not None:
        dbscan_umap = DBSCAN(eps=4, min_samples=5).fit(X_umap)
        clusters_umap_dbscan = dbscan_umap.labels_
        plt.figure()
        plt.scatter(X_umap[:,0], X_umap[:,1], c=clusters_umap_dbscan, cmap='tab10')
        plt.title("DBSCAN Clustering (UMAP)")
        plt.savefig(f"{output_dir}/{combo_name}_dbscan_umap_scatter.png")
        plt.close()

    # --- GMM ---
    gmm_pca = GaussianMixture(n_components=3, random_state=42).fit(X_pca)
    clusters_pca_gmm = gmm_pca.predict(X_pca)
    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters_pca_gmm, cmap='tab10')
    plt.title("GMM Clustering (PCA)")
    plt.savefig(f"{output_dir}/{combo_name}_gmm_scatter.png")
    plt.close()

    if X_umap is not None:
        gmm_umap = GaussianMixture(n_components=3, random_state=42).fit(X_umap)
        clusters_umap_gmm = gmm_umap.predict(X_umap)
        plt.figure()
        plt.scatter(X_umap[:,0], X_umap[:,1], c=clusters_umap_gmm, cmap='tab10')
        plt.title("GMM Clustering (UMAP)")
        plt.savefig(f"{output_dir}/{combo_name}_gmm_umap_scatter.png")
        plt.close()

    # Dendrogram
    try:
        if X_pca.shape[0] > 2:
            Z = fastcluster.linkage_vector(X_pca, method="ward")
            scipy.io.savemat(f"{output_dir}/{combo_name}_linkage_matrix.mat", {"Z": Z})
            plt.figure(figsize=(12, 5))
            dendrogram(Z, labels=feature_matrix_std.index.tolist(), leaf_rotation=90)
            plt.title("Dendrograma com PCA")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{combo_name}_dendrogram.png")
            plt.close()
    except Exception as e:
        print(f"Dendrogram failed for {combo_name}: {e}")

if __name__ == "__main__":
    args_list = []
    combo_id = 0
    for metrics in metric_combos:
        for agg_funcs in agg_func_combos:
            combo_id += 1
            args_list.append((metrics, agg_funcs, combo_id, total_combinations, all_metrics, all_agg_funcs, output_dir, stats_dir))

    max_workers = 22  # Ajuste conforme sua CPU/RAM
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_combination, args_list)

    print("All combinations finished.")