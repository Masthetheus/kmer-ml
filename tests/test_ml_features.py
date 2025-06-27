from kmerml.ml.features import KmerFeatureBuilderAgg
from kmerml.ml.dimensionality import reduce_dimensions, plot_reduced
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import scipy.io
import scipy.io
from scipy.cluster.hierarchy import to_tree
import gc
import fastcluster

stats_dir = "data/processed/features"
builder = KmerFeatureBuilderAgg(stats_dir=stats_dir)
feature_matrix = builder.build_aggregated_features(
    metrics=['gc_percent', 'gc_skew', 'unique_kmer_ratio'],
    agg_funcs=['mean'],
    n_jobs=20
)
feature_matrix_std = builder.standardize_features()

print("Initializing dimensionality reduction...")
# Redução de dimensionalidade
X_pca = reduce_dimensions(feature_matrix_std, method="pca", n_components=2)
plot_reduced(X_pca, labels=None, title="PCA - Organismos")

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# inertias = []
# silhouette_scores = []
# K_range = range(2, 10)
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
#     inertias.append(kmeans.inertia_)
#     labels = kmeans.labels_
#     silhouette_scores.append(silhouette_score(X_pca, labels))

# plt.figure()
# plt.plot(K_range, inertias, marker='o')
# plt.xlabel("n_clusters")
# plt.ylabel("Inertia")
# plt.title("Elbow Method for KMeans")
# plt.show()

# plt.figure()
# plt.plot(K_range, silhouette_scores, marker='o')
# plt.xlabel("n_clusters")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score for KMeans")
# plt.show()

from sklearn.neighbors import NearestNeighbors

min_samples = 5  # or try 2 * X_pca.shape[1]
neigh = NearestNeighbors(n_neighbors=min_samples)
nbrs = neigh.fit(X_pca)
distances, indices = nbrs.kneighbors(X_pca)
distances = np.sort(distances[:, min_samples-1])
plt.figure()
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{min_samples}th NN distance")
plt.title("k-distance plot for DBSCAN")
plt.show()
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
selector.fit(feature_matrix_std)
selected_features = feature_matrix_std.columns[selector.get_support()]
print("Selected features by variance:", list(selected_features))

corr = feature_matrix_std.corr()
print(corr)
# Optionally, visualize with seaborn heatmap
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

X_umap = reduce_dimensions(feature_matrix_std, method="umap", n_components=2)
plot_reduced(X_umap, labels=None, title="UMAP - Organismos")

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)  # Use PCA-reduced data
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=4, min_samples=5)
clusters = dbscan.fit_predict(X_pca)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4, random_state=42)
clusters = gmm.fit_predict(X_pca)
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("Alternative Clustering")
plt.show()

np.savetxt("data/results/X_pca.csv", X_pca, delimiter=",")
np.savetxt("data/results/X_umap.csv", X_umap, delimiter=",")

pca = PCA(n_components=feature_matrix_std.shape[1])
pca.fit(feature_matrix_std)
plt.figure()
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Número de componentes")
plt.ylabel("Variância explicada acumulada")
plt.title("Scree plot PCA")
plt.grid()
plt.tight_layout()
plt.savefig("data/results/scree_plot_pca.png")
# plt.show()

def write_newick(node, labels, file_handle, parentdist=0):
    """Escreve o Newick diretamente em disco durante a travessia da árvore."""
    if node.is_leaf():
        name = labels[node.id] if labels is not None else str(node.id)
        file_handle.write(f"{name}:{parentdist - node.dist:.2f}")
    else:
        file_handle.write("(")
        write_newick(node.get_left(), labels, file_handle, node.dist)
        file_handle.write(",")
        write_newick(node.get_right(), labels, file_handle, node.dist)
        file_handle.write(f"):{parentdist - node.dist:.2f}")


# --- Clusterização e dendrograma ---
print("Performing hierarchical clustering (10 PCs)...")

Z = fastcluster.linkage_vector(X_pca, method="ward")
scipy.io.savemat("data/results/newward_10pc_linkage_matrix.mat", {"Z": Z})
plt.figure(figsize=(12, 5))
dendrogram(Z, labels=feature_matrix_std.index.tolist(), leaf_rotation=90)
plt.title("Dendrograma com 10 PCs")
plt.tight_layout()
plt.show()

tree = to_tree(Z)
labels = list(feature_matrix_std.index)
with open("data/results/newward_tree_10pc.nwk", "w") as f:
    write_newick(tree, labels, f)
    f.write(";")
