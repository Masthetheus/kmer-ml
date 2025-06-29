import os
import pandas as pd
import numpy as np
import re

output_dir = "data/results/auto_combinations_all_org"

summary = []

for fname in os.listdir(output_dir):
    if fname.endswith("_silhouette.csv"):
        combo_name = fname.replace("_silhouette.csv", "")
        silhouette_csv = os.path.join(output_dir, fname)
        try:
            sil = pd.read_csv(silhouette_csv, header=None)
            # Take the max silhouette score (best k)
            silhouette_val = sil.values.flatten().max()
        except Exception:
            silhouette_val = np.nan

        # Try to get number of clusters from the best k (row index of max silhouette)
        try:
            best_k_idx = sil.values.flatten().argmax()
            best_k = best_k_idx + 2  # since K_range starts at 2
        except Exception:
            best_k = np.nan

        # Optionally, count number of features from combo_name
        metrics_match = re.search(r"metrics_([^_]+(?:_[^_]+)*)__agg_", combo_name)
        n_features = len(metrics_match.group(1).split("_")) if metrics_match else np.nan

        summary.append({
            "combo": combo_name,
            "silhouette": silhouette_val,
            "best_k": best_k,
            "n_features": n_features
        })

# Convert to DataFrame and rank
df = pd.DataFrame(summary)
df = df.sort_values(by="silhouette", ascending=False)

# Save summary
df.to_csv(os.path.join(output_dir, "summary_ranked_combinations.csv"), index=False)
print(df.head(10))