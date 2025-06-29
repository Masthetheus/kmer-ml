import os

output_dir = "data/results/auto_combinations_all_org"
combos = []

for fname in os.listdir(output_dir):
    if fname.endswith("_silhouette.png"):
        combo_name = fname.replace("_silhouette.png", "")
        combos.append(combo_name)

with open("combos_to_fill.csv", "w") as f:
    f.write("combo,silhouette,best_k,n_features\n")
    for combo in combos:
        # Optionally, count features:
        n_features = combo.split("metrics_")[1].split("__agg_")[0].count("_") + 1
        f.write(f"{combo},,,{n_features}\n")