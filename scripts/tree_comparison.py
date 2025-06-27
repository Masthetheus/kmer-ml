from ete3 import Tree
import glob

# Path to your reference tree
reference_path = "/home/leveduras/integranteslab/matheus/kmer-ml/reference_tree.nwk"
reference_tree = Tree(reference_path)
search_path = "/home/leveduras/integranteslab/matheus/kmer-ml/data/results/newward_tree_10pc.nwk"
t = Tree(search_path)
print(t)
print(reference_tree)
rf, max_rf, *_ = reference_tree.robinson_foulds(t)
norm_rf = rf / max_rf if max_rf else 0
print(f"Robinson-Foulds distance: {rf}, Normalized RF: {norm_rf:.4f}")