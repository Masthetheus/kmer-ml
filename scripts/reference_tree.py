from ete3 import NCBITaxa, Tree
import pandas as pd

df = pd.read_csv("data/ncbi_dataset.tsv", sep="\t")
organism_names = df["Organism Name"].tolist()


df["Assembly Accession Short"] = df["Assembly Accession"].str.split(".").str[0]
name_to_gcf = dict(zip(df["Organism Name"], df["Assembly Accession Short"]))

ncbi = NCBITaxa()

# Get NCBI taxids for your organisms
name2taxid = ncbi.get_name_translator(organism_names)
taxids = [name2taxid[name][0] for name in organism_names if name in name2taxid]

# Build the tree
tree = ncbi.get_topology(taxids)

for leaf in tree.iter_leaves():
    for name, tid in name2taxid.items():
        if leaf.name == str(tid[0]):
            leaf.name = name_to_gcf.get(name, name)

tree.write(outfile="reference_tree.nwk")
print("Reference tree saved as reference_tree.nwk")