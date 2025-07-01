from ete3 import NCBITaxa, Tree
import pandas as pd
import glob
import argparse

def tree_validation(reference_path, search_path):
    """
    Generate a reference tree from NCBI taxids and compare it with a search tree.
    This function retrieves taxonomic information for a list of organisms, builds a reference tree,
    and compares it with a search tree using the Robinson-Foulds distance.
    The reference tree is saved in Newick format.
    """
   
    reference_tree = Tree(reference_path)
    # Path to the search tree
    if search_path is None:
        search_path = glob.glob("data/results/*.nwk")[0]
    else:
        if not search_path.endswith(".nwk"):
            raise ValueError("Search path must point to a Newick file (.nwk)")
    t = Tree(search_path)
    response = input(f"Do you wish to print the trees? (yes/[no])")
    if response.lower() == 'yes':
        print("Printing found tree:")
        print(t)
        print("Printing reference tree:")
        print(reference_tree)
        
    rf, max_rf, *_ = reference_tree.robinson_foulds(t)
    norm_rf = rf / max_rf if max_rf else 0
    
    print(f"Robinson-Foulds distance: {rf}, Normalized RF: {norm_rf:.4f}")
    if norm_rf > 0.1:
        print("Warning: Trees differ significantly (RF distance > 0.1)")
    else:
        print("Trees are similar (RF distance <= 0.1)")

def reference_tree_building():
    """
    Build a reference tree from NCBI taxids and save it in Newick format.
    This function retrieves taxonomic information for a list of organisms,
    builds a tree using the NCBITaxa API, and saves it as 'reference_tree.nwk'.
    """
    
    # Load organism names from the TSV file
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

def main():
    parser = argparse.ArgumentParser(description="Tree validation script")
    parser.add_argument(
        "--reference-tree",
        type=str,
        default="data/reference_tree.nwk",
        help="Path to the reference tree in Newick format (default: data/reference_tree.nwk)"
    )
    parser.add_argument(
        "--search-tree",
        type=str,
        default=None,
        help="Path to the search tree in Newick format (default: first found in data/results/*.nwk)"
    )
    parser.add_argument(
        "--build-reference",
        action='store_true',
        help="Build the reference tree if it does not exist"
    )
    args = parser.parse_args()

    # Build reference tree if it doesn't exist
    if not args.build_reference and not args.reference_tree.endswith(".nwk"):
        print(f"Reference tree file {args.reference_tree} does not exist. Use --build-reference to create it.")
        return
    if args.build_reference:
        print("Building reference tree...")
        reference_tree_building()
        print("Reference tree built successfully.")
        return

    tree_validation(args.reference_tree, args.search_tree)

if __name__ == '__main__':
    main()
