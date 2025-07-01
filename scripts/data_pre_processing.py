from kmerml.utils.data_processing import add_phylum_family_to_tsv, summarize_taxa, filter_for_clustering_unique_species, preview_filtering
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process NCBI dataset TSV files.")
    parser.add_argument(
        "--accession-list",
        type=str,
        default="data/ncbi_dataset.tsv",
        help="Path to the file containing assembly accession numbers (default: data/ncbi_dataset.tsv)"
    )
    parser.add_argument(
        "--taxid-column",
        type=str,
        default="Organism Taxonomic ID",
        help="Column name for taxonomic IDs in the TSV file (default: Organism Taxonomic ID)"
    )
    parser.add_argument(
        "--email",
        type=str
    )
    args = parser.parse_args()
    accession_file = args.accession_list
    taxid_column = args.taxid_column
    email = args.email
    print(f"Consulting NCBI for taxonomic information using email: {email}")
    add_phylum_family_to_tsv(accession_file, accession_file, taxid_column=taxid_column, email=email)
    print("Taxonomic information added to the TSV file.")
    print("Summarizing taxa...")
    summarize_taxa(accession_file, phylum_col='phylum', family_col='family ', output_prefix='data/ncbi_summary')
    print("Taxonomic summary saved.")
    print("Do you want to preview the filtering for clustering unique species? (yes/[no])")
    preview = input().strip().lower()
    if preview == 'yes':
        i = True
        while i:
            try:
                print("Please, specify max and min per family. To exit, type 'exit'.")
                max_per_family = int(input("Max per family: "))
                if max_per_family == 'exit':
                    print("Exiting preview.")
                    i = False
                    return
                min_per_family = int(input("Min per family: "))
            except ValueError:
                print("Invalid input. Please enter integers for max and min per family.")
            print("Previewing filtering...")
            preview_filtering(accession_file, family_col='family', min_per_family=3, max_per_family=20)
    else:
        print("Skipping preview.")
    print("Filtering for clustering unique species...")
    print("Please, specify max and min per family.")
    max_per_family = int(input("Max per family: "))
    min_per_family = int(input("Min per family: "))
    filter_for_clustering_unique_species(accession_file, 'data/ncbi_dataset_filtered.tsv',
                                         phylum_col='phylum', family_col='family', species_col='Organism Name',
                                         max_per_family=20, min_per_family=3)

if __name__ == "__main__":
    main()