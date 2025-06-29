import pandas as pd
from Bio import Entrez

# def add_phylum_family_to_tsv(tsv_path, output_path, taxid_column="tax_id", email="seu@email.com"):
#     """
#     Lê um TSV com uma coluna tax_id, adiciona colunas 'phylum' e 'family' usando o NCBI, e salva o resultado.
#     """
#     Entrez.email = email
#     df = pd.read_csv(tsv_path, sep="\t")
#     phyla = []
#     families = []
#     for taxid in df[taxid_column]:
#         try:
#             handle = Entrez.efetch(db="taxonomy", id=str(taxid), retmode="xml")
#             records = Entrez.read(handle)
#             lineage = {item['Rank']: item['ScientificName'] for item in records[0]['LineageEx']}
#             phyla.append(lineage.get('phylum', 'NA'))
#             families.append(lineage.get('family', 'NA'))
#         except Exception as e:
#             print(f"Erro ao buscar taxid {taxid}: {e}")
#             phyla.append('NA')
#             families.append('NA')
#     df['phylum'] = phyla
#     df['family'] = families
#     df.to_csv(output_path, sep="\t", index=False)
#     print(f"Arquivo salvo em: {output_path}")

# add_phylum_family_to_tsv('data/ncbi_dataset.tsv', 'data/ncbi_dataset.tsv', taxid_column='Organism Taxonomic ID',email = 'matheuspedroncassol@gmail.com')

# def summarize_taxa(tsv_path, phylum_col='phylum', family_col='family', output_prefix='summary'):
#     df = pd.read_csv(tsv_path, sep='\t')
#     total = len(df)

#     # Phylum
#     phylum_counts = df[phylum_col].value_counts(dropna=False).reset_index()
#     phylum_counts.columns = [phylum_col, 'count']
#     phylum_counts['percent'] = 100 * phylum_counts['count'] / total
#     phylum_counts.to_csv(f"{output_prefix}_phylum.tsv", sep='\t', index=False)

#     # Family
#     family_counts = df[family_col].value_counts(dropna=False).reset_index()
#     family_counts.columns = [family_col, 'count']
#     family_counts['percent'] = 100 * family_counts['count'] / total
#     family_counts.to_csv(f"{output_prefix}_family.tsv", sep='\t', index=False)

#     print(f"Resumo salvo em {output_prefix}_phylum.tsv e {output_prefix}_family.tsv")
#     print(phylum_counts)
#     print(family_counts)

# summarize_taxa('data/ncbi_dataset.tsv', phylum_col='phylum', family_col='family', output_prefix='data/ncbi_summary')

def filter_for_clustering_unique_species(tsv_path, output_path, phylum_col='phylum', family_col='family', species_col='species', max_per_family=20, min_per_family=3):
    df = pd.read_csv(tsv_path, sep='\t')
    # Remove missing phylum, family, or species
    df = df[df[phylum_col].notna() & (df[phylum_col] != '') &
            df[family_col].notna() & (df[family_col] != '') &
            df[species_col].notna() & (df[species_col] != '')]
    # Keep only one genome per species per family
    df = df.drop_duplicates(subset=[family_col, species_col])
    # Keep only families with at least min_per_family representatives
    family_counts = df[family_col].value_counts()
    valid_families = family_counts[family_counts >= min_per_family].index
    df = df[df[family_col].isin(valid_families)]
    # Sample up to max_per_family per family
    df_filtered = df.groupby(family_col).apply(lambda x: x.sample(n=min(len(x), max_per_family), random_state=42)).reset_index(drop=True)
    df_filtered.to_csv(output_path, sep='\t', index=False)
    print(f"Filtered dataset saved to {output_path}")

filter_for_clustering_unique_species('data/ncbi_dataset.tsv', 'data/ncbi_dataset_filtered.tsv', phylum_col='phylum', family_col='family', species_col='Organism Name', max_per_family=20, min_per_family=3)

def preview_filtering(tsv_path, family_col='family', min_per_family=3, max_per_family=20):
    df = pd.read_csv(tsv_path, sep='\t')
    df = df[df[family_col].notna() & (df[family_col] != '')]
    family_counts = df[family_col].value_counts()
    valid_families = family_counts[family_counts >= min_per_family].index
    df = df[df[family_col].isin(valid_families)]
    kept_per_family = df.groupby(family_col).size().clip(upper=max_per_family)
    total_genomes = kept_per_family.sum()
    print(f"Families kept: {len(kept_per_family)}")
    print(f"Total genomes after filtering: {total_genomes}")

preview_filtering('data/ncbi_dataset_filtered.tsv', family_col='family', min_per_family=3, max_per_family=20)