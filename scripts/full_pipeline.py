import os
import argparse
import pandas as pd
import numpy as np
from kmerml.ml.features import KmerFeatureBuilderAgg
from kmerml.ml.dimensionality import reduce_dimensions
from kmerml.ml.scouting import ClusteringScout
from kmerml.ml.visualization import plot_clustering_results, plot_scout_summary

def load_biological_labels(labels_file: str, feature_index: pd.Index) -> pd.Series:
    """Carrega e alinha labels biológicos"""
    labels_df = pd.read_csv(labels_file, sep="\t", index_col=0)
    
    # Normaliza índices
    feature_index_clean = feature_index.str.replace(r'\.\d+$', '', regex=True)
    labels_df.index = labels_df.index.str.replace(r'\.\d+$', '', regex=True)
    
    # Alinha
    return labels_df.loc[feature_index_clean, "family"]

def analyze_biological_consistency(clusters: np.ndarray, 
                                 biological_labels: pd.Series,
                                 output_dir: str) -> None:
    """Analisa consistência entre clusters e labels biológicos"""
    df_analysis = pd.DataFrame({
        'cluster': clusters,
        'family': biological_labels.values
    })
    
    # Tabela cruzada
    crosstab = pd.crosstab(df_analysis['cluster'], df_analysis['family'])
    crosstab.to_csv(f"{output_dir}/cluster_family_crosstab.csv")
    
    # Conta famílias por cluster
    family_counts = df_analysis.groupby('cluster')['family'].value_counts()
    family_counts.to_csv(f"{output_dir}/family_counts_by_cluster.csv")
    
    print("Análise biológica salva em:")
    print(f"  - {output_dir}/cluster_family_crosstab.csv")
    print(f"  - {output_dir}/family_counts_by_cluster.csv")

def main():
    parser = argparse.ArgumentParser(description='K-mer ML Pipeline com Scouting')
    parser.add_argument('--stats-dir', default='data/processed/features',
                       help='Diretório com features calculadas')
    parser.add_argument('--output-dir', default='data/results/',
                       help='Diretório de saída')
    parser.add_argument('--labels-file', default='data/ncbi_dataset_all.tsv',
                       help='Arquivo com labels biológicos')
    parser.add_argument('--metrics', nargs='+', 
                       default=['gc_percent', 'gc_skew', 'unique_kmer_ratio', 
                               'palindrome_ratio', 'normalized_entropy'],
                       help='Métricas para usar')
    parser.add_argument('--agg-funcs', nargs='+', default=['mean'],
                       help='Funções de agregação')
    parser.add_argument('--methods', nargs='+', 
                       default=['kmeans', 'gmm', 'dbscan', 'agglomerative'],
                       help='Métodos de clustering para testar')
    parser.add_argument('--n-components', type=int, default=2,
                       help='Componentes para redução de dimensionalidade')
    parser.add_argument('--reduction-method', default='pca',
                       choices=['pca', 'umap'], help='Método de redução')
    
    args = parser.parse_args()
    
    # Cria diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== K-mer ML Pipeline com Scouting ===")
    print(f"Stats dir: {args.stats_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Métodos: {args.methods}")
    
    # 1. Carrega e processa features
    print("\n1. Carregando features...")
    builder = KmerFeatureBuilderAgg(stats_dir=args.stats_dir)
    feature_matrix = builder.build_aggregated_features(
        metrics=args.metrics,
        agg_funcs=args.agg_funcs,
        n_jobs=-1
    )
    feature_matrix_std = builder.standardize_features()
    print(f"   Features shape: {feature_matrix_std.shape}")
    
    # 2. Redução de dimensionalidade
    print(f"\n2. Redução de dimensionalidade ({args.reduction_method})...")
    X_reduced = reduce_dimensions(
        feature_matrix_std, 
        method=args.reduction_method, 
        n_components=args.n_components
    )
    np.savetxt(f"{args.output_dir}/X_{args.reduction_method}.csv", 
               X_reduced, delimiter=",")
    
    # 3. Carrega labels biológicos (se disponível)
    biological_labels = None
    if os.path.exists(args.labels_file):
        print("\n3. Carregando labels biológicos...")
        try:
            biological_labels = load_biological_labels(
                args.labels_file, feature_matrix_std.index
            )
            print(f"   Labels carregados: {len(biological_labels)} amostras")
        except Exception as e:
            print(f"   Erro ao carregar labels: {e}")
    
    # 4. Scouting de clustering
    print("\n4. Executando scouting de clustering...")
    scout = ClusteringScout(methods=args.methods)
    scout_results = scout.scout(X_reduced, args.output_dir)
    
    # 5. Análise do melhor resultado
    print("\n5. Analisando melhor resultado...")
    best_result = scout_results['best_result']
    
    if best_result:
        # Plota melhor resultado
        plot_clustering_results(
            X_reduced,
            best_result['labels'],
            f"{best_result['method']} (Best)",
            f"{args.output_dir}/best_clustering.png",
            biological_labels.values if biological_labels is not None else None
        )
        
        # Análise biológica
        if biological_labels is not None:
            analyze_biological_consistency(
                best_result['labels'],
                biological_labels,
                args.output_dir
            )
        
        # Plota resumo do scouting
        plot_scout_summary(scout_results, args.output_dir)
        
        # Salva resultado final
        final_results = pd.DataFrame({
            'sample_id': feature_matrix_std.index,
            'cluster': best_result['labels']
        })
        
        if biological_labels is not None:
            final_results['family'] = biological_labels.values
            
        final_results.to_csv(f"{args.output_dir}/final_clusters.csv", index=False)
        
        print(f"\nResultados salvos em: {args.output_dir}")
        print("Arquivos gerados:")
        print("  - clustering_scout_results.csv (todos os resultados)")
        print("  - clustering_ranking.csv (top 10)")
        print("  - best_clustering.png (visualização)")
        print("  - scout_summary.png (resumo)")
        print("  - final_clusters.csv (clusters finais)")
        
        if biological_labels is not None:
            print("  - cluster_family_crosstab.csv (análise biológica)")
            print("  - family_counts_by_cluster.csv")

if __name__ == "__main__":
    main()