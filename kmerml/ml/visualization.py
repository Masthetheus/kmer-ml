import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

def plot_clustering_results(X: np.ndarray, 
                          labels: np.ndarray,
                          method_name: str,
                          output_path: Optional[str] = None,
                          biological_labels: Optional[np.ndarray] = None) -> None:
    """
    Plots clustering results in 2D space.
    
    Args:
        X: Data (assuming 2D for visualization)
        labels: Labels for clusters
        method_name: Method name
        output_path: Path to save the plot (if None, only shows plot)
        biological_labels: Family, genus or species labels for coloring (optional)
    """
    fig, axes = plt.subplots(1, 2 if biological_labels is not None else 1, 
                            figsize=(15, 6) if biological_labels is not None else (8, 6))
    
    if biological_labels is not None:
        axes = [axes] if not hasattr(axes, '__len__') else axes
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1 = axes
    
    # Plot 1: Clusters
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.7)
    ax1.set_title(f"{method_name} Clustering")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    
    # Plot 2: Biological labels
    if biological_labels is not None:
        unique_labels = np.unique(biological_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = biological_labels == label
            ax2.scatter(X[mask, 0], X[mask, 1], c=[color], label=label, alpha=0.7)
        
        ax2.set_title("Biological Labels")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_scout_summary(results: Dict[str, Any], output_dir: str):
    """
    Plots a summary of clustering scouting results.
    """
    all_results = results['all_results']
    
    # Data extraction
    methods = [r['method'] for r in all_results]
    silhouette_scores = [r['metrics']['silhouette'] for r in all_results]
    n_clusters = [r['metrics']['n_clusters'] for r in all_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Silhouette per method
    df_plot = pd.DataFrame({'Method': methods, 'Silhouette': silhouette_scores})
    sns.boxplot(data=df_plot, x='Method', y='Silhouette', ax=axes[0, 0])
    axes[0, 0].set_title('Silhouette Score per Method')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Clusters per method
    df_plot2 = pd.DataFrame({'Method': methods, 'N_Clusters': n_clusters})
    sns.boxplot(data=df_plot2, x='Method', y='N_Clusters', ax=axes[0, 1])
    axes[0, 1].set_title('Clusters found per Method')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Metrics correlation
    metrics_data = {
        'Silhouette': [r['metrics']['silhouette'] for r in all_results],
        'Calinski-Harabasz': [r['metrics']['calinski_harabasz'] for r in all_results],
        'Davies-Bouldin': [r['metrics']['davies_bouldin'] for r in all_results if r['metrics']['davies_bouldin'] != float('inf')]
    }
    
    df_corr = pd.DataFrame(metrics_data)
    sns.heatmap(df_corr.corr(), annot=True, ax=axes[1, 1])
    axes[1, 1].set_title('Metrics correlation')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scout_summary.png", dpi=300, bbox_inches='tight')
    plt.close()