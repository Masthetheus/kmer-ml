import itertools
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .clustering import get_clustering_methods
from .evaluation import evaluate_clustering, rank_results
from .visualization import plot_clustering_results

class ClusteringScout:
    """
    Classe principal para scouting automático de métodos de clustering
    """
    
    def __init__(self, methods: Optional[List[str]] = None):
        """
        Args:
            methods: Lista de métodos para testar. Se None, testa todos.
        """
        all_methods = get_clustering_methods()
        if methods is None:
            self.methods = all_methods
        else:
            self.methods = {k: v for k, v in all_methods.items() if k in methods}
    
    def scout(self, X: np.ndarray, output_dir: str = "data/results/") -> Dict[str, Any]:
        """
        Executa scouting completo de métodos de clustering
        
        Args:
            X: Dados para clustering
            output_dir: Diretório para salvar resultados
            
        Returns:
            Dicionário com melhor resultado e todos os resultados
        """
        all_results = []
        
        print(f"Iniciando scouting com {len(self.methods)} métodos...")
        
        for method_name, method in self.methods.items():
            print(f"Testando {method.name}...")
            
            # Gera todas as combinações de parâmetros
            param_combinations = [
                dict(zip(method.param_grid.keys(), values))
                for values in itertools.product(*method.param_grid.values())
            ]
            
            for params in param_combinations:
                try:
                    # Executa clustering
                    labels = method.fit_predict(X, **params)
                    
                    # Avalia resultado
                    metrics = evaluate_clustering(X, labels)
                    
                    # Armazena resultado
                    result = {
                        'method': method.name,
                        'method_key': method_name,
                        'params': params,
                        'labels': labels,
                        'metrics': metrics
                    }
                    all_results.append(result)
                    
                    print(f"  {params} -> Silhouette: {metrics['silhouette']:.3f}")
                    
                except Exception as e:
                    print(f"  Erro com {params}: {e}")
                    continue
        
        # Rankeia resultados
        ranked_results = rank_results(all_results)
        best_result = ranked_results[0] if ranked_results else None
        
        if best_result:
            print(f"\nMelhor resultado:")
            print(f"Método: {best_result['method']}")
            print(f"Parâmetros: {best_result['params']}")
            print(f"Silhouette: {best_result['metrics']['silhouette']:.3f}")
            print(f"Clusters: {best_result['metrics']['n_clusters']}")
        
        # Salva resultados
        self._save_results(all_results, ranked_results, output_dir)
        
        return {
            'best_result': best_result,
            'all_results': all_results,
            'ranked_results': ranked_results
        }
    
    def _save_results(self, all_results: List, ranked_results: List, output_dir: str):
        """Salva resultados em CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Converte para DataFrame
        data = []
        for result in all_results:
            row = {
                'method': result['method'],
                'params': str(result['params']),
                **result['metrics']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/clustering_scout_results.csv", index=False)
        
        # Salva ranking
        ranking_data = []
        for i, result in enumerate(ranked_results[:10]):  # Top 10
            row = {
                'rank': i + 1,
                'method': result['method'],
                'params': str(result['params']),
                **result['metrics']
            }
            ranking_data.append(row)
        
        df_rank = pd.DataFrame(ranking_data)
        df_rank.to_csv(f"{output_dir}/clustering_ranking.csv", index=False)