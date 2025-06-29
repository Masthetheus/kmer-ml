import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler
from kmerml.utils.path_utils import find_files
import concurrent.futures

class KmerFeatureBuilderAgg:
    """Build ML-ready feature matrices by aggregating k-mer features per organism."""

    def __init__(self, stats_dir: Union[str, Path] = None):
        self.stats_dir = Path(stats_dir) if stats_dir else None
        self.feature_matrix = None
        self.organisms = []
        self.scaler = None

    def _process_file(self, file_path, metrics, agg_funcs):
        organism_id = self._extract_organism_id(file_path)
        try:
            df = pd.read_csv(file_path)
            if not all(m in df.columns for m in metrics):
                raise ValueError(f"Required columns not found in {file_path}.")
            agg = {}
            for metric in metrics:
                for func in agg_funcs:
                    agg[f"{metric}_{func}"] = getattr(df[metric], func)()
            return organism_id, agg
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return organism_id, None

    def build_aggregated_features(self,
                                 metrics: List[str] = None,
                                 agg_funcs: List[str] = None,
                                 file_pattern: str = "*kmer_features.csv",
                                 n_jobs: int = 4) -> pd.DataFrame:
        """
        Build aggregated feature matrix from k-mer statistics files.
        Each organism is a row; columns are aggregated statistics (mean, std, min, max) for each metric.
        Args:
            metrics: List of feature columns to aggregate (e.g., ["gc_percent", "shannon_entropy"])
            agg_funcs: List of aggregation functions (e.g., ["mean", "std", "min", "max"])
            file_pattern: Pattern to match statistics files
            n_jobs: Number of parallel workers (default: 4)
        Returns:
            DataFrame with organisms as rows and aggregated features as columns
        """
        if metrics is None:
            metrics = ['relative_freq', 'gc_skew', 'at_skew',
                'shannon_entropy', 'is_palindrome', 'unique_kmer_ratio',
                'palindrome_ratio', 'noncanonical_ratio']
        if agg_funcs is None:
            agg_funcs = ["mean", "std", "min", "max"]

        if not self.stats_dir:
            raise ValueError("Statistics directory not set")
        stats_files = find_files(self.stats_dir, patterns=[file_pattern], recursive=True)
        if not stats_files:
            raise ValueError(f"No statistics files found matching pattern: {file_pattern}")

        rows = []
        organisms = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._process_file, file_path, metrics, agg_funcs)
                for file_path in stats_files
            ]
            for future in concurrent.futures.as_completed(futures):
                organism_id, agg = future.result()
                if agg is not None:
                    rows.append(agg)
                    organisms.append(organism_id)

        self.feature_matrix = pd.DataFrame(rows, index=organisms)
        self.organisms = organisms
        return self.feature_matrix

    def _extract_organism_id(self, file_path: Path) -> str:
        name_parts = file_path.stem.split('_')
        if len(name_parts) >= 2:
            return f"{name_parts[0]}_{name_parts[1]}"
        return file_path.stem

    def standardize_features(self) -> pd.DataFrame:
        """
        Standardize the aggregated feature matrix (z-score: mean=0, std=1).
        Returns:
            Standardized DataFrame (same shape as feature_matrix)
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not built yet.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.feature_matrix.values)
        self.scaler = scaler
        df_scaled = pd.DataFrame(X_scaled, index=self.feature_matrix.index, columns=self.feature_matrix.columns)
        return df_scaled