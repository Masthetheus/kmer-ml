import json
from pathlib import Path
import pandas as pd

class KmerMetadataManager:
    """Manage basic k-mer metadata and integrate with genome metadata"""
    
    def __init__(self, metadata_file="data/metadata/genome_metadata.json"):
        from kmerml.utils.genome_metadata import GenomeMetadataManager
        
        self.metadata_file = Path(metadata_file)
        # Leverage existing GenomeMetadataManager
        self.genome_manager = GenomeMetadataManager(metadata_file)
        self.metadata = self.genome_manager.metadata
    
    def add_kmer_metadata(self, kmer_files, recalculate=False):
        """
        Process k-mer files and add basic statistics to metadata.
        
        Args:
            kmer_files (list): List of k-mer files to process
            recalculate (bool): Whether to recalculate existing stats
            
        Returns:
            dict: Updated metadata
        """
        files_by_organism = self._group_files_by_organism(kmer_files)
        
        # Process each organism
        for organism, files in files_by_organism.items():
            # Create k-mer section in metadata if needed
            if organism not in self.metadata:
                self.metadata[organism] = {}
            
            if 'kmers' not in self.metadata[organism] or recalculate:
                self.metadata[organism]['kmers'] = {}
            
            # Process each k-mer file
            for kmer_file in files:
                k_val = self._extract_k_from_filename(kmer_file.name)
                if k_val is None:
                    continue
                    
                # Skip if already calculated and not recalculating
                if str(k_val) in self.metadata[organism]['kmers'] and not recalculate:
                    continue
                
                # Calculate statistics
                stats = self._calculate_basic_stats(kmer_file, k_val)
                
                # Store in metadata
                self.metadata[organism]['kmers'][str(k_val)] = stats
        
        # Save updated metadata
        self._save_metadata()
        
        return self.metadata
    
    def _calculate_basic_stats(self, kmer_file, k_val):
        """Calculate only basic statistics for a k-mer file"""
        df = self._load_kmer_file(kmer_file)
        
        # Basic statistics only
        stats = {
            'file_path': str(kmer_file),
            'k_value': k_val,
            'total_kmers': int(df['count'].sum()),
            'unique_kmers': len(df),
            'max_count': int(df['count'].max()),
            'min_count': int(df['count'].min()),
            'mean_count': float(df['count'].mean()),
            'median_count': float(df['count'].median())
        }
        
        # Calculate estimated genome size (simple formula)
        stats['estimated_genome_size'] = stats['total_kmers'] + k_val - 1
        
        return stats
    
    def _group_files_by_organism(self, kmer_files):
        """Group k-mer files by organism name"""
        from collections import defaultdict
        
        files_by_organism = defaultdict(list)
        
        for file_path in kmer_files:
            path = Path(file_path)
            organism = path.parent.name
            files_by_organism[organism].append(path)
        
        return files_by_organism
    
    def _extract_k_from_filename(self, filename):
        """Extract k value from filename like 'k8.txt'"""
        import re
        match = re.search(r'k(\d+)', str(filename))
        return int(match.group(1)) if match else None
    
    def _load_kmer_file(self, filepath):
        """Load a k-mer file into a pandas DataFrame"""
        # Check if file is gzipped
        is_gzipped = str(filepath).endswith('.gz')
        compression = 'gzip' if is_gzipped else None
        
        try:
            # Simple loading approach
            df = pd.read_csv(filepath, sep='\t', header=None, 
                           names=['kmer', 'count'], compression=compression)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame(columns=['kmer', 'count'])
        
        return df
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)