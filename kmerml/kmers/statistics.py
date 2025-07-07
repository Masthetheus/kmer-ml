import pandas as pd
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
from kmerml.utils.progress import progress_bar
import gzip

class KmerFeatureExtractor:
    """Extract machine learning features from k-mer data"""
    
    def __init__(self, input_paths=None, output_dir=None, metadata_file=None):
        """
        Initialize feature extractor for k-mer data.
        
        Args:
            input_paths (list): Paths to k-mer files or directories
            output_dir (str): Directory to save feature files
            metadata_file (str): Path to genome metadata file
        """
        self.input_paths = [Path(p) for p in input_paths] if input_paths else []
        self.output_dir = Path(output_dir) if output_dir else Path("kmer_features")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load metadata if provided
        self.metadata = None
        if metadata_file:
            from kmerml.utils.genome_metadata import GenomeMetadataManager
            self.metadata_manager = GenomeMetadataManager(metadata_file)
    
    def add_paths(self, paths):
        """Add additional paths to process"""
        self.input_paths.extend([Path(p) for p in paths])
    
    def extract_features(self, required_features=None):
        """
        Extract features from all input files and generate CSV files.
        
        Args:
            required_features (list): List of features to include
                                     (defaults to all available)
        
        Returns:
            dict: Mapping of organism names to output CSV paths
        """
        # Define default features if not specified
        if required_features is None:
            required_features = [
                'relative_freq', 'gc_skew', 'at_skew',
                'shannon_entropy', 'normalized_entropy',
                'is_palindrome'
            ]
        
        # Group files by organism
        files_by_organism = self._group_files_by_organism()
        
        # For progress bar
        organisms = list(files_by_organism.keys())
        organisms = len(organisms)
        cont = 1
        start = time.time()
        # Process each organism
        output_files = {}
        for organism, files in files_by_organism.items():
            output_csv = self._process_organism_kmers(
                organism, files, required_features
            )
            output_files[organism] = output_csv
            start = progress_bar(cont, organisms, start_time=start, title="Organisms processed")
            cont += 1
            
        return output_files
    
    def _group_files_by_organism(self):
        """Group input files by organism name"""
        files_by_organism = defaultdict(list)
        
        for path in self.input_paths:
            # Case 1: File directly in organism directory
            # e.g., /path/to/kmers/GCF_123456/k8.txt
            if path.is_file():
                organism = path.parent.name
                files_by_organism[organism].append(path)
            
            # Case 2: Directory contains organism directories
            # e.g., /path/to/kmers/ containing organism folders
            elif path.is_dir():
                for org_dir in path.iterdir():
                    if org_dir.is_dir():
                        organism = org_dir.name
                        for kmer_file in org_dir.glob("k*.txt*"):
                            files_by_organism[organism].append(kmer_file)
        
        return files_by_organism
    
    def _process_organism_kmers(self, organism, kmer_files, required_features):
        """Process all k-mer files for an organism"""
        # Get genome metadata if available
        genome_size = None
    
        if hasattr(self, 'metadata_manager'):
            genome_size = self.metadata_manager.get_genome_size(organism)
    
        # Process each k-mer file
        all_features = []
    
        # For the progress bar
        kmertot = len(kmer_files)
        cont = 0
        start = time.time()
    
        for kmer_file in kmer_files:
            # Extract k value
            k_val = self._extract_k_from_filename(kmer_file.name)
            if k_val is None:
                print(f"Warning: Could not extract k value from {kmer_file}")
                continue
    
            # Load the file
            df = self._load_kmer_file(kmer_file)
    
            # Process each k-mer
            file_features = self._extract_kmer_features(
                df, k_val, organism=organism, required_features=required_features
            )
            if file_features is not None:
                all_features.extend(file_features)
    
            cont += 1
            start = progress_bar(cont, kmertot, start_time=start, title="K values processed")
    
        # Create DataFrame
        if not all_features:
            print(f"No features extracted for {organism}")
            return None
    
        result_df = pd.DataFrame(all_features)
    
        # Save to CSV
        output_file = self.output_dir / f"{organism}_kmer_features.csv"
        result_df.to_csv(output_file, index=False)
        print(f"Created feature CSV for {organism}: {output_file}")
    
        return output_file
    
    def _extract_kmer_features(self, df, k_val, required_features=None):
        """
        Extract features from k-mer DataFrame.
        Features: relative frequency, GC%, skew, entropy, palindromes, non-canonical.
        Global features: proportion of palindromes, unique, repeated, non-canonical.
        """
        features_list = []
        kmers = df['kmer'].values
        counts = df['count'].values
        total_kmers = counts.sum()
    
        # Pre compute masks for unique and repeated k-mers
        unique_mask = counts == 1
        repeated_mask = counts > 1
    
        # Global features
        unique_ratio = unique_mask.sum() / len(counts) if len(counts) > 0 else 0
        repeated_ratio = repeated_mask.sum() / len(counts) if len(counts) > 0 else 0

        # Optmized function to check if a k-mer is a palindrome
        def is_palindrome_kmer(kmer):
            complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
            k = len(kmer)
            for i in range(k // 2):
                if kmer[i] != complement.get(kmer[-(i+1)], kmer[-(i+1)]):
                    return 0
            return 1

        # Process each k-mer
        for i in range(len(kmers)):
            kmer = kmers[i]
            count = counts[i]
            kmer_features = {
                'kmer': kmer,
                'count': count,
                'k': k_val,
                'relative_freq': count / total_kmers if total_kmers > 0 else 0
            }
            # Skew
            g = kmer.count('G')
            c = kmer.count('C')
            a = kmer.count('A')
            t = kmer.count('T')
            kmer_features['gc_skew'] = (g - c) / (g + c) if (g + c) > 0 else 0
            kmer_features['at_skew'] = (a - t) / (a + t) if (a + t) > 0 else 0
            # GC%
            gc = g+c
            kmer_features['gc_percent'] = (gc / len(kmer)) * 100 if len(kmer) > 0 else 0
            # Entropia de Shannon
            bases = np.array([a, c, g, t])
            probs = bases / len(kmer)
            entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
            kmer_features['shannon_entropy'] = entropy
            kmer_features['normalized_entropy'] = entropy / 2.0  # máximo para DNA
            # Palindromes
            kmer_features['is_palindrome'] = is_palindrome_kmer(kmer)
            # Non canonical
            kmer_features['noncanonical'] = int(any(base not in 'ACGT' for base in kmer))
            features_list.append(kmer_features)
    
        # Global features
        for f in features_list:
            f['unique_kmer_ratio'] = unique_ratio
            f['repeated_kmer_ratio'] = repeated_ratio
        # Palindrome and non-canonical ratios
        palindrome_ratio = sum(f['is_palindrome'] for f in features_list) / len(features_list) if features_list else 0
        noncanonical_ratio = sum(f['noncanonical'] for f in features_list) / len(features_list) if features_list else 0
        for f in features_list:
            f['palindrome_ratio'] = palindrome_ratio
            f['noncanonical_ratio'] = noncanonical_ratio
    
        return features_list
    
    def _extract_k_from_filename(self, filename):
        """Extract k value from filename like 'k8.txt'"""
        import re
        match = re.search(r'k(\d+)', filename)
        return int(match.group(1)) if match else None
    
    def _load_kmer_file(self, filepath):
        """Load a k-mer file using direct line parsing"""
        
        # Check if file is gzipped
        is_gzipped = str(filepath).endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        mode = 'rt' if is_gzipped else 'r'
        
        kmers = []
        counts = []
        
        with open_func(filepath, mode) as f:
            for line in f:
                # Split on any whitespace (tab or space)
                parts = line.strip().split()
                if len(parts) == 2:
                    kmer, count = parts
                    kmers.append(kmer)
                    try:
                        counts.append(int(count))
                    except ValueError:
                        print(f"Warning: Non-integer count in {filepath}: {count}")
                        counts.append(0)
        
        return pd.DataFrame({'kmer': kmers, 'count': counts})