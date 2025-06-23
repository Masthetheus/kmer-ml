from pathlib import Path
import pandas as pd
import numpy as np
import math
from collections import defaultdict

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
                'base_counts', 'gc_content', 'cpg_sites', 
                'entropy', 'repeats', 'presence'
            ]
        
        # Group files by organism
        files_by_organism = self._group_files_by_organism()
        
        # Process each organism
        output_files = {}
        for organism, files in files_by_organism.items():
            output_csv = self._process_organism_kmers(
                organism, files, required_features
            )
            output_files[organism] = output_csv
            
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
        gc_content = None
        
        if hasattr(self, 'metadata_manager'):
            genome_size = self.metadata_manager.get_genome_size(organism)
            # Try to get GC content if available
            if organism in self.metadata_manager.metadata:
                gc_content = self.metadata_manager.metadata[organism].get('gc_content')
        
        # Process each k-mer file
        all_features = []
        
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
                df, k_val, organism, required_features
            )
            all_features.extend(file_features)
        
        # Create DataFrame
        if not all_features:
            print(f"No features extracted for {organism}")
            return None
        
        result_df = pd.DataFrame(all_features)
        
        # Add genome metadata
        if genome_size:
            result_df['genome_size'] = genome_size
        if gc_content:
            result_df['genome_gc_content'] = gc_content
        
        # Save to CSV
        output_file = self.output_dir / f"{organism}_kmer_features.csv"
        result_df.to_csv(output_file, index=False)
        print(f"Created feature CSV for {organism}: {output_file}")
        
        return output_file
    
    def _extract_kmer_features(self, df, k_val, organism, required_features):
        """Extract features for each k-mer in DataFrame"""
        features_list = []
        
        for _, row in df.iterrows():
            kmer_encoded = row['kmer']
            count = row['count']
            
            # Decode k-mer if necessary
            kmer = self._decode_kmer(kmer_encoded) if str(kmer_encoded).isdigit() else kmer_encoded
            
            # Basic features (always included)
            kmer_features = {
                'kmer': kmer,
                'count': count,
                'k': k_val,
                'organism': organism
            }
            
            # Add additional requested features
            if 'gc_content' in required_features:
                self._add_gc_features(kmer_features, kmer)
                
            if 'base_counts' in required_features:
                self._add_base_count_features(kmer_features, kmer)
                
            if 'presence' in required_features:
                self._add_presence_features(kmer_features, kmer)
                
            if 'cpg_sites' in required_features:
                self._add_cpg_features(kmer_features, kmer)
                
            if 'entropy' in required_features:
                self._add_entropy_features(kmer_features, kmer)
                
            if 'repeats' in required_features:
                self._add_repeat_features(kmer_features, kmer)
            
            features_list.append(kmer_features)
            
        return features_list
    
    def _add_gc_features(self, features_dict, kmer):
        """Add GC content related features"""
        gc_count = kmer.count('G') + kmer.count('C')
        features_dict['gc_percent'] = (gc_count / len(kmer)) * 100 if len(kmer) > 0 else 0
    
    def _add_base_count_features(self, features_dict, kmer):
        """Add base count features"""
        for base in 'ACGT':
            features_dict[f'{base}_count'] = kmer.count(base)
    
    def _add_presence_features(self, features_dict, kmer):
        """Add binary presence features"""
        for base in 'ACGT':
            features_dict[f'{base}_present'] = 1 if base in kmer else 0
    
    def _add_cpg_features(self, features_dict, kmer):
        """Add CpG site features"""
        # Count CpG dinucleotides
        cpg_count = sum(1 for i in range(len(kmer)-1) if kmer[i:i+2] == 'CG')
        features_dict['cpg_count'] = cpg_count
        
        # Calculate CpG observed/expected ratio
        c_freq = kmer.count('C') / len(kmer) if len(kmer) > 0 else 0
        g_freq = kmer.count('G') / len(kmer) if len(kmer) > 0 else 0
        expected = c_freq * g_freq * (len(kmer) - 1) if c_freq * g_freq > 0 else 0.001
        features_dict['cpg_obs_exp'] = cpg_count / expected if expected > 0 else 0
    
    def _add_entropy_features(self, features_dict, kmer):
        """Add sequence complexity/entropy features"""
        # Shannon entropy
        base_counts = {base: kmer.count(base) for base in set(kmer)}
        entropy = 0
        for base, count in base_counts.items():
            prob = count / len(kmer)
            entropy -= prob * math.log2(prob) if prob > 0 else 0
        features_dict['shannon_entropy'] = entropy
        
        # Normalize by max possible entropy (2 for DNA)
        features_dict['normalized_entropy'] = entropy / 2.0
    
    def _add_repeat_features(self, features_dict, kmer):
        """Add repeat sequence features"""
        # Detect simple repeats
        features_dict['has_repeat'] = 0
        
        # Look for dinucleotide repeats
        for i in range(len(kmer)-3):
            di = kmer[i:i+2]
            if di == kmer[i+2:i+4]:
                features_dict['has_repeat'] = 1
                break
    
    def _extract_k_from_filename(self, filename):
        """Extract k value from filename like 'k8.txt'"""
        import re
        match = re.search(r'k(\d+)', filename)
        return int(match.group(1)) if match else None
    
    def _decode_kmer(self, encoded_kmer):
        """Decode a numerically encoded k-mer to ACGT sequence"""
        encoding_map = {'0': 'A', '1': 'T', '2': 'C', '3': 'G'}
        return ''.join(encoding_map.get(c, 'N') for c in str(encoded_kmer))
    
    def _load_kmer_file(self, filepath):
        """Load a k-mer file into a pandas DataFrame"""
        # Check if file is gzipped
        is_gzipped = str(filepath).endswith('.gz')
        compression = 'gzip' if is_gzipped else None
        
        try:
            # Try to load with header
            df = pd.read_csv(filepath, sep='\t', compression=compression)
            
            # Check if the file actually has headers
            if 'kmer' not in df.columns and 'count' not in df.columns:
                # Try again without header
                df = pd.read_csv(filepath, sep='\t', header=None, 
                                names=['kmer', 'count'], compression=compression)
        except:
            # Fallback: load without header
            df = pd.read_csv(filepath, sep='\t', header=None, 
                            names=['kmer', 'count'], compression=compression)
        
        return df