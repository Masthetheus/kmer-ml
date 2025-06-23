import os
import gzip
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import math

class KmerStatistics:
    """Calculate statistics from k-mer files with numeric encoding (0,1,2,3 for A,T,C,G)"""
    
    def __init__(self, input_dir="data/processed"):
        """Initialize with directory containing k-mer files"""
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} not found")
        
        # Decoding map for display purposes
        self.decoding = {
            '0': 'A', '1': 'T', '2': 'C', '3': 'G', 'X': 'N'
        }
    
    def read_kmer_file(self, file_path):
        """Read a k-mer file and return dictionary of k-mers and counts"""
        kmers = {}
        is_compressed = str(file_path).endswith('.gz')
        open_func = gzip.open if is_compressed else open
        mode = 'rt' if is_compressed else 'r'
        
        with open_func(file_path, mode) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    numeric_kmer, count = parts
                    kmers[numeric_kmer] = int(count)
        
        return kmers
    
    def decode_kmer(self, numeric_kmer):
        """Convert numeric k-mer back to DNA sequence"""
        return ''.join(self.decoding.get(base, 'N') for base in numeric_kmer)
    
    def get_file_stats(self, file_path):
        """Calculate statistics for a single k-mer file"""
        kmers = self.read_kmer_file(file_path)
        if not kmers:
            return None
        
        # Extract k value from the first k-mer
        k = len(next(iter(kmers.keys())))
        
        # Basic statistics
        total_kmers = sum(kmers.values())
        unique_kmers = len(kmers)
        
        # Calculate GC content (from numeric encoding)
        gc_count = 0
        total_bases = 0
        
        for numeric_kmer, count in kmers.items():
            total_bases += len(numeric_kmer) * count
            # '2' is C and '3' is G in our encoding
            gc_count += (numeric_kmer.count('2') + numeric_kmer.count('3')) * count
        
        gc_content = gc_count / total_bases if total_bases > 0 else 0
        
        # K-mer diversity
        possible_kmers = 4**k
        diversity = unique_kmers / possible_kmers
        
        # Most common k-mers (top 5)
        most_common = sorted(kmers.items(), key=lambda x: x[1], reverse=True)[:5]
        most_common_decoded = [(self.decode_kmer(kmer), count) for kmer, count in most_common]
        
        # Calculate Shannon entropy
        probabilities = [count/total_kmers for count in kmers.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        # Frequency spectrum (count of singletons, doubletons, etc.)
        freq_spectrum = Counter(kmers.values())
        singletons = freq_spectrum.get(1, 0)  # k-mers that appear exactly once
        
        # Calculate evenness (normalized entropy)
        max_entropy = math.log2(unique_kmers) if unique_kmers > 0 else 0
        evenness = entropy / max_entropy if max_entropy > 0 else 0
        
        # Create results dictionary
        stats = {
            'file_name': file_path.name,
            'k': k,
            'total_kmers': total_kmers,
            'unique_kmers': unique_kmers,
            'gc_content': gc_content,
            'diversity': diversity,
            'entropy': entropy,
            'evenness': evenness,
            'singletons': singletons,
            'most_common_kmers': ';'.join([f"{kmer}({count})" for kmer, count in most_common_decoded])
        }
        
        return stats
    
    def process_directory(self, pattern="*_k*.txt*"):
        """Process all k-mer files in directory matching pattern"""
        results = []
        
        # Find all k-mer files
        kmer_files = list(self.input_dir.glob(pattern))
        
        if not kmer_files:
            print(f"No k-mer files found matching pattern {pattern}")
            return pd.DataFrame()
        
        print(f"Processing {len(kmer_files)} k-mer files...")
        
        # Process each file
        for file_path in kmer_files:
            print(f"Analyzing {file_path.name}...")
            stats = self.get_file_stats(file_path)
            if stats:
                results.append(stats)
        
        # Convert to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def save_stats_to_csv(self, output_file="kmer_statistics.csv", pattern="*_k*.txt*"):
        """Process k-mer files and save statistics to CSV"""
        df = self.process_directory(pattern)
        
        if df.empty:
            print("No results to save.")
            return False
        
        # Save to CSV
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        print(f"Statistics saved to {output_path}")
        return True

