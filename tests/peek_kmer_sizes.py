#!/usr/bin/env python3
import gzip
import os
from pathlib import Path

def check_kmer_sizes(base_dir):
    """Check if all k-mers in files have the correct length"""
    base_path = Path(base_dir)
    
    # Find all k-mer files
    for k in range(8, 13):  # k=8 to k=12
        pattern = f"**/k{k}.txt*"
        files = list(base_path.glob(pattern))
        
        if not files:
            print(f"No files found for k={k}")
            continue
            
        sample_file = files[0]
        print(f"\nChecking k={k} file: {sample_file}")
        
        # Open the file (handling gzip if needed)
        is_gzipped = str(sample_file).endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        mode = 'rt' if is_gzipped else 'r'
        
        try:
            with open_func(sample_file, mode) as f:
                total = 0
                incorrect = 0
                
                for line_num, line in enumerate(f, 1):
                    if line_num > 1000:  # Just check the first 1000 lines
                        break
                        
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                        
                    kmer = parts[0]
                    total += 1
                    
                    # Check if numerically encoded
                    if all(c in '0123' for c in kmer):
                        # Each digit represents one base
                        if len(kmer) != k:
                            incorrect += 1
                            print(f"  Line {line_num}: {kmer} has length {len(kmer)}, expected {k}")
                    else:
                        # Direct sequence
                        if len(kmer) != k:
                            incorrect += 1
                            print(f"  Line {line_num}: {kmer} has length {len(kmer)}, expected {k}")
                
                if total == 0:
                    print("  File appears to be empty")
                elif incorrect == 0:
                    print(f"  ✓ All {total} k-mers checked have correct length {k}")
                else:
                    print(f"  ✗ Found {incorrect}/{total} k-mers with incorrect length")
                    
        except Exception as e:
            print(f"  Error reading file: {e}")

if __name__ == "__main__":
    # Adjust this to your k-mer data directory
    kmer_dir = "data/processed/kmers/"
    check_kmer_sizes(kmer_dir)