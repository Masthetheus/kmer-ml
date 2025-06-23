import os
import gzip
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class KmerExtractor:
    """
    Extract k-mers from genomic sequences for multiple k values.
    Optimized for processing multiple organisms in parallel.
    """
    
    def __init__(self, output_dir="kmer_data", compress=True):
        """
        Initialize the KmerExtractor.
        
        Args:
            output_dir (str): Directory to save k-mer files
            compress (bool): Whether to compress output files using gzip
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compress = compress
    
    def extract_kmers_from_fasta(self, fasta_file, k_values, organism_id=None):
        """
        Extract k-mers directly from a FASTA file, respecting chromosome boundaries.
        
        Args:
            fasta_file (str): Path to the FASTA file
            k_values (list): List of k values to extract
            organism_id (str): Identifier for the organism (defaults to filename if None)
        """
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("BioPython is required. Install with: pip install biopython")
        
        # Extract organism ID from filename if not provided
        if organism_id is None:
            organism_id = Path(fasta_file).stem
        
        # Initialize k-mer dictionaries for each k value
        all_kmers = {k: defaultdict(int) for k in k_values}
        
        # Process each chromosome/contig separately
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Get the sequence as a string
            sequence = str(record.seq).upper()
            
            # Skip sequences shorter than the largest k value
            if len(sequence) < max(k_values):
                print(f"Skipping {record.id}: too short for k-mer extraction")
                continue
            
            # Extract k-mers for each k value from this chromosome
            for k in k_values:
                # Extract k-mers using sliding window
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i:i+k]
                    
                    # Skip k-mers with non-ACGT characters (optional)
                    if not all(base in "ACGT" for base in kmer):
                        continue
                        
                    all_kmers[k][kmer] += 1
            
            print(f"Processed chromosome/contig: {record.id}")
        
        # Save k-mers to files
        for k, kmers in all_kmers.items():
            self._save_kmers_to_file(kmers, organism_id, k)
        
        return organism_id
    
    def _save_kmers_to_file(self, kmers, organism_id, k):
        """Save k-mers with numeric encoding"""
        # Define encoding
        encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        
        filename = f"{organism_id}_k{k}.txt"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        
        open_func = gzip.open if self.compress else open
        mode = 'wt' if self.compress else 'w'
        
        with open_func(filepath, mode) as f:
            for kmer, count in kmers.items():
                # Convert k-mer to numeric representation
                numeric_kmer = ''.join(str(encoding.get(base, 'X')) for base in kmer)
                f.write(f"{numeric_kmer}\t{count}\n")
    
    def batch_extract(self, sequences, k_values, batch_size=10):
        """
        Process organisms in batches to manage memory.
        
        Args:
            sequences (list): List of (organism_id, sequence) tuples
            k_values (list): List of k values to extract
            batch_size (int): Number of organisms to process in each batch
        """
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            for organism_id, sequence in batch:
                self.extract_kmers_to_file(sequence, organism_id, k_values)
                print(f"Processed {organism_id}")
    
    def parallel_extract(self, sequences, k_values, max_workers=None):
        """
        Process organisms in parallel using multiple CPU cores.
        This is much faster for multiple organisms.
        
        Args:
            sequences (list): List of (organism_id, sequence) tuples
            k_values (list): List of k values to extract
            max_workers (int): Maximum number of parallel workers
        """
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
            
        print(f"Using {max_workers} CPU cores for parallel processing")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs - one job per organism
            futures = []
            for organism_id, sequence in sequences:
                future = executor.submit(
                    self._process_single_organism, 
                    sequence, 
                    organism_id, 
                    k_values
                )
                futures.append((future, organism_id))
            
            # Process results as they complete
            for future, organism_id in futures:
                try:
                    future.result()  # This will raise any exceptions that occurred
                    print(f"Completed processing {organism_id}")
                except Exception as e:
                    print(f"Error processing {organism_id}: {str(e)}")
    
    def _process_single_organism(self, sequence, organism_id, k_values):
        """
        Process a single organism (for parallel execution).
        
        Args:
            sequence (str): The DNA/RNA sequence
            organism_id (str): Identifier for the organism
            k_values (list): List of k values to extract
            
        Returns:
            str: The organism_id that was processed
        """
        self.extract_kmers_to_file(sequence, organism_id, k_values)
        return organism_id