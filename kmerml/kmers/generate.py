import gzip
from collections import defaultdict
from pathlib import Path
from Bio import SeqIO
from kmerml.utils.path_utils import ensure_directory_exists
import time
import concurrent.futures

class KmerExtractor:
    """Extract k-mers from genomic sequences for multiple k values."""
    
    def __init__(self, output_dir="kmer_data", compress=True):
        """
        Initialize the KmerExtractor.
        
        Args:
            output_dir (str): Directory to save k-mer files
            compress (bool): Whether to compress output files using gzip
        """
        self.output_dir = ensure_directory_exists(Path(output_dir))
        self.compress = compress
    
    def extract_kmers_from_fasta(self, fasta_file, k_values, organism_id=None):
        """
        Extract k-mers directly from a FASTA file, respecting chromosome boundaries.
        
        Args:
            fasta_file (str): Path to the FASTA file
            k_values (list): List of k values to extract
            organism_id (str): Identifier for the organism (defaults to filename if None)
        """

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
        """Save k-mers with numeric encoding in organism-specific folders"""
        # Define encoding
        encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        
        # Create organism-specific directory
        organism_dir = self.output_dir / organism_id
        ensure_directory_exists(organism_dir)
        
        # Create filename within organism directory
        filename = f"k{k}.txt"
        if self.compress:
            filename += ".gz"
        
        filepath = organism_dir / filename
        
        open_func = gzip.open if self.compress else open
        mode = 'wt' if self.compress else 'w'
        
        with open_func(filepath, mode) as f:
            for kmer, count in kmers.items():
                # Convert k-mer to numeric representation
                numeric_kmer = ''.join(str(encoding.get(base, 'X')) for base in kmer)
                f.write(f"{numeric_kmer}\t{count}\n")
    
    def _process_single_genome(self, args):
        """Helper method for parallel processing of a single genome"""
        fasta_path, org_id, k_values = args
        try:
            self.extract_kmers_from_fasta(fasta_path, k_values, org_id)
            return (org_id, True, None)
        except Exception as e:
            return (org_id, False, str(e))
    
    def extract_from_genome_list(self, genome_paths, k_values, organism_ids=None, 
                                    n_processes=None):
        """
        Extract k-mers from multiple genome files for multiple k values in parallel.
        
        Args:
            genome_paths (list): List of paths to FASTA genome files
            k_values (list): List of k values to extract
            organism_ids (list, optional): List of organism identifiers.
                If None, will use filenames as identifiers.
            n_processes (int, optional): Number of parallel processes to use.
                If None, uses number of CPU cores.
                
        Returns:
            list: List of successfully processed organism IDs
        """
        if organism_ids is None:
            organism_ids = [Path(path).stem for path in genome_paths]
        
        # Validate input lists have matching lengths
        if len(organism_ids) != len(genome_paths):
            raise ValueError("Number of organism IDs must match number of genome paths")
        
        # Determine number of processes (default to number of CPU cores)
        if n_processes is None:
            n_processes = os.cpu_count()
        n_processes = max(1, min(n_processes, len(genome_paths)))
        
        print(f"Processing {len(genome_paths)} genomes using {n_processes} processes")
        
        # Prepare arguments for parallel processing
        process_args = [(genome_paths[i], organism_ids[i], k_values) 
                        for i in range(len(genome_paths))]
        
        processed_ids = []
        errors = []
        
        # Use ProcessPoolExecutor for parallel processing
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_genome = {executor.submit(self._process_single_genome, args): args[1] 
                                for args in process_args}
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_genome):
                completed += 1
                org_id, success, error = future.result()
                
                # Report progress
                elapsed = time.time() - start_time
                genomes_per_sec = completed / elapsed if elapsed > 0 else 0
                remaining = (len(genome_paths) - completed) / genomes_per_sec if genomes_per_sec > 0 else 0
                
                print(f"[{completed}/{len(genome_paths)}] Processed {org_id} - " 
                        f"{'Success' if success else 'Failed'} "
                        f"({genomes_per_sec:.2f} genomes/sec, ~{remaining:.1f}s remaining)")
                
                if success:
                    processed_ids.append(org_id)
                else:
                    errors.append((org_id, error))
        
        # Print summary
        print(f"\nCompleted processing {len(processed_ids)} out of {len(genome_paths)} genomes")
        if errors:
            print(f"Encountered {len(errors)} errors:")
            for org_id, error in errors:
                print(f"  - {org_id}: {error}")
        
        return processed_ids