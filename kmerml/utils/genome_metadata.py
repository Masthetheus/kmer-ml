import json
import os
from datetime import datetime
from pathlib import Path
from Bio import SeqIO
from kmerml.utils.path_utils import find_files, ensure_directory_exists

class GenomeMetadataManager:
    """Collect, store, and retrieve genome metadata across pipeline runs."""
    
    def __init__(self, metadata_file="data/metadata/genome_metadata.json"):

        self.metadata_file = Path(metadata_file)
        ensure_directory_exists(self.metadata_file.parent)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def _save_metadata(self):
        """Save current metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def collect_metadata(self, genome_dir, refresh=False, patterns=None):
        """Collect metadata for all genomes in the directory."""
        patterns = patterns or ["*.fa", "*.fasta", "*.fna"]
        genome_files = find_files(genome_dir, patterns=patterns, recursive=True)
        if not genome_files:
            print("No genome files found. Please check your data/raw/ directory.")
            return
        
        for genome_file in genome_files:
            genome_id = genome_file.stem
            
            # Skip if metadata exists and not refreshing
            if not refresh and genome_id in self.metadata:
                continue
            
            # Collect metadata
            genome_data = self._extract_genome_metadata(genome_file)
            self.metadata[genome_id] = genome_data
        
        # Save updated metadata
        self._save_metadata()
        return self.metadata
    
    def _extract_genome_metadata(self, genome_file):
        """Extract metadata from a genome file."""
        metadata = {
            "file_path": str(genome_file),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "contigs": 0,
            "total_size": 0,
            "gc_content": 0,
            "n_count": 0
        }
        
        total_gc = 0
        
        for record in SeqIO.parse(genome_file, "fasta"):
            metadata["contigs"] += 1
            sequence = str(record.seq).upper()
            seq_length = len(sequence)
            metadata["total_size"] += seq_length
            
            # Count GC and N content
            gc_count = sequence.count('G') + sequence.count('C')
            n_count = sequence.count('N')
            
            total_gc += gc_count
            metadata["n_count"] += n_count
            
        # Calculate GC percentage
        if metadata["total_size"] > 0:
            metadata["gc_content"] = (total_gc / metadata["total_size"]) * 100
        
        return metadata
    
    def get_genome_size(self, genome_id):
        """Get the size of a specific genome."""
        if genome_id in self.metadata:
            return self.metadata[genome_id]["total_size"]
        return None
    
    def list_available_genomes(self):
        """List all genomes with available metadata."""
        return list(self.metadata.keys())