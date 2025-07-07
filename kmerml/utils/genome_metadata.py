import json
from datetime import datetime
from pathlib import Path
from Bio import SeqIO
from Bio import Entrez
from kmerml.utils.path_utils import find_files, ensure_directory_exists

class GenomeMetadataManager:
    """Collect, store, and retrieve genome metadata across pipeline runs, including NCBI metadata."""
    
    def __init__(self, metadata_file="data/metadata/genome_metadata.json", email="your_email@example.com"):
        self.metadata_file = Path(metadata_file)
        ensure_directory_exists(self.metadata_file.parent)
        self.metadata = self._load_metadata()
        Entrez.email = email  # Set your email for NCBI Entrez

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
    
    def collect_metadata(self, genome_dir, refresh=False, patterns=None, fetch_ncbi=False):
        """Collect metadata for all genomes in the directory, optionally fetching NCBI metadata."""
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
            
            # Collect local metadata
            genome_data = self._extract_genome_metadata(genome_file)
            
            # Optionally fetch NCBI metadata if genome_id looks like an accession
            if fetch_ncbi and (genome_id.startswith("GCF_") or genome_id.startswith("GCA_")):
                ncbi_data = self._fetch_ncbi_metadata(genome_id)
                if ncbi_data:
                    genome_data.update(ncbi_data)
            
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
            gc_count = sequence.count('G') + sequence.count('C')
            n_count = sequence.count('N')
            total_gc += gc_count
            metadata["n_count"] += n_count
        if metadata["total_size"] > 0:
            metadata["gc_content"] = (total_gc / metadata["total_size"]) * 100
        return metadata

    def _fetch_ncbi_metadata(self, accession):
        """Fetch organism name, taxid, and genome length from NCBI for a given accession."""
        try:
            handle = Entrez.esearch(db="assembly", term=accession, retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            if not record["IdList"]:
                return None
            assembly_id = record["IdList"][0]
            handle = Entrez.esummary(db="assembly", id=assembly_id, retmode="xml")
            summary = Entrez.read(handle)
            handle.close()
            docsum = summary['DocumentSummarySet']['DocumentSummary'][0]
            organism = docsum.get('Organism', None)
            taxid = docsum.get('Taxid', None)
            try:
                genome_length = int(docsum.get('AssemblySpan', 0))
            except Exception:
                genome_length = None
            return {
                "ncbi_organism": organism,
                "ncbi_taxid": taxid,
                "ncbi_genome_length": genome_length
            }
        except Exception as e:
            print(f"NCBI metadata fetch failed for {accession}: {e}")
            return None

    def get_genome_size(self, genome_id):
        """Get the size of a specific genome."""
        if genome_id in self.metadata:
            return self.metadata[genome_id].get("total_size")
        return None
    
    def list_available_genomes(self):
        """List all genomes with available metadata."""
        return list(self.metadata.keys())