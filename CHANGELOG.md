# Changelog
All notable changes to kmerml will be documented in this file.

## [Unreleased]
### Coming Soon
- Visualization for clustering results
- Command-line interface for ML pipelines
- Advanced phylogenetic analysis tools

## [0.3.0] - 2025-06-30
### Added
- ml: Complete machine learning pipeline implementation
- ml: KmerFeatureBuilderAgg for aggregated feature matrix building
- ml: Clustering module with hierarchical, k-means, DBSCAN, and GMM clustering
- ml: Dimensionality reduction with PCA, t-SNE, and UMAP
- ml: End-to-end pipeline for k-mer based clustering analysis
- utils: Genome metadata generator, to store general genome characteristics
- utils: Kmer metadata generator, to store a kmer analysis summary to the metadata file
- utils: Path utils, for repetitive path and file manipulation functions
- kmers: Generate, that exports in .txt or compressed all kmers for given k-values and their occurence total
- kmers: Statistics, calculates and exports in csv, per organism and per kmer, metrics of interest
- tests: Gerar_kmer, testing for kmer generation pipeline
- tests: Retrieve_metadata, testing for metadata fetching and/or calculation

### Fixed
- utils: Logger overwrite fixed
- setup: Updated dependencies to include all required ML packages
- versioning: Synchronized version numbers across all files

### Organized
- ml: Consolidated ML modules, removed deprecated ml_old and ml_old_two directories
- dependencies: Synchronized requirements.txt and setup.py

## [0.2.0] - 2025-06-18
### Added
- download: Allows genome download via NCBI API
- download: Acession list support for .txt and .csv/.tsv
- utils: Interactive progress bar
- utils: Email and Entrez tool validation

### Fixed
- download: Invalid access number check
- utils: Estimated time prediction fixed

## [0.1.0] - 2025-06-01
### Added
- Basic project structure
- Base project documentation

[Unreleased]: https://github.com/username/kmerml/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/username/kmerml/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/username/kmerml/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/username/kmerml/releases/tag/v0.1.0
