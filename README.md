# kmer-ml

[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow)](https://github.com/Masthetheus/kmer-ml)
[![Version](https://img.shields.io/badge/version-0.3.0--alpha-orange)](https://github.com/Masthetheus/kmer-ml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Machine learning tool for analysis, comparison, and phylogenetic clustering based on k-mers from multiple species.

---

## Table of Contents

- [Description](#description)
- [Project Status](#project-status)
- [Installation](#installation)
- [Usage](#usage)
    - [1. Download genomes from NCBI via Entrez](#1-download-genomes-from-ncbi-via-entrez)
- [Example Usage](#example-usage)
    - [1. Genome download via API](#1-genome-download-via-api)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Description

The current project aims to apply machine learning in k-mer sequences distribution and composition analysis across multiple genomes, via a clusterization approach.

**Academic Context**

- The project was proposed during the Machine Learning for Bioinformatics subject, that represents three credits at the Master's course at PPGBCM - UFRGS.
- The objective itself is to apply machine learning methods to an biologic question of choice.
- For organization and easier access to the developed scripts and general logic, the kmer-ml repository was created.

**Biological Question**

- K-mer sequences represents an metric of interest in the genomics field as a whole.
- Given the existent relations found between k-mer composition and distribution accross species, and genome availability, one main question arose. What information, if any, of these sequences can led to a clustering method?

---

## Project Status

This project is being developed as part of a Master's course subject and is currently in **alpha stage**. 

**Current development focus:**
- ✅ Genome download from NCBI via Entrez API Script
- ✅ K-mer generation and processing
- 🔄 Machine learning pipeline
- ❌ Visualization tools
- ❌ Analysis tools
- ❌ Final panoram
  
Legend: ✅ Implemented | 🔄 In progress | ❌ Planned

---

## Installation

> **Note**: This is a pre-release version developed for a Master's course subject. While you're welcome to use and experiment with it, expect frequent changes.

### Option 1: Development installation
```bash
git clone https://github.com/Masthetheus/kmer-ml.git
cd kmer-ml
conda env create -f environment.yaml
conda activate kmerml
pip install -e .
```
---

## Usage

### 1. Download genomes from NCBI via Entrez

If the intended genomes for usage in the pipeline are already downloaded, in FASTA format and in all uppercase, this step can be skipped. 

If genome file integrity, mainly uppercase setting, isn't granted, it is strongly advised to gather the genomes using the provided script.

**Prerequisites**
- A valid email address for NCBI Entrez API usage.
- A registered Entrez tool, allowing correct usage of the API. For further tool registration, seek NCBI Entrez documentation.
- All dependencies installed (see [Installation](#installation)).

**Input:**
- A .csv, .tsv or .txt file containing at least one assembly accession number (e.g., `GCF_000146045.2`).
    - For `.csv` or `.tsv`: Use the same pattern as can be obtained from NCBI Download Table function. If not, make sure to specify the correct column name. 
    - For `.txt`: One accession number per line.
- If no --accession-list is provided, data/ncbi_dataset.tsv will be used as default.

**Command:**
```sh
python -m scripts.download_genomes --accession-list data/accession_list.txt
```

---

## Example Usage

### 1. Genome download via API

1. Create a file with an token accession number (e.g., `accessions.txt`):
    ```txt
    GCF_000146045.2 GCF_000002945.2
    ```
2. Run the download script:
```bash
python -m scripts.download_genomes
```
3. The genomes related to given access files will then be downloaded to data/raw/ and automatically processed (converted to all uppercase).
---

## Project Structure

```
kmer-ml/
    ├── CHANGELOG.md
    ├── LICENSE
    ├── README.md
    ├── config
    ├── data
    │   ├── processed
    │   └── raw
    ├── environment.yaml
    ├── kmerml
    │   ├── __init__.py
    │   ├── download
    │   │   ├── __init__.py
    │   │   └── ncbidownload.py
    │   └── utils
    │       ├── __init__.py
    │       ├── progress.py
    │       ├── validation.py
            └── logger.py
    ├── logs
    ├── requirements.txt
    ├── scripts
    │   ├── __init__.py
    │   ├── download_genomes.py
    │   └── generate_kmer_dataset.py
    ├── setup.py
    └── tests
        └── testes_desenvolvimento_api_ncbi.py
```

**Folder descriptions:**

- **data/**: Stores raw and processed data.
- **scripts/**: Utility and execution scripts.
- **kmerml/**: Main project source code (Python modules).
- **tests/**: Modular test of scripts.
- **config/**: Configuration files.
- **results/**: Results, figures, tables, and experiment outputs.
- **environment.yaml**: Conda environment file.
- **README.md**: This file.
- **.gitignore**: Files/folders ignored by git.
- **setup.py**: (Optional) Script to package the project as a Python library.

---

## Contributing

Contributions to kmerml are welcome! Here's how you can help:

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/Masthetheus/kmer-ml.git`
3. Set up development environment:
   ```bash
   conda env create -f environment.yaml
   conda activate kmerml
   pip install -e .
   ```
Contribution Guidelines:

- Bug Reports: Open an issue describing the bug and how to reproduce it
- Feature Requests: Open an issue describing the feature and its use case

We appreciate your help in improving kmerml!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows for reuse with few restrictions. You can use, modify, and distribute this code, even for commercial purposes, provided that you include the original copyright notice and disclaimer.

---

## Contact

- **Author**: Matheus Pedron Cassol
- **Email**: matheuspedroncassol@gmail.com
- **GitHub**: [@Masthetheus](https://github.com/Masthetheus)

This project is being developed as part of the Cellular and Molecular Biology Master's program (PPGBCM) at UFRGS (Universidade Federal do Rio Grande do Sul).
