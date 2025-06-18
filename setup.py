from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("kmerml/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    # Basic package information
    name="kmerml",
    version=version,
    author="Matheus Pedron Cassol",
    author_email="matheuspedroncassol@gmail.com",
    description="K-mer based machine learning for genomic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/kmer-ml",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],

    python_requires=">=3.8",
    
    packages = find_packages() + ['scripts'],
    
    # Dependencies
    install_requires=[
        "biopython>=1.85",
        "requests>=2.32",
        "beautifulsoup4",
        "tqdm",
        "numpy",
        "pandas",
    ],

    # Command line scripts
    entry_points={
        "console_scripts": [
            "kmerml-download=scripts.download_genomes:main",
        ],
    },

)