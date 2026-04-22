"""Package setup for pdf-rag-pipeline."""

from setuptools import find_packages, setup

setup(
    name="pdf-rag-pipeline",
    version="0.1.0",
    description="Local PDF ingestion pipeline for RAG",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "PyMuPDF>=1.23.0",
        "pdfplumber>=0.10.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "PyYAML>=6.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "ocr": ["pytesseract>=0.3.10"],
        "embeddings": ["sentence-transformers>=2.2.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "pdf-pipeline=cli:cli",
        ],
    },
)
