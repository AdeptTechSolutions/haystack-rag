from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DocumentProcessingConfig:
    split_length: int = 150
    split_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cache_dir: Path = Path("./.document_cache")
    tracking_dir: Path = Path("./.tracking")
    top_k: int = 5


@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    temp_dir: Path = Path("temp")
    resources_dir: Path = Path("resources")
