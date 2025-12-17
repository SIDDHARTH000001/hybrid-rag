
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


# -------------------- LLM CONFIG --------------------

class LLMConfig(BaseModel):
    provider: str = "gcp"
    model_name: str = "gemini-2.5-flash"
    api_key: str
    endpoint: str = ""
    version: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048


# -------------------- EMBEDDINGS CONFIG --------------------

class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


# -------------------- HYBRID SEARCH CONFIG --------------------

class HybridSearchConfig(BaseModel):
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    top_k: int = 5

    def validate_weights(self):
        total = self.bm25_weight + self.semantic_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Hybrid search weights must sum to 1.0, got {total}"
            )


# -------------------- DOCUMENT PROCESSING --------------------

class DocumentProcessingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50


# -------------------- VECTOR STORE --------------------

class VectorStoreConfig(BaseModel):
    faiss_index_path: str = "./data/faiss_index"
    metadata_path: str = "./data/metadata.pkl"


# -------------------- SERVER --------------------

class ServerConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50


# -------------------- ROOT CONFIG --------------------

class Config(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    hybrid_search: HybridSearchConfig
    document_processing: DocumentProcessingConfig
    vector_store: VectorStoreConfig
    server: ServerConfig



class ConfigManager:

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Config:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with self.config_path.open("r") as f:
            config_dict = yaml.safe_load(f)

        return Config(**config_dict)

    def _validate_config(self):
        if not self.config.llm.api_key:
            raise ValueError(
                "LLM api_key must be provided in config.yaml"
            )

        self.config.hybrid_search.validate_weights()

        Path(self.config.server.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.vector_store.faiss_index_path).parent.mkdir(
            parents=True, exist_ok=True
        )

    def get_config(self) -> Config:
        return self.config



_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(config_path: str = "config.yaml") -> Config:
    return get_config_manager(config_path).get_config()
