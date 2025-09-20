"""
Configuration settings for the RAG Pipeline
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline settings"""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    
    # Pinecone Settings
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-pdf-index")
    embedding_dimension: int = 1536  # OpenAI text-embedding-3-small dimension
    
    # Text Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval Settings
    retrieval_k: int = 4  # Number of documents to retrieve
    
    # LLM Settings
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.0
    
    # OCR Settings
    ocr_languages: List[str] = None
    
    # File Settings
    temp_dir: str = "temp_pdfs"
    max_file_size_mb: int = 50
    
    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ['en']
    
    def validate(self):
        """Validate configuration settings"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")


# Global configuration instance
config = RAGConfig()