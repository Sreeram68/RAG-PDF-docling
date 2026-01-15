"""
Embedding Generator Module
Uses BGE-M3 model for generating dense, sparse, and multi-vector embeddings.
BGE-M3 is multilingual and excels at handling long documents.
"""

import os
import ssl
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Disable SSL verification for corporate networks without admin access
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Monkey-patch SSL to bypass certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import numpy as np
from FlagEmbedding import BGEM3FlagModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentEmbedding:
    """Represents embeddings for a document chunk."""
    chunk_id: str
    text: str
    dense_embedding: np.ndarray
    sparse_embedding: Optional[dict] = None  # Token weights for sparse retrieval
    colbert_embedding: Optional[np.ndarray] = None  # Multi-vector for ColBERT-style retrieval


def _patch_requests_ssl():
    """Patch requests library to disable SSL verification."""
    import requests
    from requests.adapters import HTTPAdapter
    
    old_init = HTTPAdapter.__init__
    def new_init(self, *args, **kwargs):
        kwargs.setdefault('pool_connections', 10)
        kwargs.setdefault('pool_maxsize', 10)
        old_init(self, *args, **kwargs)
    HTTPAdapter.__init__ = new_init
    
    old_send = HTTPAdapter.send
    def new_send(self, request, *args, **kwargs):
        kwargs['verify'] = False
        return old_send(self, request, *args, **kwargs)
    HTTPAdapter.send = new_send

_patch_requests_ssl()


def _get_best_device() -> str:
    """Detect and return the best available device (CUDA GPU or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {device_name}")
            return "cuda"
        else:
            logger.info("No CUDA GPU available, using CPU")
            return "cpu"
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}, falling back to CPU")
        return "cpu"


class EmbeddingGenerator:
    """
    Generates embeddings using BGE-M3 model.
    Supports dense, sparse, and ColBERT-style multi-vector embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
        max_length: int = 8192
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name for BGE-M3
            use_fp16: Use half precision for faster inference
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length (BGE-M3 supports up to 8192)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect best device if not specified
        if device is None:
            device = _get_best_device()
        
        self.device = device
        logger.info(f"Loading BGE-M3 model: {model_name} on device: {device}")
        
        # Initialize BGE-M3 model
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 if device == "cuda" else False,  # FP16 only works well on GPU
            device=device
        )
        
        logger.info("BGE-M3 model loaded successfully")
    
    def generate_embeddings(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False
    ) -> list[DocumentEmbedding]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text chunks to embed
            return_dense: Return dense embeddings
            return_sparse: Return sparse embeddings (for hybrid search)
            return_colbert: Return ColBERT multi-vector embeddings
            
        Returns:
            List of DocumentEmbedding objects
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        
        # Generate embeddings using BGE-M3
        embeddings = self.model.encode(
            texts,
            batch_size=12,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )
        
        # Package results
        results = []
        dense_vecs = embeddings.get('dense_vecs', [])
        lexical_weights = embeddings.get('lexical_weights', [])
        colbert_vecs = embeddings.get('colbert_vecs', [])
        
        for i, text in enumerate(texts):
            doc_embedding = DocumentEmbedding(
                chunk_id=f"chunk_{i}",
                text=text,
                dense_embedding=dense_vecs[i] if return_dense and i < len(dense_vecs) else None,  # type: ignore
                sparse_embedding=dict(lexical_weights[i]) if return_sparse and i < len(lexical_weights) else None,  # type: ignore
                colbert_embedding=colbert_vecs[i] if return_colbert and i < len(colbert_vecs) else None  # type: ignore
            )
            results.append(doc_embedding)
        
        logger.info(f"Generated embeddings: dense={return_dense}, sparse={return_sparse}, colbert={return_colbert}")
        
        return results
    
    def generate_query_embedding(
        self,
        query: str,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False
    ) -> dict:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            return_dense: Return dense embedding
            return_sparse: Return sparse embedding
            return_colbert: Return ColBERT multi-vector embedding
            
        Returns:
            Dictionary with query embeddings
        """
        embeddings = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )
        
        return {
            'dense': embeddings['dense_vecs'][0] if return_dense else None,
            'sparse': embeddings['lexical_weights'][0] if return_sparse else None,
            'colbert': embeddings['colbert_vecs'][0] if return_colbert else None
        }
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of dense embeddings (1024 for BGE-M3)."""
        return 1024


if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()
    
    sample_texts = [
        "The financial report shows quarterly revenue growth of 15%.",
        "Machine learning models require large datasets for training.",
        "Climate change impacts global agricultural production."
    ]
    
    embeddings = generator.generate_embeddings(sample_texts)
    
    for emb in embeddings:
        print(f"{emb.chunk_id}: dense shape = {emb.dense_embedding.shape}")
