"""
Compressor models for evidence compression and selection.
"""

import torch
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer

from .evidence_selection_factory import EvidenceSelectionFactory


class BaseCompressor(ABC):
    """Abstract base class for compressor models."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Get embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_embed_dim(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def get_embed_length(self) -> int:
        """Get embedding length."""
        pass
    
    @property
    @abstractmethod
    def tokenizer(self):
        """Get tokenizer."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass


class SFRCompressor(BaseCompressor):
    """SFR (Salesforce) compressor implementation."""
    
    def __init__(self, model_name_or_path: str, device: Union[torch.device, str], dtype: torch.dtype = torch.float32):
        self.model_name_str = model_name_or_path
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.model.half() # bfloat16
        # self.model.float() # float32
    
    @property
    def model_name(self) -> str:
        return self.model_name_str
    
    @property
    def tokenizer(self):
        return self.model.tokenizer
    
    def get_embed_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def get_embed_length(self) -> int:
        return self.model.max_seq_length
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Get embeddings for texts using SFR model."""
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)


class SentenceBERTCompressor(BaseCompressor):
    """SentenceBERT compressor implementation."""
    
    def __init__(self, model_name_or_path: str, device: Union[torch.device, str], dtype: torch.dtype = torch.float32):
        self.model_name_str = model_name_or_path
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device, dtype=dtype)
    
    @property
    def model_name(self) -> str:
        return self.model_name_str
    
    @property
    def tokenizer(self):
        return self.model.tokenizer
    
    def get_embed_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def get_embed_length(self) -> int:
        return self.model.max_seq_length
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Get embeddings for texts using SentenceBERT model."""
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)


class CompressorFactory:
    """Factory for creating compressor models."""
    
    @staticmethod
    def create_compressor(model_name_or_path: str, device: Union[torch.device, str], dtype: torch.dtype = torch.float32) -> BaseCompressor:
        """Create appropriate compressor based on model name."""
        
        if model_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
            return SFRCompressor(model_name_or_path, device, dtype)
        else:
            return SentenceBERTCompressor(model_name_or_path, device, dtype)


class EvidenceSelector:
    """Unified evidence selection interface using factory pattern."""
    
    def __init__(self, compressor: BaseCompressor, selection_method: str = None):
        self.compressor = compressor
        self.selection_method = selection_method
        self.selector = None
        
        # Initialize selector if method is specified
        if selection_method:
            self._initialize_selector()
    
    def _initialize_selector(self):
        """Initialize the evidence selector based on method."""
        if self.selection_method == "embedding":
            self.selector = EvidenceSelectionFactory.create_selector(
                method="embedding", 
                compressor=self.compressor
            )
        elif self.selection_method == "self-info":
            # CSI model will be provided during selection
            self.selector = None
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def select_evidence(self, documents: List[str], question: str, k: int, 
                       device: str, csi_model=None) -> Tuple[List[str], List[str]]:
        """
        Select evidence using the specified method.
        
        Returns:
            Tuple of (selected_docs, remaining_docs)
        """
        if not self.selection_method or len(documents) <= k:
            # No selection needed or not enough documents
            selected = documents[:k]
            remaining = documents[k:]
            return selected, remaining
        
        # Use factory-based selector
        if self.selection_method == "embedding":
            return self.selector.select_evidence(documents, question, k, device)
        elif self.selection_method == "self-info":
            # Create CSI selector on-demand
            if csi_model is None:
                raise ValueError("CSI model required for self-info selection")
            csi_selector = EvidenceSelectionFactory.create_selector(
                method="csi", 
                csi_model=csi_model
            )
            return csi_selector.select_evidence(documents, question, k, device)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")