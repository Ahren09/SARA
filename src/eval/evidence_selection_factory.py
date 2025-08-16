"""
Evidence Selection Factory for Unified Evidence Selection Interface.

This module provides a clean, modular interface for different evidence selection strategies
including embedding-based novelty and conditional self-information (CSI).
"""

import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from src.metrics.metrics_distance import cosine_distance_torch
from src.metrics.metrics_conditional_perplexity import ConditionalSelfInformation


class BaseEvidenceSelector(ABC):
    """Abstract base class for evidence selection strategies."""
    
    @abstractmethod
    def select_evidence(self, documents: List[str], question: str, k: int, 
                       device: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Select evidence using the implemented strategy.
        
        Args:
            documents: List of candidate documents
            question: Query question
            k: Number of evidence pieces to select
            device: Device for computation
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (selected_docs, remaining_docs)
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the selection method."""
        pass


class EmbeddingBasedSelector(BaseEvidenceSelector):
    """Evidence selection using embedding-based novelty."""
    
    def __init__(self, compressor):
        self.compressor = compressor
    
    @property
    def method_name(self) -> str:
        return "embedding"
    
    def select_evidence(self, documents: List[str], question: str, k: int, 
                       device: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Select evidence using embedding-based novelty.
        
        Formula: argmin ||v_q - Aggregate({Enc(v) | v ∈ V_sel ∪ {v_i}})||_2
        """
        if len(documents) <= k:
            selected = documents[:k]
            remaining = documents[k:]
            return selected, remaining
        
        # Get embeddings for documents and question
        all_texts = documents + [question]
        embeds = self.compressor.get_embeddings(all_texts)
        
        doc_embeds = embeds[:-1]
        question_embed = embeds[-1]
        
        # Start with first document
        selected_indices = torch.tensor([0], dtype=int, device=device)
        # Enrich query representation with top-1 retrieved context
        v_a = torch.mean(torch.stack([question_embed, doc_embeds[0]]), dim=0)
        
        # Select additional documents iteratively
        for _ in range(k - 1):
            selected_embeds = doc_embeds[selected_indices]
            mask = torch.ones(len(doc_embeds), dtype=torch.bool, device=device)
            mask[selected_indices] = False
            
            if not any(mask):
                break
            
            # Calculate average of selected embeddings
            avg_selected = torch.mean(selected_embeds, dim=0)
            # Adjust target by subtracting average of selected
            adjusted_target = v_a - avg_selected
            # Find document that minimizes distance to adjusted target
            distances = cosine_distance_torch(adjusted_target, doc_embeds, return_tensor=True)
            best_idx = torch.nonzero(mask).reshape(-1)[torch.argsort(distances[mask])[0]]
            selected_indices = torch.cat([selected_indices, best_idx.reshape(-1)])
        
        # Create result lists
        selected_indices = selected_indices.tolist()
        selected_docs = [documents[i] for i in selected_indices]
        remaining_docs = [documents[i] for i in range(len(documents)) if i not in selected_indices]
        
        return selected_docs, remaining_docs


class CSIBasedSelector(BaseEvidenceSelector):
    """Evidence selection using conditional self-information."""
    
    def __init__(self, csi_model: ConditionalSelfInformation):
        self.csi_model = csi_model
    
    @property
    def method_name(self) -> str:
        return "csi"
    
    def select_evidence(self, documents: List[str], question: str, k: int, 
                       device: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Select evidence using conditional self-information.
        
        Formula: argmax I(v_i|V_sel) where I(v_i|V_sel) = -log P(v_i|V_sel)
        """
        if len(documents) <= k:
            selected = documents[:k]
            remaining = documents[k:]
            return selected, remaining
        
        docs_copy = documents.copy()
        selected = [docs_copy.pop(0)]  # Start with first document
        
        # Iteratively select documents with highest CSI
        for _ in range(1, k):
            if not docs_copy:
                break
            
            # Compute conditional self-information for remaining documents
            csi_scores = self.csi_model.compute_conditional_self_information(
                docs_copy, "\n".join(selected)
            )
            
            # Select document with highest CSI (most novel information)
            best_idx = csi_scores.index(max(csi_scores))
            selected_doc = docs_copy.pop(best_idx)
            selected.append(selected_doc)
        
        return selected, docs_copy


class EvidenceSelectionFactory:
    """Factory for creating evidence selection strategies."""
    
    @staticmethod
    def create_selector(method: str, compressor=None, csi_model=None) -> BaseEvidenceSelector:
        """
        Create evidence selector based on method name.
        
        Args:
            method: Selection method ("embedding" or "csi")
            compressor: Compressor for embedding-based selection
            csi_model: CSI model for conditional self-information selection
            
        Returns:
            Configured evidence selector
        """
        if method == "embedding":
            if compressor is None:
                raise ValueError("Compressor required for embedding-based selection")
            return EmbeddingBasedSelector(compressor)
        
        elif method == "csi":
            if csi_model is None:
                raise ValueError("CSI model required for CSI-based selection")
            return CSIBasedSelector(csi_model)
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available selection methods."""
        return ["embedding", "csi"] 