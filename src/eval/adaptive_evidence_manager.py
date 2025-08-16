"""
Adaptive Evidence Manager for Retrieval and Context Compression Pipeline.

This module provides a clean, modular interface for managing evidence allocation
between natural language format and compressed token format.

Key concepts:
- natural_language_evidence: Evidence presented as readable text  
- compressed_token_evidence: Evidence encoded as compressed tokens (e.g., <COMPRESS>)
- k: Number of evidence pieces in natural language format
- n: Total number of evidence pieces (natural language + compressed tokens)
- n-k: Number of evidence pieces in compressed token format
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
from nltk.tokenize import sent_tokenize

from src.utils.utility import print_colored
from src.const import COMPRESS

# Define constants locally to avoid import issues
DELIMITER = "<|>"


class EvidenceFormatConfig:
    """Configuration for evidence formatting."""
    
    def __init__(self,
                 k: int = 5,  # Natural language evidence
                 n: int = 5,   # Total evidence (natural language + compressed)
                 compression_method: str = "sentence"):
        """
        Initialize evidence format configuration.
        
        Args:
            k: Number of evidence pieces in natural language format
            n: Total number of evidence pieces (natural language + compressed tokens)
            compression_method: How to compress evidence ("sentence" or "paragraph")
        """
        self.k = k
        self.n = n
        self.n_minus_k = n - k  # Compressed token evidence
        self.compression_method = compression_method
        
        # Validate configuration
        if k < 0 or n < 0:
            raise ValueError("k and n must be non-negative")
        if k > n:
            raise ValueError("k (natural language evidence) cannot exceed n (total evidence)")
    
    @property
    def num_natural_language(self) -> int:
        """Number of evidence pieces in natural language format."""
        return self.k
    
    @property
    def num_compressed_tokens(self) -> int:
        """Number of evidence pieces in compressed token format."""
        return self.n_minus_k
    
    @property
    def total_evidence_needed(self) -> int:
        """Total evidence needed."""
        return self.n
    
    def __repr__(self):
        return (f"EvidenceFormatConfig(k={self.k}, n={self.n}, n-k={self.n_minus_k}, "
                f"method={self.compression_method})")


class EvidenceAllocation:
    """Container for allocated evidence in different formats."""
    
    def __init__(self,
                 natural_language_evidence: List[str] = None,
                 compressed_token_evidence: List[str] = None,
                 compressed_prompt_text: str = "",
                 metadata: Dict[str, Any] = None):
        """
        Initialize evidence allocation.
        
        Args:
            natural_language_evidence: Evidence in natural language format
            compressed_token_evidence: Evidence for compression 
            compressed_prompt_text: Formatted prompt text for compressed evidence
            metadata: Additional metadata
        """
        self.natural_language_evidence = natural_language_evidence or []
        self.compressed_token_evidence = compressed_token_evidence or []
        self.compressed_prompt_text = compressed_prompt_text
        self.metadata = metadata or {}
    
    def get_natural_language_text(self) -> str:
        """Get natural language evidence as combined text."""
        return "\n\n".join(self.natural_language_evidence)
    
    def has_natural_language_evidence(self) -> bool:
        """Check if there is any natural language evidence."""
        return len(self.natural_language_evidence) > 0
    
    def has_compressed_evidence(self) -> bool:
        """Check if there is any compressed evidence."""
        return len(self.compressed_token_evidence) > 0
    
    def __repr__(self):
        return (f"EvidenceAllocation(natural_language={len(self.natural_language_evidence)}, "
                f"compressed_tokens={len(self.compressed_token_evidence)})")


class AdaptiveEvidenceManager:
    """
    Manager for adaptive retrieval and context compression pipeline.
    
    Handles allocation of evidence between natural language and compressed token formats.
    """
    
    def __init__(self, config: EvidenceFormatConfig):
        """
        Initialize the adaptive evidence manager.
        
        Args:
            config: Configuration for evidence formatting
        """
        self.config = config
    
    def allocate_evidence(self, 
                         retrieved_docs: List[str],
                         selected_evidence: List[str] = None,
                         model_name: str = "",
                         args = None) -> EvidenceAllocation:
        """
        Allocate retrieved evidence between natural language and compressed formats.
        
        Args:
            retrieved_docs: All retrieved documents
            selected_evidence: Pre-selected evidence (if evidence selection was used)
            model_name: Model name for format-specific handling
            args: Additional arguments object
            
        Returns:
            EvidenceAllocation with evidence split between formats
        """
        # Use selected evidence if available, otherwise use retrieved docs
        available_evidence = selected_evidence if selected_evidence else retrieved_docs
        
        # Ensure we don't exceed available evidence
        total_needed = min(self.config.total_evidence_needed, len(available_evidence))
        evidence_to_use = available_evidence[:total_needed]
        
        print_colored(f"Allocating {len(evidence_to_use)} evidence pieces: "
                    f"k={self.config.k} natural language + "
                     f"n-k={self.config.n_minus_k} compressed tokens", "blue")
        
        # Split evidence between formats
        natural_language_evidence = evidence_to_use[:self.config.k]
        compressed_token_evidence = evidence_to_use[self.config.k:self.config.n]
        
        # Generate compressed prompt text if needed
        compressed_prompt_text = ""
        if compressed_token_evidence:
            compressed_prompt_text = self._generate_compressed_prompt(
                compressed_token_evidence, model_name, args
            )
        
        return EvidenceAllocation(
            natural_language_evidence=natural_language_evidence,
            compressed_token_evidence=compressed_token_evidence,
            compressed_prompt_text=compressed_prompt_text,
            metadata={
                "k": self.config.k,
                "n": self.config.n,
                "n_minus_k": self.config.n_minus_k,
                "total_allocated": len(evidence_to_use)
            }
        )
    
    def _generate_compressed_prompt(self, 
                                   compressed_evidence: List[str], 
                                   model_name: str,
                                   args = None) -> str:
        """
        Generate compressed prompt text for compressed evidence.
        
        Args:
            compressed_evidence: Evidence to be compressed
            model_name: Model name for format-specific handling
            args: Additional arguments
            
        Returns:
            Formatted compressed prompt text
        """
        if not compressed_evidence:
            return ""
        
        # Use compression method from config
        if self.config.compression_method == "sentence":
            # Compress each evidence piece at sentence level
            compressed_texts = []
            for evidence in compressed_evidence:
                sentences = sent_tokenize(evidence)
                compressed_text = f"{COMPRESS} {' '.join(sentences)}"
                compressed_texts.append(compressed_text)
        else:
            # Compress at paragraph level
            compressed_texts = [f"{COMPRESS} {evidence}" for evidence in compressed_evidence]
        
        # Join compressed texts with delimiter
        return f" {DELIMITER} ".join(compressed_texts)
    
    def create_qa_inputs(self, 
                        allocation: EvidenceAllocation,
                        question: str,
                        choices: List[str] = None,
                        answer_format: str = "short_answer") -> Dict[str, Any]:
        """
        Create inputs for QA model from evidence allocation.
        
        Args:
            allocation: Evidence allocation
            question: Question to answer
            choices: Multiple choice options
            answer_format: Answer format ("short_answer" or "multiple_choice")
            
        Returns:
            Dictionary with QA model inputs
        """
        # Prepare natural language context
        context = allocation.get_natural_language_text()
        
        # Prepare additional context (compressed evidence)
        additional_context = allocation.compressed_token_evidence
        
        # Prepare compressed prompt text
        compressed_prompt_text = allocation.compressed_prompt_text
        
        return {
            "context": context,
            "additional_context": additional_context,
            "compressed_prompt_text": compressed_prompt_text,
            "question": question,
            "choices": choices,
            "answer_format": answer_format
        }


def create_evidence_manager_from_args(args) -> AdaptiveEvidenceManager:
    """
    Create evidence manager from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured evidence manager
    """
    # Extract parameters with new naming convention
    k = getattr(args, 'k', 5)  # Natural language evidence
    n = getattr(args, 'n', 5)  # Total evidence
    
    # Override if explicit n is provided
    if hasattr(args, 'total_evidence') and args.total_evidence is not None:
        n = args.total_evidence
    
    compression_method = getattr(args, 'compressor_encode_method', 'sentence')
    
    config = EvidenceFormatConfig(
        k=k,
        n=n,
        compression_method=compression_method
    )
    
    return AdaptiveEvidenceManager(config)