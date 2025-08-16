"""
Unified QA model interface for the SARA evaluation pipeline (Mistral-only public release).
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils.eval_utils import load_eval_model, load_hf_pipeline
from src.prompts import PROMPT_HF_MODELS, PROMPT_HF_MODELS_QUESTION_ONLY, ANSWER_INSTRUCTION, ANSWER_INSTRUCTION_MULTI_CHOICE, MISTRAL_CHAT_TEMPLATE


class BaseQAModel(ABC):
    """Abstract base class for all QA models."""
    
    @abstractmethod
    def generate_answer(self, question: str, context: str = None, choices: List[str] = None, 
                       additional_context: List[str] = None, retrieval_kwargs: Dict = None, 
                       **kwargs) -> str:
        """Generate answer for a given question and context."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class HuggingFaceQAModel(BaseQAModel):
    """Unified HuggingFace model interface (Llama, Mistral, etc.)."""
    
    def __init__(self, model_name_or_path: str, quantization: str = None, device: str = "cuda", 
                 max_new_tokens: int = 64):
        self.model_name_str = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Load model based on whether it's a checkpoint or pipeline model
        if ("checkpoint" in model_name_or_path or
            any(keyword in model_name_or_path.lower() for keyword in ["mistral", "llama", "gemma"])):
            self.model, self.tokenizer = load_eval_model(model_name_or_path, quantization)
            self.is_pipeline = False
            
            # Set chat template for Mistral if needed
            if self.tokenizer.chat_template is None and "mistral" in model_name_or_path.lower():
                self.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        else:
            self.pipeline = load_hf_pipeline(model_name_or_path, quantization=quantization,
                                           max_new_tokens=max_new_tokens, device=device)
            self.is_pipeline = True
    
    @property
    def model_name(self) -> str:
        return self.model_name_str

    def generate_answers_batched(self, prompts, retrieval_embeds=None, per_example_counts=None,
                                 repetition_penalty: float = 1.0):
        """Single source of truth for batched generation (release plan §2/§9).

        ``prompts``: list of fully-formatted prompt strings (chat template applied by caller is OK; if a
        raw string is passed we apply the chat template here). ``retrieval_embeds``: flat
        ``(total_compress, retriever_hidden_size)`` tensor ordered example-major, document-order (SARA);
        ``None`` for Standard RAG. ``per_example_counts``: per-prompt ``<COMPRESS>`` counts (SARA only).

        Returns a list of decoded answer strings (only the newly generated tokens; see
        ``XMistralForCausalLM.generate`` / ``extract_generated_text``).
        """
        import torch
        from src.model.xMistral import extract_generated_text

        tok = self.tokenizer
        # Apply chat template once, batched. Callers pass the user message text; we wrap it.
        rendered = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True)
            for p in prompts
        ]
        enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
            repetition_penalty=repetition_penalty, pad_token_id=tok.pad_token_id,
        )
        if retrieval_embeds is not None:
            embeds = retrieval_embeds.to(device=self.model.device)
            out = self.model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                retrieval_embeds=embeds, per_example_counts=per_example_counts, **gen_kwargs,
            )
        else:
            # Standard RAG: route through the same inputs_embeds path so the returned tensor is only-new.
            out = self.model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs,
            )
        texts = extract_generated_text(tok, out)
        return [self._clean_response(t) for t in texts], enc["input_ids"], enc["attention_mask"]

    def generate_answer(self, question: str, context: str = None, choices: List[str] = None,
                       additional_context: List[str] = None, retrieval_kwargs: Dict = None,
                       length_penalty: float = 1.0, repetition_penalty: float = 1.0, **kwargs) -> str:
        """Generate answer using HuggingFace model."""
        
        if self.is_pipeline:
            return self._generate_with_pipeline(question, context, choices, additional_context)
        else:
            return self._generate_with_model(question, context, choices, additional_context, 
                                           retrieval_kwargs, length_penalty, repetition_penalty)
    
    def _generate_with_pipeline(self, question: str, context: str = None, choices: List[str] = None,
                               additional_context: List[str] = None) -> str:
        """Generate using HuggingFace pipeline."""
        if context is None:
            prompt = PROMPT_HF_MODELS_QUESTION_ONLY.format(question=question, answer_instruction=ANSWER_INSTRUCTION)
        else:
            if choices:
                question_with_choices = f"{question}\n"
                for i, choice in enumerate(choices):
                    question_with_choices += f"{'ABCD'[i]}. {choice}\n"
                prompt = PROMPT_HF_MODELS.format(question=question_with_choices, context=context,
                                               answer_instruction=ANSWER_INSTRUCTION_MULTI_CHOICE)
            else:
                prompt = PROMPT_HF_MODELS.format(question=question, context=context,
                                               answer_instruction=ANSWER_INSTRUCTION)
        
        response = self.pipeline(prompt)[0]['generated_text']
        if prompt in response:
            response = response.split(prompt)[1].strip()
        
        return self._clean_response(response)
    
    def _generate_with_model(self, question: str, context: str = None, choices: List[str] = None,
                            additional_context: List[str] = None, retrieval_kwargs: Dict = None,
                            length_penalty: float = 1.0, repetition_penalty: float = 1.0) -> str:
        """Generate using loaded model directly."""
        
        # Create prompt based on context availability and question type
        if context is None:
            prompt = PROMPT_HF_MODELS_QUESTION_ONLY.format(question=question, answer_instruction=ANSWER_INSTRUCTION)
        else:
            # Add compression tokens to context if additional_context is provided
            if additional_context and hasattr(self.model, 'compress_token_id'):
                # Add compression tokens to the context
                compress_token_text = " " + " ".join(["<COMPRESS>"] * len(additional_context))
                context_with_compress = context + "\n ## Additional context (compression tokens):\n" + compress_token_text
            else:
                context_with_compress = context
            
            if choices:
                question_with_choices = f"{question}\n"
                for i, choice in enumerate(choices):
                    question_with_choices += f"{'ABCD'[i]}. {choice}\n"
                prompt = PROMPT_HF_MODELS.format(question=question_with_choices, context=context_with_compress,
                                               answer_instruction=ANSWER_INSTRUCTION_MULTI_CHOICE)
            else:
                prompt = PROMPT_HF_MODELS.format(question=question, context=context_with_compress,
                                               answer_instruction=ANSWER_INSTRUCTION)
        
        # Tokenize input
        if "gemma" in self.model_name_str.lower():
            inputs = self.tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt").to(self.model.device)
        else:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(inputs, add_special_tokens=False, padding=True, return_tensors="pt").to(self.model.device)
        
        # Set up retrieval kwargs if provided
        if retrieval_kwargs is None:
            retrieval_kwargs = {}
        
        # Generate response
        answer_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            min_length=50,
            length_penalty=length_penalty,
            num_beams=5,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            repetition_penalty=repetition_penalty,
            **retrieval_kwargs
        )
        
        response = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        
        # Extract the generated part
        if isinstance(prompt, str) and prompt in response:
            response = response.split(prompt)[1].strip()
        
        return self._clean_response(response)
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove common artifacts
        if "---Answer---" in response:
            response = response.split("---Answer---")[1].strip()
        if "---Response---" in response:
            response = response.split("---Response---")[1].strip()
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()
        if "ssistant\n" in response.lower():
            response = response.split("ssistant\n")[1].strip().strip("\n")
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()
        
        return response.strip()


class QAModelFactory:
    """Factory for creating QA models."""
    
    @staticmethod
    def create_qa_model(model_name_or_path: str, **kwargs) -> BaseQAModel:
        """Create the (Mistral) HuggingFace QA model for the SARA / Standard RAG pipeline."""
        quantization = kwargs.get('quantization', None)
        device = kwargs.get('device', 'cuda')
        max_new_tokens = kwargs.get('max_new_tokens', 64)
        return HuggingFaceQAModel(model_name_or_path, quantization=quantization,
                                  device=device, max_new_tokens=max_new_tokens)