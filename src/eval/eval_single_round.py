"""
Evaluate datasets where 1 document correspond to 1 questions

This module implements an adaptive retrieval and context compression pipeline
with unified interfaces for QA models, compressors, and evidence selection.
"""

import asyncio
import importlib
import json
import logging
import os
import sys
import traceback
import psutil
import threading
from collections import defaultdict
from copy import deepcopy
from time import time, perf_counter

import numpy as np
import torch
from accelerate.logging import get_logger
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from src.arguments import parse_args

from src.metrics.metrics_distance import cosine_distance_torch
from src.model.retriever import getBM25Retriever, getHuggingFaceRetriever
from src.prompts import (PROMPT_HF_MODELS, PROMPT_MISTRAL, PROMPT_HF_MODELS_QUESTION_ONLY, ANSWER_INSTRUCTION,
                     ANSWER_INSTRUCTION_MULTI_CHOICE, MISTRAL_CHAT_TEMPLATE
                     )
from src.model import SentenceBERTEmbedding
from src.utils.data_utils import load_dataset_for_eval
from src.utils.data_utils import write_jsonl, read_jsonl
from src.utils.eval_utils import (
    load_eval_model,
    load_hf_pipeline,
    get_retrieval_embeds,
    load_compressor_and_tokenizers_eval,
    prepare_retriever,
)
from src.utils.utility import print_colored

from src.metrics.metrics_conditional_perplexity import ConditionalSelfInformation
from src.eval.qa_models import QAModelFactory
from src.eval.compressor_models import CompressorFactory, EvidenceSelector
from src.eval.adaptive_evidence_manager import AdaptiveEvidenceManager, create_evidence_manager_from_args

logger = get_logger(__name__)


COMPRESS = "<COMPRESS>"
DELIMITER = "<|>"

# Suppress INFO logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


def format_docs(docs):
    """Format documents for display."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_list_of_documents(paragraphs):
    """Convert paragraphs to Document objects."""
    docs = []
    for paragraph in paragraphs:
        docs.append(Document(text=paragraph))
    return docs


def initialize_components(args):
    """
    Initialize all components with unified interfaces.
    
    Returns:
        Tuple of (qa_model, compressor, evidence_selector, evidence_manager, csi_model)
    """
    print_colored("Initializing unified components...", "blue")
    
    # Initialize QA model with unified interface
    qa_model = None
    try:
        qa_model = QAModelFactory.create_qa_model(
            args.model_name_or_path,
            quantization=getattr(args, 'quantization', None),
            device=args.device,
            max_new_tokens=getattr(args, 'max_new_tokens', 64)
        )
        print_colored(f"✓ Unified QA model initialized: {qa_model.model_name}", "green")
    except Exception as e:
        print_colored(f"⚠ Could not create unified QA model ({e}), will use legacy fallback", "yellow")
        qa_model = None
    
    # Initialize compressor with unified interface
    compressor = None
    evidence_selector = None
    if getattr(args, 'compressor_name_or_path', None) is not None:
        try:
            compressor = CompressorFactory.create_compressor(
                args.compressor_name_or_path, 
                args.device,
                dtype=torch.bfloat16
            )
            evidence_selector = EvidenceSelector(
                compressor, 
                getattr(args, 'evidence_selection', None)
            )
            print_colored(f"✓ Unified compressor initialized: {compressor.model_name}", "green")
        except Exception as e:
            print_colored(f"⚠ Could not create unified compressor ({e}), will use legacy fallback", "yellow")
            compressor = None
            evidence_selector = None
    
    # Initialize adaptive evidence manager
    evidence_manager = create_evidence_manager_from_args(args)
    print_colored(f"✓ Evidence manager initialized: {evidence_manager.config}", "green")
    
    # Initialize CSI model for evidence selection if needed
    csi_model = None
    if getattr(args, 'evidence_selection', False) or getattr(args, 'reconstruct_context', False):
        csi_model = ConditionalSelfInformation(model_name="gpt2", device=args.device)
        print_colored("✓ CSI model initialized for evidence selection", "green")
    
    return qa_model, compressor, evidence_selector, evidence_manager, csi_model


def get_retrieved_docs(args, question, question_index, row, retriever, question2retrieved_docs):
    """Get retrieved documents for a question."""
    if str(question_index) in question2retrieved_docs:
        return question2retrieved_docs[str(question_index)]
    retrieved = retriever.retrieve(question)
    docs = [d.text for d in retrieved]
    question2retrieved_docs[str(question_index)] = docs
    return docs


def retrieve_and_allocate_evidence(args, question, question_index, row, retriever, 
                                   evidence_selector, evidence_manager, question2retrieved_docs, csi_model):
    """
    Retrieve documents and allocate evidence using adaptive evidence manager.
    
    This function implements the core evidence allocation logic with clear parameter names:
    - k: Number of evidence pieces in natural language format
    - n: Total number of evidence pieces (natural language + compressed tokens)
    - n-k: Number of evidence pieces in compressed token format
    """
    # Get retrieved documents
    retrieved_docs = get_retrieved_docs(args, question, question_index, row, retriever, question2retrieved_docs)
    
    # Limit to total evidence needed (n)
    total_needed = evidence_manager.config.total_evidence_needed
    docs_for_selection = retrieved_docs[:total_needed] if total_needed > 0 else retrieved_docs
    
    # Apply evidence selection if configured
    selected_evidence = None
    if evidence_selector and getattr(args, 'evidence_selection', None):
        print_colored(f"Applying evidence selection: {args.evidence_selection}", "white")
        # For evidence selection, we select from total evidence needed (n)
        selected_docs, remaining_docs = evidence_selector.select_evidence(
            docs_for_selection, question, total_needed, args.device, csi_model
        )
        selected_evidence = selected_docs
    
    # Allocate evidence between natural language (k) and compressed formats (n-k)
    evidence_allocation = evidence_manager.allocate_evidence(
        retrieved_docs=docs_for_selection,
        selected_evidence=selected_evidence,
        model_name=args.model_name_or_path,
        args=args
    )
    
    return evidence_allocation


def count_tokens(text, tokenizer=None):
    """Count tokens in text. Returns word count as approximation if tokenizer not available."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except:
            pass
    # Fallback to word count approximation (typically ~1.3 tokens per word for English)
    return int(len(text.split()) * 1.3)


def generate_answer_with_unified_interface(qa_model, question, context_list, additional_context,
                                         choices, compressor, args):
    """Generate answer using unified QA model interface."""
    track_metrics = args.track_metrics

    # Initialize metrics
    metrics = {}
    input_tokens = 0

    if track_metrics:
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory tracking
        gpu_memory_before = 0
        gpu_memory_peak = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()

        # Count input tokens
        tokenizer = getattr(qa_model, 'tokenizer', None)
        input_tokens += count_tokens(question, tokenizer)
        if context_list:
            for ctx in context_list:
                input_tokens += count_tokens(ctx, tokenizer)
        if additional_context:
            for add_ctx in additional_context:
                input_tokens += count_tokens(add_ctx, tokenizer)
        if choices:
            for choice in choices:
                input_tokens += count_tokens(str(choice), tokenizer)

        start_time = perf_counter()
        ttft_captured = False
        ttft = None

        def capture_ttft():
            nonlocal ttft_captured, ttft
            if not ttft_captured:
                ttft = perf_counter() - start_time
                ttft_captured = True

        ttft_thread = threading.Timer(0.001, capture_ttft)
        ttft_thread.start()

    retrieval_kwargs = {}
    if compressor and len(additional_context) > 0:
        embeds = compressor.get_embeddings(additional_context)
        retrieval_kwargs['retrieval_embeds'] = embeds.to(args.device)

    response = qa_model.generate_answer(
        question=question,
        context="\n\n".join(context_list) if context_list else None,
        choices=choices,
        additional_context=additional_context,
        retrieval_kwargs=retrieval_kwargs,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty
    )
    
    if track_metrics:
        end_time = perf_counter()
        e2e_latency = end_time - start_time
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory tracking
        gpu_memory_after = 0
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Count output tokens
        output_tokens = count_tokens(response, tokenizer)
        total_tokens = input_tokens + output_tokens

        if ttft is None:
            ttft = e2e_latency

        ttft_thread.cancel()

        metrics = {
            'ttft': ttft,
            'e2e_latency': e2e_latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_peak_mb': memory_after,
            'gpu_memory_before_mb': gpu_memory_before,
            'gpu_memory_after_mb': gpu_memory_after,
            'gpu_memory_peak_mb': gpu_memory_peak,
            'gpu_memory_delta_mb': gpu_memory_after - gpu_memory_before
        }

    if track_metrics:
        return {'response': response, 'metrics': metrics}
    else:
        return response


def generate_answer_with_legacy_fallback(model, tokenizer, pipe, prompt, additional_context,
                                       compressor, args):
    """Generate answer using legacy fallback methods."""
    track_metrics = args.track_metrics

    # Initialize metrics
    metrics = {}
    input_tokens = 0

    if track_metrics:
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory tracking
        gpu_memory_before = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()

        # Count input tokens
        input_tokens = count_tokens(prompt, tokenizer)

        start_time = perf_counter()

    retrieval_kwargs = {}
    if compressor and len(additional_context) > 0:
        embeds = compressor.get_embeddings(additional_context)
        retrieval_kwargs['retrieval_embeds'] = embeds.to(model.device)
    
    if "gemma" in args.model_name_or_path.lower():
        inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt").to(
            model.device)
    else:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
    
    # TTFT approximation - first token generated
    if track_metrics:
        generation_start = perf_counter()
    
    answer_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        min_length=50,
        length_penalty=args.length_penalty,
        num_beams=5,
        early_stopping=True,
        stop_sequence=None,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        repetition_penalty=args.repetition_penalty,
        **retrieval_kwargs
    )
    
    response = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[1].strip()
    
    if track_metrics:
        end_time = perf_counter()
        e2e_latency = end_time - start_time
        ttft = perf_counter() - generation_start  # Approximation
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory tracking
        gpu_memory_after = 0
        gpu_memory_peak = 0
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Count output tokens
        output_tokens = count_tokens(response, tokenizer)
        total_tokens = input_tokens + output_tokens

        metrics = {
            'ttft': ttft,
            'e2e_latency': e2e_latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_peak_mb': memory_after,
            'gpu_memory_before_mb': gpu_memory_before,
            'gpu_memory_after_mb': gpu_memory_after,
            'gpu_memory_peak_mb': gpu_memory_peak,
            'gpu_memory_delta_mb': gpu_memory_after - gpu_memory_before
        }
        return {'response': response, 'metrics': metrics}

    return response


def clean_response(response):
    """Clean up response text."""
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
    return response


def eval_single_round(args):
    """Batched evaluation of Standard RAG or SARA (release plan §2/§9).

    method == 'rag':  retrieved docs rendered as plain text (k == n, no compression).
    method == 'sara': k natural-language pieces + (n-k) compressed retrieval embeddings.

    Generation is fully batched; only lightweight per-example formatting/retrieval runs in Python loops
    (CLAUDE rule 8). Emits per-example records + a run-level ``.meta.json`` (release plan §9).
    """
    import time

    k = getattr(args, 'k', 0)
    n = getattr(args, 'n', 0)
    method = getattr(args, 'method', None) or ('rag' if (n - k) <= 0 else 'sara')
    n_minus_k = 0 if method == 'rag' else (n - k)
    batch_size = (getattr(args, 'batch_size', None)
                  or getattr(args, 'per_device_eval_batch_size', None) or 16)

    print_colored(f"[Eval] method={method} k={k} n={n} compressed={n_minus_k} batch_size={batch_size}", "blue")

    dataset, original_dataset = load_dataset_for_eval(args)
    qa_model, compressor, evidence_selector, evidence_manager, csi_model = initialize_components(args)
    if method == 'rag':
        compressor = None  # Standard RAG never uses the compressor (single shared pipeline, §10)
    if method == 'sara' and compressor is None:
        raise ValueError("SARA evaluation requires a compressor (compressor_name_or_path); none loaded.")
    if qa_model is None or not hasattr(qa_model, 'generate_answers_batched'):
        raise ValueError("Batched eval requires a HuggingFace QA model exposing generate_answers_batched().")

    if args.retriever_name_or_path in ["Salesforce/SFR-Embedding-Mistral", "BAAI/bge-reranker-v2-m3"]:
        Settings.embed_model = HuggingFaceEmbedding(model_name=args.retriever_name_or_path)

    # ---- Pass 1: lightweight per-example retrieval + allocation + prompt construction ----
    work = []
    max_samples = getattr(args, 'max_eval_samples', 10**9)
    answer_format = getattr(args, 'answer_format', 'short_answer')
    for index_row, row in enumerate(tqdm(dataset, desc=f"[Pass1] {args.dataset_name.split('/')[-1]}")):
        if index_row >= max_samples:
            break
        questions = row["question"] if isinstance(row["question"], list) else [row["question"]]
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        choices = row.get("choices")
        ans_reform = row.get("answer_reformatted")
        if ans_reform is not None and not isinstance(ans_reform, list):
            ans_reform = [ans_reform]

        documents = None
        if isinstance(row['context'], str):
            documents = [row['context']]
        elif isinstance(row['context'], list) and len(row['context']) > 0:
            documents = ["\n".join(row['context'])]
        retriever = None
        if documents is not None and args.do_retrieval:
            retriever = prepare_retriever(args, documents, similarity_top_k=max(k, n))

        question2retrieved_docs = {}
        for q_idx, (question, answer) in enumerate(zip(questions, answers)):
            if args.do_retrieval and retriever is not None:
                allocation = retrieve_and_allocate_evidence(
                    args, question, q_idx, row, retriever, evidence_selector,
                    evidence_manager, question2retrieved_docs, csi_model)
                qa_inputs = evidence_manager.create_qa_inputs(allocation, question, choices, answer_format)
                nl_evidence = qa_inputs['context'] or ""
                compressed_docs = list(qa_inputs['additional_context'] or [])
                compressed_prompt_text = qa_inputs['compressed_prompt_text'] or ""
            else:
                nl_evidence = row['context'] if isinstance(row['context'], str) else "\n".join(row['context'])
                compressed_docs, compressed_prompt_text = [], ""
            if method == 'rag':
                compressed_docs, compressed_prompt_text = [], ""

            if isinstance(nl_evidence, list):
                nl_text, nl_list = "\n\n".join(nl_evidence), nl_evidence
            else:
                nl_text, nl_list = nl_evidence, ([nl_evidence] if nl_evidence else [])

            prompt = PROMPT_MISTRAL.format(question=question, context=nl_text,
                                           additional_context=compressed_prompt_text,
                                           answer_instruction=ANSWER_INSTRUCTION)
            work.append(dict(
                index_row=index_row, question_id=q_idx, example_id=row.get('example_id'),
                id=row.get('id', index_row), question=question, answer=answer,
                answer_reformatted=(ans_reform[q_idx] if ans_reform else None),
                choices=choices, correct_choice=row.get('correct_choice'),
                nl_list=nl_list, compressed_docs=compressed_docs, prompt=prompt,
            ))

    # ---- Pass 2: batched generation (single GPU call per batch; no per-example model calls) ----
    results = []
    rep = getattr(args, 'repetition_penalty', None) or 1.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_out_tokens, total_pre_trunc, total_post_trunc = 0, 0, 0
    t_start = time.perf_counter()
    for b0 in tqdm(range(0, len(work), batch_size), desc=f"[Pass2 {method}]"):
        batch = work[b0:b0 + batch_size]
        prompts = [it['prompt'] for it in batch]
        embeds, counts = None, None
        if method == 'sara' and compressor is not None:
            counts = [len(it['compressed_docs']) for it in batch]
            flat = [d for it in batch for d in it['compressed_docs']]  # example-major, document-order
            if sum(counts) > 0:
                embeds = compressor.get_embeddings(flat)
            else:
                counts = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preds, input_ids, attn = qa_model.generate_answers_batched(
            prompts, retrieval_embeds=embeds, per_example_counts=counts, repetition_penalty=rep)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        in_tokens = attn.sum(dim=1).tolist()
        for i, it in enumerate(batch):
            pred = clean_response(preds[i])
            out_tok = len(qa_model.tokenizer.encode(pred, add_special_tokens=False))
            total_out_tokens += out_tok
            total_post_trunc += int(in_tokens[i])
            results.append({
                "id": it['index_row'], "question_id": it['question_id'],
                "example_id": it['example_id'],  # paper id -> document-clustered bootstrap (§11)
                "question": it['question'],
                "natural_language_evidence": it['nl_list'],
                "compressed_token_evidence": it['compressed_docs'],
                "pred": pred, "answer": it['answer'], "answer_reformatted": it['answer_reformatted'],
                "choices": it['choices'], "correct_choice": it['correct_choice'],
                "method": method, "k": k, "n": n, "n_minus_k": n_minus_k,
                "input_tokens": int(in_tokens[i]), "output_tokens": int(out_tok),
                "num_compressed_embeddings": len(it['compressed_docs']),
                # legacy field aliases kept for the existing scorer
                "context": it['nl_list'], "additional_context": it['compressed_docs'],
            })
    total_time = time.perf_counter() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_jsonl(results, args.output_file)
    run_meta = {
        "method": method, "k": k, "n": n, "num_examples": len(results), "batch_size": batch_size,
        "total_gen_seconds": round(total_time, 3),
        "examples_per_second": round(len(results) / total_time, 3) if total_time > 0 else 0,
        "generated_tokens_per_second": round(total_out_tokens / total_time, 2) if total_time > 0 else 0,
        "mean_input_tokens": round(total_post_trunc / max(1, len(results)), 1),
        "peak_gpu_mem_mb": round(peak_mb, 1),
        "retriever": args.retriever_name_or_path, "model": args.model_name_or_path,
        "repetition_penalty": rep, "max_new_tokens": getattr(args, 'max_new_tokens', 64),
    }
    with open(args.output_file + ".meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print_colored(
        f"[Eval] wrote {len(results)} preds -> {args.output_file} | "
        f"{run_meta['examples_per_second']} ex/s | peak {run_meta['peak_gpu_mem_mb']} MB", "green")
    return results


def debug_sync_mode(args):
    """Run evaluation for debugging (eval_single_round is synchronous and already batched)."""
    print("Running in sync mode for debugging...")
    eval_single_round(args)


if __name__ == "__main__":
    args = parse_args('eval')
    if args.debug:
        debug_sync_mode(args)
    else:
        eval_single_round(args)
