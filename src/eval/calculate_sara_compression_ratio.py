from llama_index.core import Document
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import sys
import os
import nltk

# Add src to path for imports

from src.arguments import parse_args
from src.model.retriever import getBM25Retriever, getHuggingFaceRetriever
from src.utils.data_utils import load_dataset_for_eval
from src.utils.utility import print_colored

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def calculate_compression_ratio(args):
    """Calculate compression ratio for different k and n values."""
    dataset, original_dataset = load_dataset_for_eval(args)

    # Only need tokenizer for counting tokens
    qa_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Store results
    results = []

    # Different k and n combinations to test
    k_values = list(range(11))
    n_values = [10]

    # Limit samples for testing
    if args.max_eval_samples:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    for example_id, example in enumerate(tqdm(dataset, desc=args.dataset_name)):
        # Get context(s) - handle different formats
        if 'paragraphs' in example:
            # Multi-paragraph format (like eval_sara.py expects)
            contexts = [paragraph['context'] for paragraph in example['paragraphs']]
        elif 'context' in example:
            # Context can be either a string or a list of strings
            if isinstance(example['context'], list):
                contexts = example['context']
            else:
                contexts = [example['context']]
        else:
            continue

        documents = [Document(doc_id=str(doc_id), text=t) for (doc_id, t) in enumerate(contexts)]

        if len(contexts) == 0:
            continue

        # Initialize retriever
        if args.retriever_name_or_path == "bm25":
            sparse_retriever, _ = getBM25Retriever(documents, similarity_top_k=max(n_values))
        else:
            sparse_retriever, _ = getHuggingFaceRetriever(
                documents,
                model_name_or_path=args.retriever_name_or_path,
                similarity_top_k=max(n_values)
            )

        # Get question - handle different formats
        if 'question' in example:
            question = example['question']
        elif 'paragraphs' in example and len(example['paragraphs']) > 0:
            # Multi-paragraph format - get first question
            if 'qas' in example['paragraphs'][0] and len(example['paragraphs'][0]['qas']) > 0:
                question = example['paragraphs'][0]['qas'][0]['question']
            else:
                continue
        else:
            continue

        # Retrieve top-n documents
        nodes = sparse_retriever.retrieve(question)

        # Test different k and n combinations
        for n in n_values:
            if n > len(nodes):
                continue

            for k in k_values:
                if k > n:
                    continue

                # k documents in natural language
                natural_lang_text = "\n---\n".join([node.text for node in nodes[:k]])
                natural_lang_tokens = len(qa_tokenizer.encode(natural_lang_text)) if k > 0 else 0

                # (n-k) documents to be compressed
                # Each sentence becomes 1 compression token
                num_docs_to_compress = n - k
                if num_docs_to_compress > 0:
                    compressed_docs_text = "\n---\n".join([node.text for node in nodes[k:n]])
                    original_compressed_tokens = len(qa_tokenizer.encode(compressed_docs_text))

                    # Split into sentences - each sentence = 1 token
                    sentences = nltk.sent_tokenize(compressed_docs_text)
                    compressed_tokens = len(sentences)

                    compression_ratio = original_compressed_tokens / compressed_tokens if compressed_tokens > 0 else 0
                else:
                    original_compressed_tokens = 0
                    compressed_tokens = 0
                    compression_ratio = 0

                total_original_tokens = natural_lang_tokens + original_compressed_tokens
                total_final_tokens = natural_lang_tokens + compressed_tokens

                overall_compression_ratio = total_original_tokens / total_final_tokens if total_final_tokens > 0 else 0

                results.append({
                    'example_id': example_id,
                    'k': k,
                    'n': n,
                    'natural_lang_tokens': natural_lang_tokens,
                    'original_compressed_tokens': original_compressed_tokens,
                    'compressed_tokens': compressed_tokens,
                    'total_original_tokens': total_original_tokens,
                    'total_final_tokens': total_final_tokens,
                    'compression_ratio': compression_ratio,
                    'overall_compression_ratio': overall_compression_ratio
                })

    # Print summary statistics
    print_colored("\n=== Compression Ratio Summary ===", "blue")
    for n in n_values:
        for k in k_values:
            if k > n:
                continue

            filtered = [r for r in results if r['k'] == k and r['n'] == n]
            if not filtered:
                continue

            avg_compression = np.mean([r['compression_ratio'] for r in filtered if r['compression_ratio'] > 0])
            avg_overall = np.mean([r['overall_compression_ratio'] for r in filtered if r['overall_compression_ratio'] > 0])
            avg_original = np.mean([r['total_original_tokens'] for r in filtered])
            avg_final = np.mean([r['total_final_tokens'] for r in filtered])

            print_colored(f"\nk={k}, n={n}:", "yellow")
            print(f"  Avg original tokens: {avg_original:.1f}")
            print(f"  Avg final tokens: {avg_final:.1f}")
            print(f"  Avg compression ratio (compressed part only): {avg_compression:.2f}x")
            print(f"  Avg overall compression ratio: {avg_overall:.2f}x")

    return results


if __name__ == "__main__":
    args = parse_args('eval')

    # Set default values if not provided
    if not hasattr(args, 'max_eval_samples') or args.max_eval_samples is None:
        args.max_eval_samples = 10

    results = calculate_compression_ratio(args)

    print_colored(f"\n\nProcessed {len(results)} results", "green")
