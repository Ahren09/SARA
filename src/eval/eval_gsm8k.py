"""
Evaluate math reasoning on GSM8K dataset with optional retrieval.
"""

import os
import re
import sys
import torch
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


from src.arguments import parse_args, set_attributes_from_yaml, get_answer_file_path
from src.utils.data_utils import write_jsonl, read_jsonl
from src.utils.eval_utils import load_eval_model, prepare_retriever
from src.utils.utility import print_colored, get_yaml_file
from src.model import SFR
from src.model.retriever import getBM25Retriever, getHuggingFaceRetriever

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


MATH_PROMPT = """Solve the following math problem step by step.

{context}Question: {question}

Provide your reasoning and then give the final numerical answer. Format your final answer as: The answer is: <number>

Solution:"""


def extract_answer(text):
    """Extract the final numerical answer from generated text."""
    # Look for patterns like "The answer is: 42" or "#### 42"
    patterns = [
        r'[Tt]he answer is[:\s]+([0-9,]+\.?[0-9]*)',
        r'####\s*([0-9,]+\.?[0-9]*)',
        r'[Ff]inal answer[:\s]+([0-9,]+\.?[0-9]*)',
        r'=\s*([0-9,]+\.?[0-9]*)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).replace(',', '').strip()
            # Remove trailing period if present
            return answer.rstrip('.')

    # Fallback: extract last number in the text
    numbers = re.findall(r'([0-9,]+\.?[0-9]*)', text)
    if numbers:
        answer = numbers[-1].replace(',', '').strip()
        # Remove trailing period if present
        return answer.rstrip('.')

    return ""


def eval_gsm8k(args):
    """
    Evaluate on GSM8K math reasoning dataset.
    """

    print_colored(f"\n{'='*60}", "blue")
    print_colored(f"Evaluating GSM8K: {args.model_name_or_path}", "blue")
    print_colored(f"{'='*60}\n", "blue")

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split=args.split if hasattr(args, 'split') and args.split else "test")

    # Load training set for retrieval if needed
    train_dataset = None
    global_retriever = None
    if hasattr(args, 'retriever_name_or_path') and args.retriever_name_or_path and args.retriever_name_or_path.lower() != 'none':
        print_colored("Loading GSM8K training set for retrieval...", "cyan")
        train_dataset = load_dataset("openai/gsm8k", "main", split="train")

        # Prepare retriever once with all training questions
        print_colored("Preparing retriever with training questions...", "cyan")
        train_questions = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in train_dataset]
        train_documents = [Document(text=q, doc_id=str(i)) for i, q in enumerate(train_questions)]
        train_documents = train_documents[:100]
        global_retriever = prepare_retriever(args, train_documents)

    # Load model
    model, tokenizer = load_eval_model(args.model_name_or_path, quantization=args.quantization)

    # Load compressor if needed (for compressor models)
    compressor = None
    compressor_tokenizer = None
    if "checkpoint" in args.model_name_or_path.lower():
        if hasattr(args, 'embedding_model') and args.embedding_model:
            compressor = SFR.from_pretrained(args.embedding_model, torch_dtype=torch.bfloat16)
            compressor.to(args.device if hasattr(args, 'device') and args.device else 'cuda')
            compressor.eval()
            compressor_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    print_colored(f" Model loaded: {args.model_name_or_path}", "green")

    # Determine output file
    if hasattr(args, 'output_file') and args.output_file:
        output_file = args.output_file
    else:
        output_file = get_answer_file_path(
            args.output_dir,
            args.model_name_or_path,
            args.retriever_name_or_path if hasattr(args, 'retriever_name_or_path') and args.retriever_name_or_path else "no_retrieval",
            "gsm8k",
            args.k if hasattr(args, 'k') else 0,
            args.n if hasattr(args, 'n') else 0,
            args.repetition_penalty if hasattr(args, 'repetition_penalty') else 1.0
        )

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print_colored(f"Output file: {output_file}", "cyan")

    # Load existing results if any
    if os.path.exists(output_file):
        generated_result_list = read_jsonl(output_file)
    else:
        generated_result_list = []

    existing_results = set()
    for line in generated_result_list:
        existing_results.add(int(line['id']))

    # Tracking statistics
    num_new_results = 0
    correct_count = 0
    total_count = 0

    # Check if we need retrieval
    use_retrieval = hasattr(args, 'retriever_name_or_path') and args.retriever_name_or_path and args.retriever_name_or_path.lower() != 'none'

    for index_row, row in enumerate(tqdm(dataset, desc="GSM8K")):
        if index_row >= args.max_eval_samples:
            break

        if not args.debug and index_row in existing_results:
            print(f"Skipping example {index_row}")
            continue

        question = row['question']
        answer = row['answer']

        # Extract ground truth answer (GSM8K format: "... #### 42")
        gt_answer = answer.split("####")[-1].strip()

        print_colored(f"Example ID: {index_row}", "green")

        # Prepare context with RAG if enabled
        context_text = ""
        retrieved_docs = []
        few_shot_examples = []

        if use_retrieval and train_dataset is not None and global_retriever is not None:
            # Retrieve similar questions from training set
            retrieved_nodes = global_retriever.retrieve(question)

            # Take top-k examples as few-shot demonstrations
            k = getattr(args, 'k', 5)
            few_shot_examples = []
            for node in retrieved_nodes[:k]:
                # Parse the retrieved example
                train_idx = int(node.node.ref_doc_id)
                train_example = train_dataset[train_idx]
                few_shot_examples.append({
                    'question': train_example['question'],
                    'answer': train_example['answer']
                })

            # Format few-shot examples as context
            context_parts = []
            for i, ex in enumerate(few_shot_examples, 1):
                # Extract just the final answer for the example
                ex_answer = ex['answer'].split("####")[-1].strip()
                ex_reasoning = ex['answer'].split("####")[0].strip()
                context_parts.append(f"Example {i}:\nQuestion: {ex['question']}\nSolution: {ex_reasoning}\nThe answer is: {ex_answer}")

            context_text = "\n\n".join(context_parts)
            retrieved_docs = [node.text for node in retrieved_nodes[:k]]

        # Format prompt
        if context_text:
            prompt = MATH_PROMPT.format(context=f"Here are some similar examples:\n\n{context_text}\n\n", question=question)
        else:
            prompt = MATH_PROMPT.format(context="", question=question)

        # Generate answer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        retrieval_kwargs = {}
        if compressor is not None and retrieved_docs:
            # Prepare compressed representations
            tokenized_retrieval_text = compressor_tokenizer(
                retrieved_docs,
                max_length=args.max_seq_length if hasattr(args, 'max_seq_length') else 512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            from src.utils.eval_utils import get_retrieval_embeds
            retrieval_embeds = get_retrieval_embeds(
                model=compressor,
                input_ids=tokenized_retrieval_text['input_ids'].to(compressor.device),
                attention_mask=tokenized_retrieval_text['attention_mask'].to(compressor.device),
            )
            retrieval_kwargs['retrieval_embeds'] = retrieval_embeds

        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens if hasattr(args, 'max_new_tokens') else 512,
            do_sample=False,
            temperature=min(args.temperature if hasattr(args, 'temperature') else 0.0, 0.001),
            use_cache=True,
            **retrieval_kwargs
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract answer from prompt
        if "Solution:" in generated_text:
            generated_text = generated_text.split("Solution:")[-1].strip()

        # Extract predicted answer
        pred_answer = extract_answer(generated_text)

        # Calculate accuracy
        is_correct = pred_answer == gt_answer
        if is_correct:
            correct_count += 1
        total_count += 1

        print_colored(f"Q:\t{question}", "red")
        print_colored(f"A:\t{generated_text[:200]}...", "yellow")
        print_colored(f"Pred:\t{pred_answer}", "cyan")
        print_colored(f"GT:\t{gt_answer}", "blue")
        print_colored(f"Correct: {is_correct}", "green" if is_correct else "red")

        # Collect results
        result_entry = {
            "id": index_row,
            "question": question,
            "context": context_text,
            "natural_language_evidence": retrieved_docs if use_retrieval else [],
            "pred": generated_text,
            "pred_answer": pred_answer,
            "answer": answer,
            "gt_answer": gt_answer,
            "correct": is_correct,
            "k": len(retrieved_docs) if use_retrieval else 0,
            "n": len(retrieved_docs) if use_retrieval else 0,
        }

        generated_result_list.append(result_entry)
        num_new_results += 1

        # Periodic saving
        if len(generated_result_list) % 10 == 0 and num_new_results > 0:
            print(f"\nGenerated {num_new_results} new results. Writing to {output_file}...")
            print_colored(f"  Current accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}", "yellow")
            write_jsonl(generated_result_list, output_file)

    # Final save and statistics
    print(f"\nFinished evaluation. Writing to {output_file}...")
    write_jsonl(generated_result_list, output_file)

    # Calculate final accuracy
    if total_count > 0:
        accuracy = correct_count / total_count
        print_colored(f"\n{'='*60}", "blue")
        print_colored("Evaluation Summary", "blue")
        print_colored(f"{'='*60}", "blue")
        print_colored(f"Model: {args.model_name_or_path}", "white")
        print_colored(f"Retrieval: {'Enabled' if use_retrieval else 'Disabled'}", "white")
        print_colored(f"Questions evaluated: {total_count}", "white")
        print_colored(f"Correct answers: {correct_count}", "white")
        print_colored(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})", "green")
        print_colored(f"{'='*60}\n", "blue")


if __name__ == "__main__":
    args = parse_args('eval')

    # Load YAML config
    if args.config:
        yaml_config = get_yaml_file(args.config)
        args = set_attributes_from_yaml(args, yaml_config)

    # Set defaults
    if not hasattr(args, 'chunk_size') or args.chunk_size is None:
        args.chunk_size = 256
    if not hasattr(args, 'k') or args.k is None:
        args.k = 5
    if not hasattr(args, 'n') or args.n is None:
        args.n = 10
    if not hasattr(args, 'max_eval_samples') or args.max_eval_samples is None:
        args.max_eval_samples = 1000000

    eval_gsm8k(args)
