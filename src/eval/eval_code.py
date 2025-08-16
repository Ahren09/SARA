"""
Evaluate code generation on CodeRAG Bench HumanEval subset with optional retrieval.
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


CODE_PROMPT = """Complete the following Python function.

{context}

{prompt}

Provide only the complete function implementation without any explanation.
"""


def extract_code(text, entry_point):
    """Extract the generated code from the model output."""
    # Try to find code blocks
    code_block_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to find function definition
    func_pattern = rf'def {entry_point}\(.*?\):'
    match = re.search(func_pattern, text, re.DOTALL)
    if match:
        # Extract from function definition to the end or to next def
        start_pos = match.start()
        # Find the next 'def ' or end of text
        rest_of_text = text[start_pos:]
        next_def = re.search(r'\ndef\s+\w+\(', rest_of_text[4:])  # Skip past first 'def '
        if next_def:
            end_pos = start_pos + next_def.start() + 4
            code = text[start_pos:end_pos].strip()
        else:
            code = rest_of_text.strip()

        # Check if there are any import statements after the function
        # and move them to the beginning
        lines = code.split('\n')
        imports = []
        other_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(line)
            else:
                other_lines.append(line)

        if imports:
            # Put imports first
            return '\n'.join(imports) + '\n\n' + '\n'.join(other_lines)
        return code

    # Fallback: return the text as is
    return text.strip()


def check_code_correctness(code, test, entry_point, timeout=3.0):
    """Check if the generated code passes the test cases."""
    import signal
    import io
    import contextlib

    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    # Combine code and test
    full_code = code + "\n" + test

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        # Execute code
        namespace = {}
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            with contextlib.redirect_stderr(output):
                exec(full_code, namespace)

        return True
    finally:
        # Always cancel timeout, even if an exception occurred
        signal.alarm(0)


def eval_code_rag(args):
    """
    Evaluate on CodeRAG Bench HumanEval subset.
    """

    print_colored(f"\n{'='*60}", "blue")
    print_colored(f"Evaluating CodeRAG HumanEval: {args.model_name_or_path}", "blue")
    print_colored(f"{'='*60}\n", "blue")

    # Load CodeRAG Bench HumanEval subset
    dataset = load_dataset("code-rag-bench/humaneval", split=args.split if hasattr(args, 'split') and args.split else "train")

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

    print_colored(f"✓ Model loaded: {args.model_name_or_path}", "green")

    # Determine output file
    if hasattr(args, 'output_file') and args.output_file:
        output_file = args.output_file
    else:
        output_file = get_answer_file_path(
            args.output_dir,
            args.model_name_or_path,
            args.retriever_name_or_path if hasattr(args, 'retriever_name_or_path') and args.retriever_name_or_path else "no_retrieval",
            "code_rag_humaneval",
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
        existing_results.add(line['task_id'])

    # Tracking statistics
    num_new_results = 0
    correct_count = 0
    total_count = 0
    error_count = 0

    # Check if we need retrieval
    use_retrieval = hasattr(args, 'retriever_name_or_path') and args.retriever_name_or_path and args.retriever_name_or_path.lower() != 'none'

    for index_row, row in enumerate(tqdm(dataset, desc="CodeRAG HumanEval")):
        if index_row >= args.max_eval_samples:
            break

        task_id = row['task_id']

        if not args.debug and task_id in existing_results:
            print(f"Skipping task {task_id}")
            continue

        prompt = row['prompt']
        entry_point = row['entry_point']
        test = row['test']
        canonical_solution = row.get('canonical_solution', '')

        print_colored(f"Task ID: {task_id}", "green")

        # Prepare context with RAG if enabled
        context_text = ""
        retrieved_docs = []

        if use_retrieval:
            # Use provided context or canonical solution as retrievable documents
            if 'context' in row and row['context']:
                documents = row['context'] if isinstance(row['context'], list) else [row['context']]
            elif canonical_solution:
                documents = [canonical_solution]
            else:
                documents = []

            if documents:
                retriever = prepare_retriever(args, documents)

                # Retrieve context
                retrieved_nodes = retriever.retrieve(prompt)

                # Take top-k documents
                k = getattr(args, 'k', 5)
                context_text = "\n\n".join([node.text for node in retrieved_nodes[:k]])
                retrieved_docs = [node.text for node in retrieved_nodes[:k]]

        # Format prompt
        if context_text:
            full_prompt = CODE_PROMPT.format(context=f"Context:\n{context_text}\n", prompt=prompt)
        else:
            full_prompt = CODE_PROMPT.format(context="", prompt=prompt)

        # Generate code
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)

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

        # Remove the prompt from generated text - look for common markers
        markers = [
            "Provide only the complete function implementation without any explanation.",
            "Solution:",
            f"def {entry_point}("
        ]

        for marker in markers:
            if marker in generated_text:
                # For function definition marker, include it in the output
                if marker.startswith("def "):
                    idx = generated_text.find(marker)
                    generated_text = generated_text[idx:].strip()
                else:
                    generated_text = generated_text.split(marker)[-1].strip()
                break

        # Extract code from generated text
        generated_code = extract_code(generated_text, entry_point)

        if args.debug:
            print_colored(f"\n--- Extracted Code ---", "cyan")
            # Truncate very long code for readability
            if len(generated_code) > 500:
                print(generated_code[:500] + "\n... (truncated)")
            else:
                print(generated_code)
            print_colored(f"--- End Extracted Code ---\n", "cyan")

        # Check correctness by running test cases (pass@1)
        is_correct = False
        error_msg = ""

        try:
            # Run the actual test cases
            is_correct = check_code_correctness(generated_code, test, entry_point, timeout=3.0)
        except TimeoutError:
            error_msg = "Test execution timed out"
            is_correct = False
        except Exception as e:
            # Any error means the code failed the tests
            error_msg = f"{type(e).__name__}: {str(e)}"
            is_correct = False

        if is_correct:
            correct_count += 1
        else:
            error_count += 1
            if not error_msg:
                error_msg = "Test cases failed"

        total_count += 1

        print_colored(f"Prompt:\t{prompt[:100]}...", "red")
        if len(generated_code) > 200:
            print_colored(f"Generated:\t{generated_code[:200]}...", "yellow")
        else:
            print_colored(f"Generated:\t{generated_code}", "yellow")
        print_colored(f"Correct: {is_correct}", "green" if is_correct else "red")
        if error_msg:
            print_colored(f"Error: {error_msg[:200]}", "red")

        # Print cumulative pass@1 after each example
        current_pass_at_1 = correct_count / total_count
        print_colored(f"Pass@1 (examples 1-{total_count}): {current_pass_at_1:.4f} ({correct_count}/{total_count})", "cyan")

        # Collect results
        result_entry = {
            "task_id": task_id,
            "prompt": prompt,
            "context": context_text,
            "natural_language_evidence": retrieved_docs if use_retrieval else [],
            "generated_code": generated_code,
            "entry_point": entry_point,
            "correct": is_correct,
            "error_msg": error_msg,
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
        print_colored(f"Tasks evaluated: {total_count}", "white")
        print_colored(f"Correct solutions: {correct_count}", "white")
        print_colored(f"Errors: {error_count}", "white")
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

    eval_code_rag(args)
