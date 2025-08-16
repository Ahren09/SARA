"""
Calculate the number of in-context tokens used as inputs


"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from src.utils import read_jsonl


def calculate_compression_ratio(original_size, compressed_size):
    """
    Calculate the compression ratio given the original and compressed sizes.

    Args:
        original_size (int): Size of the original data in bytes.
        compressed_size (int): Size of the compressed data in bytes.

    Returns:
        float: Compression ratio.
    """
    if original_size == 0:
        raise ValueError("Original size cannot be zero.")

    return original_size / compressed_size


if __name__ == "__main__":
    retriever_name = "bm25"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/mistral-7b-instruct-v0.2")

    MAX_CONTEXTS = 10
    # for dataset_name in ["hotpot_qa", "narrativeqa", "qasper", "quality", "squad_v2", "triviaqa"]:


    with pd.ExcelWriter(f"compression_ratio_{retriever_name}.xlsx") as writer:
        for dataset_name in ["squad_v2", "qasper", "narrativeqa", "triviaqa", "quality", "hotpot_qa"]:
            context_lengths = []
            num_sentences = []
            avg_num_tokens = []
            retriever_cache_dir = os.path.join("cache", dataset_name, retriever_name)

            example_ids = os.listdir(retriever_cache_dir)
            example_ids = sorted([int(id) for id in example_ids])

            all_ratios = defaultdict(list)
            for example_id in example_ids:
                if example_id >= 200:
                    break
                example_path = os.path.join(retriever_cache_dir, str(example_id), "retrieved_docs_chunk256.json")
                if not os.path.exists(example_path):
                    print(f"File not found: {example_path}")
                    continue

                d = json.load(open(example_path, "r"))

                for question_id, contexts in d.items():
                    if len(contexts) < MAX_CONTEXTS:
                        continue
                    encoded_contexts = tokenizer.batch_encode_plus(contexts)

                    context_lengths = [len(c) for c in encoded_contexts['input_ids']]

                    num_sentences = [len(sent_tokenize(c)) for c in contexts]

                    for num_nl_evidence in range(1, MAX_CONTEXTS):
                        if num_nl_evidence >= len(contexts):
                            break
                        num_tokens = sum(context_lengths[:num_nl_evidence]) + sum(
                            num_sentences[num_nl_evidence:MAX_CONTEXTS])
                        ratio = num_tokens / sum(context_lengths[:MAX_CONTEXTS])
                        assert ratio < 1.
                        all_ratios[num_nl_evidence].append(ratio)
                        avg_num_tokens.append(num_tokens)

            df = pd.DataFrame(columns=["num_nl_evidence", "Avg", "Std", "#Tokens"])
            for i, (num_nl_evidence, ratios) in enumerate(all_ratios.items()):
                avg_ratio = np.round(np.mean(ratios) * 100, 2)
                std_ratio = np.round(np.std(ratios) * 100, 2)

                print(f"{num_nl_evidence}\t{avg_ratio:.2f}")
                df.loc[i] = [num_nl_evidence, avg_ratio, std_ratio, avg_num_tokens[i]]

            df.to_excel(writer, sheet_name=dataset_name, index=False)
    
    print("Done!")
