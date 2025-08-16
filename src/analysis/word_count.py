
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import numpy as np
import sys
import json
from llama_index.core.node_parser import SentenceSplitter


from src.arguments import parse_args
from src.utils.data_utils import load_dataset_for_eval


def count_num_tokens(tokenizer, dataset):
    
    retrieval_cache_path = os.path.join(args.cache_dir, args.dataset_name.split('/')[-1], args.retriever_name_or_path.split("/")[-1])
    
    num_tokens = []
    for index_row, row in enumerate(tqdm(dataset, desc=f"[Eval] {args.dataset_name.split('/')[-1]}")):
        # print(row)
        
        retrieved_docs_path = os.path.join(retrieval_cache_path, str(index_row),
                                               f"retrieved_docs_chunk{args.chunk_size}.json")
        
        if os.path.exists(retrieved_docs_path):
            question2retrieved_docs = json.load(open(retrieved_docs_path, 'r'))
        else:
            continue
        
        for question, retrieved_docs in question2retrieved_docs.items():
            for doc in retrieved_docs:
                if len(doc) > 0:
                    num_tokens.append(len(tokenizer(doc)['input_ids']))
                    
        
    return num_tokens
        
def count_context_length(tokenizer, dataset):
    context_length_li = []

    prev = None
    for index_row, row in enumerate(tqdm(dataset, desc=f"{args.dataset_name.split('/')[-1]}")):
        contexts = None
        if isinstance(row['context'], str):
            contexts = row['context']
        elif isinstance(row['context'], list):
            contexts = " ".join(row['context'])
        else:
            raise ValueError("context should be str or list")

        context_length = len(tokenizer.encode(contexts))
        if context_length == prev:
            continue


        prev = context_length
        context_length_li.append(context_length)

    return context_length_li

def count_num_tokens2(tokenizer, dataset):
    """Count #Evidence based on the original dataset.
    """
    
    num_tokens_li = []

    text_splitter = SentenceSplitter(separator="\n", chunk_size=args.chunk_size, chunk_overlap=5)

    prev = None
    for index_row, row in enumerate(tqdm(dataset, desc=f"{args.dataset_name.split('/')[-1]}")):
        contexts = None
        if isinstance(row['context'], str):
            contexts = row['context']
        elif isinstance(row['context'], list):
            contexts = " ".join(row['context'])
        else:
            raise ValueError("context should be str or list")


        splits = text_splitter.split_text(contexts)
        for doc in splits:
            num_tokens = len(tokenizer(doc)['input_ids'])

            if len(num_tokens_li) > 0 and num_tokens == num_tokens_li[-1]:
                continue
            num_tokens_li.append(num_tokens)


    return num_tokens_li
    
    

if __name__ == "__main__":
    args = parse_args('metrics')
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/mistral-7b-instruct-v0.2")
    
    
    path = "outputs/num_tokens.json"
    path_context_length = "outputs/context_length.json"

    if os.path.exists(path) and not args.overwrite:
        num_tokens_d = json.load(open(path, 'r'))
    else:
        num_tokens_d = {}

    if os.path.exists(path_context_length) and not args.overwrite:
        context_length_d = json.load(open(path_context_length, 'r'))
    else:
        context_length_d = {}
        
    args.retriever_name_or_path = "bm25"
    

    for dataset_name, subset_name, split in [
        ("THUDM/LongBench", "qmsum", "test"),
        ("THUDM/LongBench", "2wikimqa", "test"),
        ("THUDM/LongBench", "multifieldqa_en", "test"),
        # ("allenai/qasper", None, "test"),
        # ("deepmind/narrativeqa", None, "validation"), 
        # ("Ahren09/hotpotqa", None, "validation"), 
        # ("emozilla/quality", None, "validation")
    ]:
        if dataset_name in num_tokens_d and not args.overwrite:
            continue
        args.dataset_name = dataset_name
        args.subset_name = subset_name
        args.split = split
        dataset = load_dataset_for_eval(args)
        if dataset_name == "THUDM/LongBench":
            num_tokens_d[subset_name] = count_num_tokens(tokenizer, dataset)
        else:
            num_tokens_d[dataset_name] = count_num_tokens(tokenizer, dataset)
        del dataset
    json.dump(num_tokens_d, open(path, 'w'), indent=2)
    
    for dataset_name, subset_name, split in [
        ("THUDM/LongBench", "qmsum", "test"),
        ("THUDM/LongBench", "2wikimqa", "test"),
        ("THUDM/LongBench", "multifieldqa_en", "test"),
        # ("allenai/qasper", None, "test"),
        # ("deepmind/narrativeqa", None, "validation"), 
        # ("Ahren09/hotpotqa", None, "validation"), 
        # ("emozilla/quality", None, "validation"),
        # ("rajpurkar/squad_v2", None, "validation"), 
        # ("triviaqa", None, "train"),
    ]:
        # if dataset_name in num_tokens_d:
        #     continue
            
        args.dataset_name = dataset_name
        args.subset_name = subset_name
        args.split = split

        args.split = split
        
        dataset, original_dataset = load_dataset_for_eval(args)
        # num_tokens_d[dataset_name] = count_num_tokens2(tokenizer, original_dataset)

        if dataset_name == "THUDM/LongBench":
            context_length_d[subset_name] = count_context_length(tokenizer, original_dataset)
        else:
            context_length_d[dataset_name] = count_context_length(tokenizer, original_dataset)
        
    
    json.dump(context_length_d, open(path_context_length, 'w'), indent=2)
    
    
    print("Done!")