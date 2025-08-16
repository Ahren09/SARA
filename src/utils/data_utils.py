import json
import os
import re
from argparse import Namespace
from collections import Counter
from typing import List

import datasets
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
from llama_index.core.node_parser import SentenceSplitter
from src.model.retriever import getBM25Retriever
from src.utils.general_utils import make_json_serializable

from llama_index.core import Document
import logging
from typing import Union, List


def write_jsonl(data, path):
    with open(path, 'w') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')


def rank_documents(documents: Union[Document, List[Document]], question: str, chunk_size: int = 256, top_k: int = 3):
    parser = SentenceSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=0)
    if isinstance(documents, str):
        documents = [Document(text=documents)]
    
    elif isinstance(documents, Document):
        documents = [documents]
        
    elif isinstance(documents, list) and isinstance(documents[0], str):
        documents = [Document(text="\n\n".join(documents))]
        
        documents = [Document(text=doc) for doc in documents]
        
    else:
        raise NotImplementedError("Wrong type for documents")
        
    
    assert isinstance(documents, list) and isinstance(documents[0], Document)
    nodes = parser.get_nodes_from_documents(documents)
     
    documents = [Document(doc_id=str(doc_id), text=node.get_content()) for doc_id, node in enumerate(nodes)]
    sparse_retriever, prepare_time = getBM25Retriever(documents, similarity_top_k=len(documents))
    ranked_nodes = sparse_retriever.retrieve(question)
    """
    selected_nodes = ranked_nodes[:top_k]
    selected_node_ids = [node.node.ref_doc_id for node in selected_nodes]
    
    context = [node.get_content() for node in selected_nodes]
    context = "\n".join(context)
    additional_context = [doc.text for doc_id, doc in enumerate(documents) if str(doc_id) not in selected_node_ids]
    additional_context = [node.get_content() for node in ranked_nodes[top_k:]]
    """
    context = [node.get_content() for node in ranked_nodes]
    
    return context

def load_dataset_for_eval(args: Namespace) -> datasets.arrow_dataset.Dataset:
    """
    Each dataset should have 3 features:
    - id
    - example_id (Optional. Reserved for cases where an example has multiple question.)
    - question
    - context: The long evidence document
    - choices: A list of choices for multiple-choice questions
    - answer: The ground truth answer. This is a list for extractive questions and a string for everything else.
    - question_type: The type of question, including "free_form", "multiple-choice", "yes_no", "extractive"


    Args:
        dataset:
        dataset_name:

    Returns:

    """
    
    print("Loading eval dataset...")
    
    # Handle instruction dataset. A local reformatted ``data_file`` is the source of truth (avoids the
    # deprecated HF dataset-script path, e.g. allenai/qasper's qasper.py); return it directly.
    if getattr(args, "data_file", None):
        instruction_dataset = datasets.Dataset.from_list(read_jsonl(args.data_file))
        assert "answer" in instruction_dataset[0], "data_file rows must contain an 'answer' field."
        return instruction_dataset, instruction_dataset

    # Setup cache directory
    cache_dir = os.path.join(args.output_dir, "cache", "dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{args.dataset_name.split('/')[-1]}_{args.split}")

    # return instruction_dataset

    # Load from cache if available
    if os.path.exists(cache_path):
        print(f"Loading dataset from cache: {cache_path}")
        reformat_dataset = datasets.load_from_disk(cache_path)
        assert "answer" in reformat_dataset[0] # and "answer_reformatted" in reformat_dataset[0]
        return reformat_dataset, reformat_dataset
    
    
    rows = []
    
    if args.dataset_name == "THUDM/LongBench":
        dataset = load_dataset(args.dataset_name, args.subset_name, split="test")

        for index_row, row in enumerate(tqdm(dataset, desc=args.dataset_name)):
            
            if index_row >= args.max_eval_samples:
                break
            
            rows.append({
                "id": index_row,
                "example_id": index_row,
                "question": row['input'],
                "context": row['context'],
                "answer": row['answers'],
                
            })
        
                   
        reformat_dataset = datasets.Dataset.from_list(rows)
        return reformat_dataset, reformat_dataset
        
    
    
    elif args.dataset_name == "inscit":
        dataset: datasets.arrow_dataset.Dataset = load_dataset("nvidia/ChatRAG-Bench", args.dataset_name,
                                                               split=args.split)
        reformat_dataset = dataset

    elif args.dataset_name == "emozilla/quality":
        dataset = load_dataset(args.dataset_name, split=args.split)

        answers_reformatted = read_jsonl("data/reformatted/QuALITY_validation.jsonl")
        question2answer_reformatted = {x['question']: x['answer_reformatted'] for x in answers_reformatted}

        for index_row, row in enumerate(tqdm(dataset, desc=args.dataset_name)):
            assert row['answer'] in range(4)
            context = row['article']
            question = row['question']
            answer = row['options'][row['answer']]
            assert len(row['options']) == 4

            rows.append({
                "id": index_row,
                "example_id": index_row,
                "question": question,
                "context": context,
                "choices": row['options'],
                "correct_choice": "ABCD"[row['answer']],
                "answer": answer,
                "answer_reformatted": question2answer_reformatted[question] if question2answer_reformatted[question] is not None else answer,
            })

        reformat_dataset = datasets.Dataset.from_list(rows)

    elif args.dataset_name == "deepmind/narrativeqa":
        dataset = load_dataset(args.dataset_name, split=args.split)
        for index_row, row in enumerate(tqdm(dataset, desc=args.dataset_name)):
            id = f"{len(rows)}"
            question = row['question']['text'].strip().capitalize()
            context = row['document']['text']
            answer = [ans['text'].strip().capitalize() for ans in row['answers']]
            
            if answer:

                rows.append({
                    "id": id,
                    "example_id": id,
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "question_type": "free_form",
                })

        reformat_dataset = datasets.Dataset.from_list(rows)




    elif args.dataset_name == "triviaqa":
        # Clone https://github.com/mandarjoshi90/triviaqa and put it under ../../NLP/triviaqa
        # Adopted from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

        # you need to create a soft-link from the triviaqa directory to `data/TriviaQA`
        TRIVIAQA = os.path.abspath(os.path.join("data", "TriviaQA"))

        """
        """
        path_verified_web_dev = os.path.join(TRIVIAQA, "qa", "verified-web-dev.json")
        dataset = json.load(open(path_verified_web_dev, 'r'))

        for i, entry in enumerate(dataset['Data']):

            context = []
            for search_result in entry['SearchResults']:
                with open(os.path.join(TRIVIAQA, "evidence", "web", search_result['Filename'])) as f:
                    evidence = f.read()

                context.append(evidence)
                
            answer = entry["Answer"]["Aliases"]
            
            if answer:

                datum = {
                    "id": i,
                    "example_id": entry["QuestionId"],
                    "question": entry["Question"],
                    "context": context,
                    "answer": answer,
                    "question_type": "free_form",
                }

                rows.append(datum)

        reformat_dataset = datasets.Dataset.from_list(rows)

    elif args.dataset_name == "rajpurkar/squad_v2":
        dataset = load_dataset(args.dataset_name, split=args.split)
        for index, row in enumerate(tqdm(dataset, desc="SQuADv2")):
            question = row['question']
            context = row['context']

            if not row['answers']['text']:
                answer = []
                answerable = False

            else:
                answer = list(set(row['answers']['text']))  # Note that we allow multiple valid answers
                answerable = True
                
            if answerable: 

                rows.append({
                    "id": str(index),
                    "example_id": str(index),
                    "question": question,
                    "context": [context],
                    "answer": answer,
                    "question_type": "close_qa",
                    "answerable": answerable
                })

        reformat_dataset = datasets.Dataset.from_list(rows)

    elif args.dataset_name == "allenai/qasper":

        dataset = load_dataset(args.dataset_name, split=args.split)
        answers_reformatted = read_jsonl("data/reformatted/QASPER_test.jsonl")
        question2answer_reformatted = {x['question']: x['answer_reformatted'] for x in answers_reformatted}

        for index_row, row in enumerate(tqdm(dataset, desc=args.dataset_name)):

            full_text = get_full_text_qasper(row)
            questions, answers, answerables = [], [], []
            for index_question, question in enumerate(row['qas']["question"]):
                answer, question_type, answerable = extract_answer_and_question_type_qasper(row['qas']['answers'][
                                                                                                index_question][
                                                                                                'answer'][
                                                                                                0])
                # Note: here we only consider answerable questions
                if not answerable:
                    continue

                questions.append(question)
                answers.append(answer)
                
            if len(questions) > 0:

                rows.append({
                    "id": str(index_row),
                    "example_id": str(index_row),
                    "question": questions,
                    "context": full_text,
                    "answer": answers,
                    "answer_reformatted": [question2answer_reformatted[question] for question in questions],
                    "question_type": "close_qa",
                })

        reformat_dataset = datasets.Dataset.from_list(rows)
            
            
    elif args.dataset_name == "Ahren09/hotpotqa":

        dataset = load_dataset(args.dataset_name, split=args.split)
        
        for index, row in enumerate(tqdm(dataset, desc="HotpotQA")):
            question = row['question']
            answer = row['answer']
            context = row['context']
            
            if context:
                assert isinstance(context, list) and isinstance(context[0], str)
                
                context = [re.sub(r"\s+", " ", sent) for sent in context]
                
            else:
                assert context == []
                context = []
                
            rows.append({
                "id": str(index),
                "example_id": str(index),
                "question": question,
                "context": context,
                "answer": answer,
                "question_type": "open_qa",
            })
            
        reformat_dataset = datasets.Dataset.from_list(rows)
            
        
            


    else:
        dataset = load_dataset(args.dataset_name, split=args.split)
        reformat_dataset = dataset
        
    # instruction_dataset_df = instruction_dataset.to_pandas()
    # reformat_dataset_df = reformat_dataset.to_pandas()
    # merged_df = instruction_dataset_df.merge(reformat_dataset_df[['question', 'answer']], on='question', how='left')
    # instruction_dataset = datasets.Dataset.from_pandas(merged_df)


    if rows:  # Only print feature info if rows were manually created
        for feature_name in rows[0].keys():
            print(f"Feature: {feature_name}")
            print(Counter([type(row[feature_name]) for row in rows]))

    from datasets import DatasetDict
    pushed_dataset = DatasetDict({
        "test": reformat_dataset,
    })


    return instruction_dataset, reformat_dataset


def keyword_extraction_with_tfidf(documents, topk=1):
    """
    Documents: List[String]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    ret = []
    for doc_index, doc in enumerate(documents):
        doc_tfidf_scores = tfidf_matrix.toarray()[doc_index]
        keywords_with_scores = {feature_names[col]: doc_tfidf_scores[col] for col in range(len(feature_names))}
        top_keywords = sorted(keywords_with_scores.items(), key=lambda item: item[1], reverse=True)[:topk]

        keywords = []
        for keyword, _ in top_keywords:
            keywords.append(keyword)
        ret.append(" ".join(keywords))

    return ret


def load_dataset_for_rag(data_dir, dataset_name: str, use_rag: bool, args: Namespace):
    dev_data = None
    test_path = f"{data_dir}/eval/{dataset_name}/test.jsonl"
    test_data = None
    if os.path.isfile(test_path):
        test_data = read_jsonl(test_path)

    else:
        raise ValueError(f"Test data not found at {test_path}")

    if use_rag:

        test_retrieval_path = os.path.join(f"data/eval/{dataset_name}/retrieval/{args.retrieval_prefix}", "test.jsonl")
        test_retrieval = read_jsonl(test_retrieval_path)
        assert len(test_retrieval) == len(test_data)
        for idx in range(len(test_data)):
            test_data[idx]['background'] = [test_retrieval[idx]['topk'][rank]['text'] for rank in args.retrieval_topk]

        if args.tf_idf_topk > 0:
            assert args.use_rag
            documents = [x['background'][0] for x in test_data]
            keywords = keyword_extraction_with_tfidf(documents, topk=args.tf_idf_topk)
            for idx in range(len(test_data)):
                test_data[idx]['background'] = [keywords[idx]]

        if args.compressor_name_or_path is not None and args.compressor_name_or_path.lower() == "intfloat/e5-large-v2":
            for idx in range(len(test_data)):
                test_data[idx]['background'] = ["passage: " + x for x in test_data[idx]['background']]

    return dev_data, test_data


def read_jsonl(f, max_lines=None):
    if max_lines is None:
        return [json.loads(x) for x in open(f).readlines()]
    import itertools
    with open(f) as fh:
        return [json.loads(x) for x in itertools.islice(fh, max_lines)]

def read_json(f):
    return json.load(open(f, 'r'))


def reformat_prompt_qasper(question_type):
    suffix = " Keep your answer succinct. The answer is:"

    if question_type == "yes_no":
        suffix = " Please answer \"Yes\" or \"No\" without explanations:"
    elif question_type == "extractive":
        suffix = " Keep your answer succinct. Answer with original sentences in the provided:"

    return suffix


def reformat_dataset_multi_round(dataset: datasets.arrow_dataset.Dataset, args):
    rows = []
    if args.dataset_name == "allenai/qasper":
        for index_row, row in enumerate(tqdm(dataset, desc=args.dataset_name)):
            full_text = get_full_text_qasper(row)
            qa_pairs = []
            for index_question, question in enumerate(row['qas']["question"]):
                answer, question_type, answerable = extract_answer_and_question_type_qasper(row['qas']['answers'][
                                                                                                index_question][
                                                                                                'answer'][
                                                                                                0])
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "question_type": question_type,
                    "answerable": answerable
                })

                # suffix = reformat_prompt_qasper(question_type)

            rows.append({
                "id": f"{index_row}",
                "context": full_text,
                "qa_pairs": qa_pairs
            })

    reformatted_dataset = datasets.Dataset.from_list(rows)
    return reformatted_dataset


def extract_answer_and_question_type_qasper(answer):
    free_form_answer: str = answer['free_form_answer']
    extractive_spans: list = answer['extractive_spans']
    yes_no: str = answer['yes_no']
    answerable: bool = not answer['unanswerable']

    question_type = None

    if extractive_spans:
        assert isinstance(extractive_spans, list)

        question_type = "extractive"
        answer = ", ".join(extractive_spans).strip()
        answer = re.sub(r'\s+', ' ', answer)


    elif yes_no in [True, False]:
        question_type = "yes_no"
        answer = "Yes" if yes_no else "No"


    elif free_form_answer != "":
        question_type = "free_form"
        answer = free_form_answer

    else:
        assert not answerable
        question_type = "unanswerable"
        answer = "Unanswerable"
        # raise ValueError("Question type not found")

    assert question_type is not None
    
    answer = re.sub(r'BIBREF\d+', '', answer)

    return answer, question_type, answerable


def get_full_text_qasper(sample: dict) -> List[str]:
    """
    Gget full-text of the research papers from the QASPER dataset
    :param dict sample: the row sample from QASPER
    """
    title = sample["title"]
    abstract = sample["abstract"]
    sections_list = sample["full_text"]["section_name"]
    paragraph_list = sample["full_text"]["paragraphs"]

    assert len(sections_list) == len(paragraph_list), "Not the same number of sections as paragraphs list"

    combined_sections_with_paras = [title + "\t" + abstract]
    for index in range(0, len(sections_list)):
        tmp = str(sections_list[index]) + "\t"
        if len(paragraph_list[index]) == 0 or len(paragraph_list[index]) == 1 and paragraph_list[index][0] == "":
            continue
        tmp += " ".join(paragraph_list[index])
        tmp = re.sub(r'BIBREF\d+', '', tmp)
        combined_sections_with_paras += [tmp]

    return combined_sections_with_paras


def reformat_generated_results(generated_results, dataset_dict, dataset_name):
    """Merge `answer_reformatted` from the reference dataset into generated results.

    Always drops any pre-existing `answer_reformatted` column and re-merges to ensure
    the column reflects the current reference data.
    """
    df = pd.DataFrame(generated_results)
    if "answer_reformatted" in df.columns:
        df = df.drop(columns=["answer_reformatted"])

    original_length = len(df)
    dataset_df = dataset_dict[dataset_name][['question', 'answer_reformatted']]
    dataset_df = dataset_df.drop_duplicates(subset='question')
    df = df.merge(dataset_df, on='question', how='left')

    assert len(df) == original_length, (
        f"Length mismatch: {len(df)} vs {original_length}. "
        "Some rows in the answer file are not matched to the original dataset."
    )

    answers = df.to_dict(orient="records")
    answers = [make_json_serializable(r) for r in answers]
    return answers
