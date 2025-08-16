#!/usr/bin/env python3
"""
Calculate QA Performance Metrics for a Single File

This script calculates automatic metrics (F1, ROUGE-L, etc.) for a single QA evaluation results file.
"""

import json
import os
import re
import sys
import numpy as np
from tqdm.contrib import tzip

# Add the src directory to the path

from src.metrics.metrics import rougel_score, compute_f1
from src.utils.data_utils import read_jsonl

def clean_answer(answer):
    """Clean answer text by removing response markers."""
    if "---Response---\n\n" in answer:
        answer = answer.split("---Response---\n\n")[1]
    if "---Answer---\n\n" in answer:
        answer = answer.split("---Answer---\n\n")[1]
    return answer

def calculate_qa_single_file(generated_results):
    """
    Calculate QA performance metrics including F1, ROUGE-L, and entity matching.
    Modified to handle cases where answer_reformatted is not present.
    
    Args:
        generated_results (list): List of generated results with predictions and answers
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    answers, answers_reformatted, generates = [], [], []
    answerables = []

    substrings_to_remove = ["<unk>", "<pad>", "</s>", "<s>"]

    for generated in generated_results:
        predicted = generated['pred']
        # Use a regular expression to replace all in one step
        predicted = re.sub("|".join(map(re.escape, substrings_to_remove)), "", predicted).strip()
        predicted = re.sub(r'BIBREF\d+', '', predicted)

        if not isinstance(generated['answer'], (list, str)):
            print(f"Ground-truth answer must be a string or a list, but got {generated['answer']}")
            continue

        if isinstance(generated['answer'], str):
            answer = re.sub(r'BIBREF\d+', '', generated['answer'])
            # If answer_reformatted doesn't exist, use the original answer
            if 'answer_reformatted' not in generated or generated['answer_reformatted'] is None:
                answer_reformatted = answer
            elif isinstance(generated['answer_reformatted'], list):
                answer_reformatted = generated['answer_reformatted'][0]
            else:
                answer_reformatted = re.sub(r'BIBREF\d+', '', generated['answer_reformatted'])
        elif isinstance(generated['answer'], list):
            answer = [re.sub(r'BIBREF\d+', '', ans) for ans in generated['answer'] if ans is not None]
            if 'answer_reformatted' not in generated or generated['answer_reformatted'] is None:
                answer_reformatted = answer
            else:
                answer_reformatted = [re.sub(r'BIBREF\d+', '', ans) for i, ans in enumerate(generated['answer_reformatted']) if ans is not None]
        else:
            answer = answer_reformatted = None

        if answer in [None, "", []]:
            answerable = False
        else:
            answerable = True

        answers.append(answer)
        answers_reformatted.append(answer_reformatted)
        generates.append(predicted)
        answerables.append(answerable)

    print(f"We expect the model to answer {sum(answerables)}/{len(answerables)} questions")

    # Filter out non-answerable questions
    answers = [ans for ans, answerable in zip(answers, answerables) if answerable]
    answers_reformatted = [ans for ans, answerable in zip(answers_reformatted, answerables) if answerable]
    generates = [g for g, answerable in zip(generates, answerables) if answerable]

    assert len(answers) == len(generates) == sum(answerables)

    precision_list, recall_list = [], []
    f1_score_list, bleu_score_list = [], []
    f1_score_list_ori = []  # Using the original answer without reformatting
    entity_match_list = []

    # Calculate F1 scores
    for pred, ans, ans_reformat in tzip(generates, answers, answers_reformatted, desc="F1"):
        assert ans is not None and ans_reformat is not None, "Answer cannot be None"
        pred = clean_answer(pred)

        # Evaluate w.r.t. the reformatted answer
        if isinstance(ans, str):
            assert isinstance(ans_reformat, str), f"Answer reformatted must be a string, but got {type(ans_reformat)}"
            p, r, f1 = compute_f1(pred, ans_reformat)
            entity_match = ans in pred
            _, _, f1_ori = compute_f1(pred, ans)

        elif isinstance(ans, list):
            assert isinstance(ans_reformat, list), f"Answer reformatted must be a list, but got {type(ans_reformat)}"
            assert len(ans) > 0, f"Answer list is empty: {ans}"
            p, r, f1 = [], [], []
            f1_ori = []
            for a, a_reformat in zip(ans, ans_reformat):
                p_, r_, f1_ = compute_f1(pred, a_reformat)
                p.append(p_)
                r.append(r_)
                f1.append(f1_)
                _, _, f1_ori_ = compute_f1(pred, a)
                f1_ori.append(f1_ori_)

            if not p:
                continue
            p, r, f1 = max(p), max(r), max(f1)
            f1_ori = max(f1_ori)
            entity_match = any([a in pred for a in ans])
        else:
            raise ValueError(f"Unexpected type for ans: {type(ans)}")

        precision_list.append(p)
        recall_list.append(r)
        f1_score_list.append(f1)
        f1_score_list_ori.append(f1_ori)
        entity_match_list.append(entity_match)

    # Calculate ROUGE-L scores
    rougel_list, rougel_list_ori = [], []
    for g, a, a_reformat in tzip(generates, answers, answers_reformatted, desc="Rouge-L"):
        rougel_list.append(rougel_score(g, a_reformat))
        rougel_list_ori.append(rougel_score(g, a))

    # Calculate final metrics
    rougel = np.array([s for s in rougel_list if s is not None])
    rougel_ori = np.array([s for s in rougel_list_ori if s is not None])
    f1_score_list = np.array(f1_score_list)
    f1_score_list_ori = np.array(f1_score_list_ori)

    print(f"Precision: {np.mean(precision_list) * 100:.2f} +/- {np.std(precision_list) * 100:.2f}")
    print(f"Recall: {np.mean(recall_list) * 100:.2f} +/- {np.std(recall_list) * 100:.2f}")
    print(f"F1 score: {np.mean(f1_score_list) * 100:.2f} +/- {np.std(f1_score_list) * 100:.2f} | Original F1: {np.mean(f1_score_list_ori) * 100:.2f} +/- {np.std(f1_score_list_ori) * 100:.2f}")
    print(f"Entity match: {np.mean(entity_match_list) * 100:.2f}")
    print(f"Rouge-L: {np.mean(rougel) * 100:.2f} | Ori Rouge-L: {np.mean(rougel_ori) * 100:.2f}")

    return {
        "f1": round(np.mean(f1_score_list) * 100, 2),
        "rougel": round(np.mean(rougel) * 100, 2),
        "f1_ori": round(np.mean(f1_score_list_ori) * 100, 2),
        "rougel_ori": round(np.mean(rougel_ori) * 100, 2),
        "EM": round(np.mean(entity_match_list) * 100, 2),
    }

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python calculate_single_file_performance.py <path_to_jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    print(f"Calculating performance for: {file_path}")
    print("="*60)
    
    # Read the JSONL file
    generated_results = read_jsonl(file_path)
    print(f"Loaded {len(generated_results)} examples")
    
    # Calculate metrics
    metrics = calculate_qa_single_file(generated_results)
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
