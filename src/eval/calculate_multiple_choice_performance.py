import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.utils import project_setup, get_yaml_file


from src.arguments import parse_args, set_attributes_from_yaml
from src.utils import print_file_timestamps

from src.utils.data_utils import (
    read_jsonl, load_dataset_for_eval, reformat_generated_results
)
from src.utils.general_utils import make_json_serializable


def clean_answer(answer):
    if "answer is" in answer:
        answer = answer.split("answer is")[-1].strip()
    if not answer:
        return None

    if len(answer) == 1 and answer[0] in "ABCD":
        return answer[0]

    if answer[0] in "ABCD":
        return answer[0]
    print(f"Invalid answer: {answer}")
    return None


def calculate_qa(generated_results):
    df = pd.DataFrame(generated_results)

    # Need to mask out the experiments where the model is not expected to answer
    preds_clean, answers_clean = [], []
    for index_row, row in df.iterrows():
        answer = row['correct_choice']
        pred = clean_answer(row['pred'])
        preds_clean.append(pred)
        answers_clean.append(answer)

    count_pred = Counter(preds_clean)
    print(f"#Invalid answers: {count_pred[None]}")
    mask = np.array(preds_clean) != None
    preds_clean = np.array(preds_clean)[mask]
    answers_clean = np.array(answers_clean)[mask]
    # p, r, f1, _ = precision_recall_fscore_support(answers_clean, preds_clean)
    acc = float(np.mean(preds_clean == answers_clean))

    # print(f"Pre: {p * 100:.2f} | Rec: {r * 100:.2f} | F1: {f1 * 100:.2f} | Acc: {acc * 100:.2f}")
    print(f"Acc: {acc * 100:.2f}")

    return {
        "accuracy": round(acc * 100, 2),
    }


if __name__ == "__main__":
    project_setup()
    args = parse_args('metrics')

    results_path = f"{args.output_dir}/qa/performance_summary_multi-choice.xlsx"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Load the dataset (raw / instruction formats)
    dataset_dict = {}

    for dataset_name in [
        "quality",
    ]:
        print(f"Loading dataset: {dataset_name}")
        dataset_config = get_yaml_file(os.path.join("config", "dataset", f"{dataset_name}.yaml"))
        args = set_attributes_from_yaml(args, dataset_config, overwrite=True)
        dataset = load_dataset_for_eval(args)
        dataset_dict[dataset_name] = dataset.to_pandas()

    args.dataset_name = None

    with pd.ExcelWriter(results_path) as writer:


        for dataset_name in [
            "quality"
        ]:

            performance_summary = []
            for model_name in [
                "Mistral7B_QA_Compress_Stage0_QuALITY",
                "Mistral7B",
                "Llama-3.1-8B-Instruct",
                "Llama-3.1-8B-Instruct_QA_Compress_Stage0_QuALITY",
                # "Llama-3.1-8B-Instruct_QA_Compress_Stage0",
                # "Llama-3.1-8B-Instruct_QA_Compress_Stage2",
            ]:
                for RETRIEVER_NAME_OR_PATH in ["bm25", "BAAI/bge-reranker-v2-m3"]:
                    for k in [5]:
                        for n in [5, 10]:
                            for evidence_selection in [None, "self-info", "embed"]:
                                for repetition_penalty in [1.5]:

                                    if evidence_selection:
                                        output_file = (f"{args.output_dir}/{model_name}/answers_{model_name}_"
                                                       f"{RETRIEVER_NAME_OR_PATH.split("/")[-1]}_"
                                                       f"{dataset_name}_multi-choice_k"
                                                       f"{k}_n{n}_rep{repetition_penalty}_evi-select-{evidence_selection}.jsonl")

                                    else:
                                        # output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{dataset_name}_num-evi{num_evidence}_num-add{num_additional_evidence}.jsonl"
                                        output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{RETRIEVER_NAME_OR_PATH.split("/")[-1]}_{dataset_name}_multi-choice_k{k}_n{n}_rep{repetition_penalty}.jsonl"

                                    if os.path.exists(output_file):
                                        created, modified = print_file_timestamps(output_file)
                                        generated_results = read_jsonl(output_file)

                                        performance = {
                                            "dataset_name": dataset_name,
                                            "RETRIEVER_NAME_OR_PATH": RETRIEVER_NAME_OR_PATH,
                                            "model_name": model_name,
                                            "k": k,
                                            "n": n,
                                            "evi-select": evidence_selection,
                                            "repetition_penalty": repetition_penalty,
                                            "modified": f"{modified.strftime('%Y-%m-%d %H:%M:%S')}",
                                        }
                                        performance.update(calculate_qa(generated_results))

                                        performance_summary.append(performance)

            performance_summary = pd.DataFrame(performance_summary).sort_values(
                ["dataset_name", "RETRIEVER_NAME_OR_PATH", "model_name", "k",
                 "n", ]).reset_index(drop=True)
            print(performance_summary)
            performance_summary.to_excel(writer, sheet_name=dataset_name, index=False)
