import asyncio
import os
import sys
import traceback
from collections import defaultdict

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ResponseRelevancy, FactualCorrectness, SemanticSimilarity, BleuScore, Faithfulness, \
    LLMContextRecall
from tqdm import tqdm



from src.arguments import parse_args, set_attributes_from_yaml
from src.eval.calculate_qa_performance import reformat_generated_results
from src.utils import read_jsonl, print_file_timestamps, get_yaml_file, load_dataset_for_eval


async def calculate_ragas_metrics(dataset_name, data):

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )
    evaluator_embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings_wrapper = LangchainEmbeddingsWrapper(evaluator_embedding)

    scorer_response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings_wrapper)
    scorer_answer_correctness = FactualCorrectness(llm=evaluator_llm)
    scorer_semantic_similarity = SemanticSimilarity(embeddings=embeddings_wrapper)

    scorer_context_recall = LLMContextRecall(llm=evaluator_llm)

    scorer_faithfulness = Faithfulness(llm=evaluator_llm)

    question_col = "question"
    pred_col = "pred"
    if "longbench" in dataset_name:
        reference_col = "answer"
    else:
        reference_col = "answer_reformatted"  # "answer"
    context_col = "context"

    metrics = ["answer_relevance", "answer_correctness", "semantic_similarity", "faithfulness", "context_recall",
               "bleu", "rouge"]

    metric_sums = defaultdict(float)

    total_items = 0

    for idx, item in enumerate(tqdm(data, desc="Processed", total=len(data)), start=1):
        if idx >= 30:
            break

        q_number = item.get("question_number", str(idx))

        question = item.get(question_col, "")
        reference = item.get(reference_col, "")
        pred = item.get(pred_col, "")
        context = item.get(context_col, "")
        additional_context = item.get("additional_context", None)

        if not pred or not reference:
            continue

        if isinstance(reference, list):
            # reference should be a str
            reference = reference[0]
        if not isinstance(reference, str):
            continue

        if isinstance(context, str):
            # context should be a list
            context = [context]

        if additional_context:
            if isinstance(additional_context, str):
                additional_context = [additional_context]
            context += additional_context

        if not context:
            continue

        sample = SingleTurnSample(
            user_input=question,
            response=pred,  # Expected answer from the model
            reference=reference,
            retrieved_contexts=context
        )

        try:

            if "answer_relevance" in metrics:
                relevancy_score = await scorer_response_relevancy.single_turn_ascore(sample)

                metric_sums["answer_relevance"] += relevancy_score

            if "answer_correctness" in metrics:
                correctness_score = await scorer_answer_correctness.single_turn_ascore(sample)

                metric_sums["answer_correctness"] += correctness_score

            if "faithfulness" in metrics:
                faithfulness_score = await scorer_faithfulness.single_turn_ascore(sample)

                metric_sums["faithfulness"] += faithfulness_score

            if "semantic_similarity" in metrics:
                semsim_score = await scorer_semantic_similarity.single_turn_ascore(sample)

                metric_sums["semantic_similarity"] += semsim_score

            if "context_recall" in metrics:
                context_recall_score = await scorer_context_recall.single_turn_ascore(sample)

                metric_sums["context_recall"] += context_recall_score

            total_items += 1
        except Exception as e:
            print(f"Error processing item {q_number}")
            traceback.print_exc()
            continue

    final_scores = {}
    for metric in metrics:
        if metric in metric_sums:
            final_scores[metric] = np.round(metric_sums[metric] / total_items * 100, 2)

    return final_scores

if __name__ == "__main__":

    args = parse_args('metrics')
    # Run the async main function

    dataset_dict = {}

    # for dataset_name in [
    #     "qasper",
    #     "narrativeqa",
    #     "triviaqa",
    #     "quality",
    #     "SQuAD-v2",
    #     "hotpotqa"
    # ]:
    #     print(f"Loading dataset: {dataset_name}")
    #     dataset_config = get_yaml_file(os.path.join("config", "dataset", f"{dataset_name}.yaml"))
    #     args = set_attributes_from_yaml(args, dataset_config, overwrite=True)
    #     dataset, original_dataset = load_dataset_for_eval(args)
    #     dataset_dict[dataset_name] = dataset.to_pandas()

    for dataset_name in [
        # "qasper",
        # "triviaqa",

        # "narrativeqa",

        # "quality",
        # "SQuAD-v2",
        # "hotpotqa",
        
        # "longbench-qmsum",
        # "longbench-2wikimqa",
        "longbench-multifieldqa_en",

    ]:
        performance_summary = []
        for model_name in [
            "Mistral7B_QA_Compress_Stage0"
        ]:
            if "Mistral7B_QA_Compress" in model_name:
                target = None
                output_file = (f"{args.output_dir}/{model_name}/answers_{model_name}_bm25_"
                               f"{dataset_name}_num-evi5_num-add5_evi-select-self-info.jsonl")

            else:
                continue

            if os.path.exists(os.path.join(args.output_dir, "qa",
                                           f"ragas_{os.path.basename(output_file).split('.jsonl')[0]}.xlsx")):
                print(f"Already evaluated {output_file}")
                continue
            if os.path.exists(output_file):
                created, modified = print_file_timestamps(output_file)
                generated_results = read_jsonl(output_file)
                generated_results = reformat_generated_results(generated_results, dataset_dict, dataset_name)
                performance = {
                    "dataset_name": dataset_name,

                    "model_name": model_name,
                    "target_length": target,
                    "modified": f"{modified.strftime('%Y-%m-%d %H:%M:%S')}",
                }
                res = asyncio.run(calculate_ragas_metrics(generated_results))
                performance.update(res)

                performance_summary = pd.Series(performance)
                print(performance_summary)
                performance_summary.to_excel(os.path.join(args.output_dir, "qa",
                                                          f"ragas_{os.path.basename(output_file).split('.jsonl')[0]}.xlsx"),
                                             sheet_name=dataset_name, index=False)
        for model_name in [
            # "Mistral7B_QA_Compress_Stage0_Linq",
            # "Mistral7B_QA_Compress_Stage0",
            # "MistralSmall_QA_Compress_Stage0",
            # "MistralNemo_QA_Compress_Stage0",
            # "MistralNemo",
            # "Mistral7B",
            # "MistralSmall",
            # "Llama-3.1-8B-Instruct"
            # "Llama-3.1-8B-Instruct_QA_Compress_Stage0"
            # "Gemma3-4B",
            # "Gemma3-27B",
            "Mistral7B",
            "Mistral7B_QA_Compress_Stage0"

        ]:
            # for retriever_type in ["bm25", "BAAI/bge-reranker-v2-m3", "SFR-Embedding-Mistral"]:
            for retriever_type in ["bm25"]:
                for k in [10]:
                    for n in [10, 15]:
                        # for evidence_selection in ["self-info", "embed", None]:
                        for evidence_selection in ["self-info", "embed", None]:
                            if evidence_selection:
                                # output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{retriever_type.split('/')[-1]}_{dataset_name}_num-evi{num_evidence}_num-add{num_additional_evidence}_evi-select-{evidence_selection}.jsonl"
                                output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{retriever_type.split('/')[-1]}_{dataset_name}_k{k}_n{n}_evi-select-{evidence_selection}.jsonl"
                            else:
                                # output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{dataset_name}_num-evi{num_evidence}_num-add{num_additional_evidence}.jsonl"
                                output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{retriever_type.split('/')[-1]}_{dataset_name}_k{k}_n{n}.jsonl"


                            if os.path.exists(os.path.join(args.output_dir, "qa", f"ragas_{os.path.basename(output_file).split('.jsonl')[0]}.xlsx")):
                                print(f"Already evaluated {output_file}")
                                continue


                            if os.path.exists(output_file):
                                created, modified = print_file_timestamps(output_file)
                                generated_results = read_jsonl(output_file)
                                
                                if "answer_reformatted" not in generated_results[0]:
                                    generated_results = reformat_generated_results(generated_results,
                                                                                    dataset_dict, dataset_name)
                                        
                                
                                
                                performance = {
                                    "dataset_name": dataset_name,
                                    "retriever_type": retriever_type,
                                    "model_name": model_name,
                                    "k": k,
                                    "n": n,
                                    "evi-select": evidence_selection,
                                    "modified": f"{modified.strftime('%Y-%m-%d %H:%M:%S')}",
                                }
                                res = asyncio.run(calculate_ragas_metrics(dataset_name, generated_results))
                                performance.update(res)


                                performance_summary = pd.Series(performance)
                                print(performance_summary)
                                performance_summary.to_excel(os.path.join(args.output_dir, "qa", f"ragas_{os.path.basename(output_file).split('.jsonl')[0]}.xlsx"), sheet_name=dataset_name, index=False)
    