"""
Calculate QA Performance Metrics

This module calculates automatic metrics (F1, ROUGE-L, etc.) for QA evaluation results.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import sacrebleu
from tqdm.contrib import tzip
from evaluate import load


from src.metrics.metrics_instructrag import compute_str_em, exact_presence
from src.utils import project_setup, get_yaml_file

from src.arguments import parse_args, set_attributes_from_yaml
from src.utils import print_file_timestamps

from src.metrics.metrics import rougel_score
from src.utils.data_utils import (
    read_jsonl, load_dataset_for_eval, write_jsonl, read_json, reformat_generated_results
)
from src.utils.general_utils import make_json_serializable

from src.metrics.metrics import compute_f1


# ===== [Utility Functions] =====
def clean_answer(answer):
    """Clean answer text by removing response markers."""
    if "---Response---\n\n" in answer:
        answer = answer.split("---Response---\n\n")[1]
    if "---Answer---\n\n" in answer:
        answer = answer.split("---Answer---\n\n")[1]
    return answer


# ===== [Metrics Calculation] =====
def calculate_qa(generated_results):
    """
    Calculate QA performance metrics including F1, ROUGE-L, and entity matching.
    
    Args:
        generated_results (list): List of generated results with predictions and answers
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    answers, answers_reformatted, generates = [], [], []
    answers_yn, generates_yn = [], []
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
            if generated['answer_reformatted'] is None or generated['answer_reformatted'] == '':
                answer_reformatted = answer
            elif isinstance(generated['answer_reformatted'], list):
                answer_reformatted = generated['answer_reformatted'][0]
            else:
                answer_reformatted = re.sub(r'BIBREF\d+', '', generated['answer_reformatted'])
        elif isinstance(generated['answer'], list):
            answer = [re.sub(r'BIBREF\d+', '', ans) for ans in generated['answer'] if ans is not None]
            if 'answer_reformatted' not in generated or generated['answer_reformatted'] is None or generated['answer_reformatted'] == '':
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
    scorer = sacrebleu.metrics.BLEU(force=True)

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


def _per_example_f1(rows):
    """Per-row token-F1 (max over reference answers), keyed by (id, question_id). Reuses compute_f1."""
    out = {}
    meta = {}
    for r in rows:
        golds = r.get("answer_reformatted") or r.get("answer")
        golds = golds if isinstance(golds, list) else [golds]
        pred = clean_answer(r.get("pred", "") or "")
        best = max((compute_f1(clean_answer(str(g)), pred)[2] for g in golds if g is not None), default=0.0)
        key = (r["id"], r["question_id"])
        out[key] = best * 100.0
        meta[key] = r.get("example_id", r["id"])
    return out, meta


def _cluster_bootstrap(diff_by_key, doc_by_key, n_boot=10000, seed=42):
    """Document-clustered paired bootstrap over per-example F1 differences (release plan §11)."""
    import numpy as _np
    docs = {}
    for key, d in diff_by_key.items():
        docs.setdefault(doc_by_key[key], []).append(d)
    doc_ids = list(docs.keys())
    doc_means = {doc: _np.mean(v) for doc, v in docs.items()}
    doc_counts = {doc: len(v) for doc, v in docs.items()}
    rng = _np.random.default_rng(seed)
    idx = _np.arange(len(doc_ids))
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        num = sum(doc_means[doc_ids[i]] * doc_counts[doc_ids[i]] for i in sample)
        den = sum(doc_counts[doc_ids[i]] for i in sample)
        boots.append(num / den if den else 0.0)
    boots = _np.array(boots)
    overall = _np.mean([d for d in diff_by_key.values()])
    return {
        "mean_diff_f1": round(float(overall), 3),
        "ci95_low": round(float(_np.percentile(boots, 2.5)), 3),
        "ci95_high": round(float(_np.percentile(boots, 97.5)), 3),
        "n_bootstrap": n_boot, "seed": seed,
        "n_documents": len(doc_ids), "n_questions": len(diff_by_key),
        "bootstrap_unit": "document",
    }


def build_comparison_report(rag_file, sara_file, out_dir, exp_id="sara_vs_rag_qasper",
                            target_improvement=5.0, hardware=None):
    """Produce the two-method comparison report + results table (release plan §13).

    Reuses calculate_qa for aggregate F1/ROUGE-L and per-example F1 for the document-clustered bootstrap.
    Reads each prediction file's sibling ``.meta.json`` for batch size / throughput / peak GPU memory.
    """
    def _load(path):
        rows = [json.loads(l) for l in open(path)]
        meta = {}
        if os.path.exists(path + ".meta.json"):
            meta = json.load(open(path + ".meta.json"))
        return rows, meta

    rag_rows, rag_meta = _load(rag_file)
    sara_rows, sara_meta = _load(sara_file)
    rag_agg, sara_agg = calculate_qa(rag_rows), calculate_qa(sara_rows)

    rag_f1, rag_doc = _per_example_f1(rag_rows)
    sara_f1, sara_doc = _per_example_f1(sara_rows)
    shared = sorted(set(rag_f1) & set(sara_f1))
    diff = {k: sara_f1[k] - rag_f1[k] for k in shared}
    doc_by_key = {k: sara_doc.get(k, rag_doc.get(k)) for k in shared}
    boot = _cluster_bootstrap(diff, doc_by_key) if shared else {}

    def _docs(rows):
        ns = [r.get("n") for r in rows if r.get("n") is not None]
        return ns[0] if ns else None

    def _mean_in_tok(rows, meta):
        if meta.get("mean_input_tokens") is not None:
            return meta["mean_input_tokens"]
        v = [r.get("input_tokens", 0) for r in rows]
        return round(sum(v) / max(1, len(v)), 1)

    improvement = round(sara_agg["f1"] - rag_agg["f1"], 2)
    rows_table = [
        ("Standard RAG", rag_agg, rag_rows, rag_meta),
        ("SARA", sara_agg, sara_rows, sara_meta),
    ]
    results = {
        "exp_id": exp_id, "rag_file": rag_file, "sara_file": sara_file,
        "rag": rag_agg, "sara": sara_agg,
        "absolute_f1_improvement": improvement,
        "target_improvement": target_improvement,
        "performance_gate_pass": bool(improvement >= target_improvement),
        "paired_cluster_bootstrap": boot,
        "rag_meta": rag_meta, "sara_meta": sara_meta, "hardware": hardware,
        "n_eval_examples_rag": len(rag_rows), "n_eval_examples_sara": len(sara_rows),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(make_json_serializable(results), f, indent=2)

    def cell(method, agg, rows, meta):
        return (f"| {method} | {agg['f1']:.2f} | {agg['rougel']:.2f} | "
                f"{_mean_in_tok(rows, meta)} | {_docs(rows)} | {meta.get('batch_size','-')} | "
                f"{meta.get('generated_tokens_per_second','-')} | {meta.get('peak_gpu_mem_mb','-')} |")

    lines = [
        f"# Standard RAG vs SARA — QASPER ({exp_id})", "",
        "| Method | QASPER F1 | ROUGE-L | Input tokens | Retrieved docs (n) | Batch size | Throughput (tok/s) | Peak GPU mem (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        cell(*rows_table[0]), cell(*rows_table[1]), "",
        f"**Absolute F1 improvement (SARA − RAG): {improvement:+.2f}** "
        f"(target ≥ {target_improvement} → {'PASS' if results['performance_gate_pass'] else 'NOT MET'})", "",
    ]
    if boot:
        lines += [
            f"**Paired document-clustered bootstrap** (unit=document, {boot['n_documents']} docs / "
            f"{boot['n_questions']} questions, {boot['n_bootstrap']} samples, seed {boot['seed']}): "
            f"mean ΔF1 = {boot['mean_diff_f1']:+.3f}, 95% CI [{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}]", "",
        ]
    lines += [
        f"- Eval examples: RAG {len(rag_rows)}, SARA {len(sara_rows)}",
        f"- RAG meta: {rag_meta}", f"- SARA meta: {sara_meta}",
    ]
    if hardware:
        lines += [f"- Hardware: {hardware}"]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out_dir}/results.json and report.md | ΔF1={improvement:+.2f} "
          f"| gate {'PASS' if results['performance_gate_pass'] else 'NOT MET'}")
    return results


def main(dataset_dict):
    """Main execution function for calculating QA performance metrics."""
    project_setup()

    results_path = f"{args.output_dir}/qa/performance_summary.xlsx"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(results_path):
        try:
            with pd.ExcelFile(results_path) as xls:
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(results_path, sheet_name=sheet_name)
                    existing_results[sheet_name] = df
            print(f"✓ Loaded existing results from {results_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load existing results: {e}")
            existing_results = {}

    args.dataset_name = None

    with pd.ExcelWriter(results_path) as writer:
        for dataset_name in datasets_to_process:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            # Load existing performance summary for this dataset
            if dataset_name in existing_results:
                performance_summary = existing_results[dataset_name].to_dict('records')
                print(f"✓ Loaded {len(performance_summary)} existing results for {dataset_name}")
            else:
                performance_summary = []
                print(f"✓ Starting fresh results for {dataset_name}")
            
            for model_name in ["Mistral7B", "Mistral7B_QA_Compress_Stage0"]:
                for n in range(1, 11):  # Total evidence pieces
                    for k in range(1, n + 1):
                        # Calculate metrics for k varying from 1 to 10
                        for retriever_name_or_path in ["bm25"]:
                            for evidence_selection in ["self-info", None]:
                                # Check if results already exist for this combination
                                existing_result = None
                                for result in performance_summary:
                                    if (result.get('dataset_name') == dataset_name and
                                        result.get('retriever_name_or_path') == retriever_name_or_path and
                                        result.get('model_name') == model_name and
                                        result.get('k') == k and
                                        result.get('n') == n and
                                        result.get('evi-select') == evidence_selection):
                                        existing_result = result
                                        break

                                if model_name in ["Mistral7B"]:
                                    evidence_selection = None
                                
                                if existing_result is not None:
                                    print(f"⏭ Skipping k={k}: Results already exist (F1={existing_result.get('f1', 'N/A')}, ROUGE-L={existing_result.get('rougel', 'N/A')})")
                                    continue
                                
                                if evidence_selection is None:
                                    output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{retriever_name_or_path.split('/')[-1]}_{dataset_name}_k{k}_n{n}.jsonl"

                                else:
                                    # Construct output file path
                                    output_file = f"{args.output_dir}/{model_name}/answers_{model_name}_{retriever_name_or_path.split('/')[-1]}_{dataset_name}_k{k}_n{n}_evi-select-{evidence_selection}.jsonl"
                                
                                print(f"Checking file: {output_file}")
                                
                                if os.path.exists(output_file):
                                    created, modified = print_file_timestamps(output_file)
                                    generated_results = read_jsonl(output_file)

                                    # Reformat results if needed
                                    if "answer_reformatted" not in generated_results[0]:
                                        generated_results = reformat_generated_results(generated_results, dataset_dict, dataset_name)
                                        write_jsonl(generated_results, output_file)

                                    performance = {
                                        "dataset_name": dataset_name,
                                        "retriever_name_or_path": retriever_name_or_path,
                                        "model_name": model_name,
                                        "k": k,
                                        "n": n,
                                        "evi-select": evidence_selection,
                                        "modified": f"{modified.strftime('%Y-%m-%d %H:%M:%S')}",
                                    }
                                    
                                    # Calculate metrics
                                    metrics = calculate_qa(generated_results)
                                    performance.update(metrics)
                                    performance_summary.append(performance)
                                    
                                    print(f"✓ k={k}: F1={metrics['f1']:.2f}, ROUGE-L={metrics['rougel']:.2f}")
                                else:
                                    print(f"⚠ File not found: {output_file}")

            # Create DataFrame and sort
            performance_summary = pd.DataFrame(performance_summary)
            try:
                performance_summary = performance_summary.sort_values(["dataset_name", "model_name", "retriever_name_or_path", "k", "n"]).reset_index(drop=True)
            except:
                pass

            performance_summary = performance_summary.reset_index(drop=True)
            print(f"\nPerformance Summary for {dataset_name}:")
            print(performance_summary)
            
            # Save to Excel
            performance_summary.to_excel(writer, sheet_name=dataset_name, index=False)
            print(f"✓ Saved results to {results_path}")


def calculate_performance_given_filename(dataset_dict, filename):
    generated_results = read_jsonl(filename)
    if "answer_reformatted" not in generated_results[0]:
        generated_results = reformat_generated_results(generated_results, dataset_dict, dataset_name)
    performance = calculate_qa(generated_results)
    return performance

if __name__ == "__main__":
    # Load the dataset (raw / instruction formats)
    dataset_dict = {}
    datasets_to_process = [
        "hotpotqa",
        "qasper",
        "narrativeqa", 
        "triviaqa",
        "quality",
        "SQuAD-v2",
        
    ]
    args = parse_args('metrics')

    for dataset_name in datasets_to_process:
        print(f"Loading dataset: {dataset_name}")
        dataset_config = get_yaml_file(os.path.join("config", "dataset", f"{dataset_name}.yaml"))
        args = set_attributes_from_yaml(args, dataset_config, overwrite=True)
        dataset, original_dataset = load_dataset_for_eval(args)
        dataset_dict[dataset_name] = dataset.to_pandas()
    main(dataset_dict)
    # calculate_performance_given_filename(dataset_dict, "outputs/CompAct/answers_CompAct_mistral-7b-instruct-v0.2_qasper_num-evi5.jsonl")
    # calculate_performance_given_filename(dataset_dict, "outputs/Beacon/answers_hotpotqa.jsonl")
