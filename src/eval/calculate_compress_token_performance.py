import math
import os
import re
import sys

import nltk
import pandas as pd
import seaborn as sns
import spacy
import torch
from nltk.translate.bleu_score import sentence_bleu
from rapidfuzz import fuzz, process
from rouge_score import rouge_scorer
from scipy.stats import linregress
from sympy.physics.control.control_plots import plt
from tqdm import trange
from transformers import AutoTokenizer, AutoModel


from src.arguments import parse_args
from src.utils.general_utils import project_setup
from src.utils.general_utils import print_file_timestamps
from src.utils.text_utils import extract_entities_and_numerics


# Need to download the punkt tokenizer for nltk
# nltk.download('punkt')

# Function to calculate BLEU score for a batch
def calculate_bleu_batch(references, predictions):
    bleu_scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = nltk.word_tokenize(ref)

        if not isinstance(pred, str) and math.isnan(pred):
            bleu_scores.append(0.)

        else:
            pred_tokens = nltk.word_tokenize(pred)
            bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens))

    return bleu_scores


# Function to calculate ROUGE scores for a batch
def calculate_rouge_batch(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for ref, pred in zip(references, predictions):

        if not isinstance(pred, str) and math.isnan(pred):
            rouge1.append(0.)
            rouge2.append(0.)
            rougeL.append(0.)

        else:
            scores = scorer.score(ref, pred)
            rouge1.append(scores['rouge1'].fmeasure)
            rouge2.append(scores['rouge2'].fmeasure)
            rougeL.append(scores['rougeL'].fmeasure)

    return rouge1, rouge2, rougeL


# Function to calculate cosine similarity using BERT embeddings for a batch
def calculate_cosine_similarity_batch(tokenizer, texts1, texts2, device):
    tokens1 = tokenizer(texts1, padding=True, truncation=True, return_tensors='pt').to(device)
    tokens2 = tokenizer(texts2, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu().tolist()
    return similarities


def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))

    if union == 0:
        assert intersection == 0
        return None

    return intersection / union


# Main function to evaluate a batch
def evaluate_batch(batch, tokenizer, nlp, device):
    references = batch['references'].tolist()
    predictions = batch['predictions'].tolist()

    avg_entity_match_scores = []
    avg_numeric_jacc = []
    word_count_ref, word_count_pred = [], []

    for ref, pred in zip(references, predictions):
        word_count_ref.append(len(ref.split()))
        word_count_pred.append(len(pred.split()))

        entities_ref, numerics_ref = extract_entities_and_numerics(nlp, ref)
        entities_pred, numerics_pred = extract_entities_and_numerics(nlp, pred)
        match_di = {}
        for pred in entities_pred:
            # Find the best match in the reference set
            best_match = process.extractOne(pred, entities_ref, scorer=fuzz.ratio)
            match_di[pred] = best_match

        # Display the matches
        match_scores = []
        for pred, match in match_di.items():
            if match is not None:
                match_scores.append(match[1])
            else:
                match_scores.append(0.)

                # print(f"Matched: {pred} -> {match[0]} ({match[1]}%)")

        jacc = jaccard_similarity(numerics_ref, numerics_pred)

        avg_entity_match_scores.append(sum(match_scores) / len(match_scores) if match_scores else 0.)
        avg_numeric_jacc.append(jacc)

    bleu_scores = calculate_bleu_batch(references, predictions)
    rouge1, rouge2, rougeL = calculate_rouge_batch(references, predictions)
    cosine_similarities = calculate_cosine_similarity_batch(tokenizer, references, predictions, device)

    return pd.DataFrame({
        'BLEU Score': bleu_scores,
        'ROUGE-1': rouge1,
        'ROUGE-2': rouge2,
        'ROUGE-L': rougeL,
        'Cosine': cosine_similarities,
        "entity_match": avg_entity_match_scores,
        "jacc": avg_numeric_jacc,
        "word_count_pred": word_count_pred,
        "word_count_ref": word_count_ref
    })


# Function to run evaluation in parallel batches
def parallel_evaluate(df, tokenizer, batch_size=512, nlp=None, device='cuda'):
    results = []
    for i in trange(0, len(df), batch_size):
        results_batch = evaluate_batch(df[i: i + batch_size], tokenizer, nlp, device)
        results.append(results_batch)

    # Concatenate all batch results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)
    return results_df


if __name__ == "__main__":

    project_setup()
    """
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
    python -m spacy download en_core_web_lg
    """
    nlp = spacy.load("en_core_web_sm")

    args = parse_args('metrics')

    # filename = os.path.join(args.output_dir, "recover_sentences", f"{args.model_name_or_path.split('/')[-1]}_{dataset_size}.xlsx")

    performances = []

    if args.output_file is None:

        for model_name, data_size in [
            ("Mistral7B", "10M"),
            # ("Llama3.1-8B", "6M"),
            # ("Mistral-Nemo-Base-2407", "500K"),
            # ("MistralSmall", "1M"),
        ]:
            for num_sent in [4]:  # range(1, 5):
                output_file = os.path.join(args.output_dir, "recover_sentences", f"{model_name}_{data_size}",
                                           f"answers_{model_name}_{data_size}_num-sent{num_sent}.xlsx")

                print_file_timestamps(output_file)

                df = pd.read_excel(output_file).dropna().reset_index(drop=True)

                # Load a BERT model and tokenizer for embeddings
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model = AutoModel.from_pretrained('bert-base-uncased').to(args.device)

                # Run parallel evaluation
                results_df = parallel_evaluate(df, tokenizer=tokenizer, batch_size=64, nlp=nlp, device=args.device)
                print("Evaluation Results:")

                aggregated_results = results_df.mean(axis=0)
                print(aggregated_results)
                aggregated_results["model_name"] = model_name
                aggregated_results["data_size"] = data_size

                performances.append(aggregated_results)

        performances = pd.DataFrame(performances)
        performances.to_csv(f"{args.output_dir}/recover_sentences/performance_summary_recover_sentences.csv",
                            index=False)

    else:
        print_file_timestamps(args.output_file)

        df = pd.read_excel(args.output_file).dropna().reset_index(drop=True)

        # Load a BERT model and tokenizer for embeddings
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased').to(args.device)

        # Run parallel evaluation
        results_df = parallel_evaluate(df, tokenizer=tokenizer, batch_size=64, nlp=nlp, device=args.device)
        print("Evaluation Results:")

        aggregated_results = results_df.mean(axis=0)
        print(aggregated_results)
        performances.append(aggregated_results)
        performances = pd.DataFrame(performances)
        performances.to_csv(f"{args.output_dir}/recover_paragraph/performance_summary_recover_paragraph.csv",
                            index=False)

    # ##############################
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.regplot(x="num_tokens", y="confidence", data=df, ci=None, line_kws={"color": "red"})
    plt.title("Relation between Confidence and Number of Tokens")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.show()

    # Checking linear relation
    slope, intercept, r_value, p_value, std_err = linregress(df["num_tokens"], df["confidence"])
    print(f"R-squared: {r_value ** 2:.4f}")
    print(f"P-value: {p_value:.4e}")
