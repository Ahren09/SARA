import json
import re
import string
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm.contrib import tzip
import traceback

from ..utils.text_utils import SimpleTokenizer


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s):
    """
    Normalize the answer by lowering the case, removing punctuation,
    articles, extra whitespace, and stop words.
    """

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_stopwords(text):
        words = text.split()
        return " ".join(word for word in words if word not in stop_words)

    def lower(text):
        return text.lower()

    if "---Response---\n\n" in s:
        s = s.split("---Response---\n\n")[1]
    if "---Answer---\n\n" in s:
        s = s.split("---Answer---\n\n")[1]

    if "Answer: " in s:
        s = s.split("Answer:")[1]

    s = white_space_fix(remove_stopwords(remove_articles(remove_punc(lower(s)))))

    return s


# Borrowed from [LLMLingua](https://github.com/microsoft/LLMLingua)
def f1_score(prediction, ground_truth, **kwargs):

    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Borrowed from [LLMLingua](https://github.com/microsoft/LLMLingua)
def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)



def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        p = r = f1 = int(gold_toks == pred_toks)
        return p, r, f1
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def eval_recall(infile):
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count / len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_fact_checking(outputs, answers):
    tokenizer = SimpleTokenizer()

    results = []
    acc_count = 0
    answer_lengths = []
    for output, answer in zip(outputs, answers):

        if answer == "False":
            answer = ["refutes", "no", "false"]
        if answer == "True":
            answer = ["supports", "yes", "true"]
        assert answer == ["refutes", "no", "false"] or answer == ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            acc_count += 1
            results.append(1.0)
        else:
            results.append(0.0)

        answer_lengths.append(len(output.split()))

    acc = round(sum(results) / len(results), 4)
    return acc, results



def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def get_exact_match_score(outputs, answers, answerables):
    import numpy as np
    assert len(outputs) == len(answers) == len(answerables)
    if not isinstance(answers[0], list):
        answers = [[x] for x in answers]
    exact_match_scores = []
    answer_lengths = []
    
    for output, answer, answerable in tzip(outputs, answers, answerables):
        if not answerable:
            continue
        if isinstance(answer, str):
            if ems(output, answer):  # EM evaluation
                exact_match_scores.append(1.0)
            else:
                exact_match_scores.append(0.0)
                
        elif isinstance(answer, list):
            em_scores = [ems(output, ans) for ans in answer]
            if any(em_scores) > 0:
                exact_match_scores.append(1.0)
            else:
                exact_match_scores.append(0.0)

        answer_lengths.append(len(output.split()))

  
    print(f"EM score: {np.mean(exact_match_scores) * 100:.2f} +/- {np.std(exact_match_scores) * 100:.2f}")
    em = round(sum(exact_match_scores) / len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, exact_match_scores


def f1_score(prediction, ground_truth):
    if isinstance(prediction, str):
        prediction_tokens = normalize_answer(prediction).split()
    else:
        prediction_tokens = prediction
    if isinstance(ground_truth, str):

        ground_truth_tokens = normalize_answer(ground_truth).split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    # no normalization
    try:
        if isinstance(ground_truth, str):
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
            return scores["rouge-l"]["f"]
        
        elif isinstance(ground_truth, list):
            
            scores = []
            
            for gt in ground_truth:
                try:
                    score = rouge.get_scores(prediction, gt, avg=True)
                    scores.append(score["rouge-l"]["f"])
                except:
                    continue
            
            return max(scores) if len(scores) > 0 else None
        else:
            raise ValueError(f"Unexpected type for ground_truth: {type(ground_truth)}")
        
    except:  # "Hypothesis is empty."
        traceback.print_exc()
        print(f"Rouge requires non-empty strings, but get:\n\tPred:\t\"{prediction}\"\n\tGT:\t"
                         f"\"{ground_truth}\"")
        return None




def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


def get_unigram_f1(text: str, answers: list[str]) -> float:
    """Calculate unigram f1 score between the text and reference answers."""

    def _get_unigram_f1(text, answers):
        if isinstance(answers, str):
            answers = [answers]
        norm_pred = normalize_answer(text)
        norm_answers = [normalize_answer(ans) for ans in answers]
        common_tokens = [
            Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
        ]
        num_same = [sum(common.values()) for common in common_tokens]

        score_list = []
        for i, num in enumerate(num_same):
            if num == 0:
                score_list.append(0.0)
            else:
                p = 1.0 * num / len(norm_pred)
                r = 1.0 * num / len(norm_answers[i])
                f1 = 2 * p * r / (p + r)
                score_list.append(f1)
        return max(score_list)

    unigram_f1 = [_get_unigram_f1(t, a) for t, a in zip(text, answers)]

    return sum(unigram_f1) / len(unigram_f1), unigram_f1
