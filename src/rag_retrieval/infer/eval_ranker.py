
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


from src.rag_retrieval import Reranker

#如果自动下载对应的模型失败，请先从huggface下载对应的模型到本地，然后这里输入本地的路径。


model_name = "BAAI/bge-reranker-base"
model_name = "BAAI/bge-reranker-v2-m3"
ranker = Reranker(model_name,dtype='fp16',verbose=0)


dataset = load_dataset("microsoft/ms_marco", "v2.1", split="validation")


# Evaluation function
def evaluate_ranking(dataset, ranker):
    total_queries = 0
    mrr = 0.0

    for index, entry in enumerate(tqdm(dataset, desc="Evaluating")):
        query = entry['query']
        passages = entry['passages']['passage_text']
        is_selected = entry['passages']['is_selected']

        # Generate query-passage pairs
        pairs = [[query, passage] for passage in passages]

        # Compute scores using the ranker
        scores = ranker.compute_score(pairs)

        # Sort passages by scores in descending order
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Find the rank of the first relevant passage
        for rank, idx in enumerate(ranked_indices, start=1):
            if is_selected[idx] == 1:  # Check if the passage is relevant
                mrr += 1.0 / rank
                break

        total_queries += 1

    # Calculate mean reciprocal rank
    mrr /= total_queries
    return mrr

mrr_score = evaluate_ranking(dataset, ranker)
print(f"Mean Reciprocal Rank (MRR): {mrr_score}")

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]


scores = ranker.compute_score(pairs)

print(scores)
