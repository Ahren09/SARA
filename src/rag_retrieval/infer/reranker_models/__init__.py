import os
import sys


AVAILABLE_RANKERS = {}

from src.rag_retrieval.infer.reranker_models.cross_encoder_ranker import CrossEncoderRanker
from src.rag_retrieval.infer.reranker_models.llm_rankers import LLMRanker
AVAILABLE_RANKERS["CrossEncoderRanker"] = CrossEncoderRanker
AVAILABLE_RANKERS["LLMRanker"] = LLMRanker


# try:
#     from rag_retrieval.reranker_models.api_rankers import APIRanker
#     AVAILABLE_RANKERS["APIRanker"] = APIRanker
# except Exception as e:
#     pass

