import os
from time import time
from typing import List
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever  # pip install llama-index-retrievers-bm25

from contextlib import redirect_stdout


def get_env():
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict


def getOpenAIRetriever(documents: list[str], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding

    openai.api_key = get_env()["OPENAI_API_KEY"]
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")

    # Set the embed_model in llama_index
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=get_env()["OPENAI_API_KEY"],
                                           title="openai-embedding")
    # model_name: "text-embedding-3-small", "text-embedding-3-large"

    # Create the OpenAI retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()

    return OpenAI_retriever, t2 - t1


def getGeminiRetriever(documents: list[str], similarity_top_k: int = 1):
    """Gemini Embedding RAG model"""
    GOOGLE_API_KEY = get_env()["GOOGLE_API_KEY"]
    from llama_index.embeddings.gemini import GeminiEmbedding # pip install llama-index-embeddings-gemini

    from llama_index.core import VectorStoreIndex
    model_name = "models/embedding-001"
    # Set the embed_model in llama_index
    Settings.embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")

    # Create the Gemini retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()

    return Gemini_retriever, t2 - t1


def getBM25Retriever(documents: List[Document], similarity_top_k: int = 1, chunk_size: int = 256):
    import Stemmer

    splitter = SentenceSplitter(chunk_size=chunk_size)

    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        bm25_retriever.persist("outputs/cache/bm25_retriever")

    return bm25_retriever, t2 - t1


def getHuggingFaceRetriever(documents: List[Document], similarity_top_k: int = 1):
    
    # pip install llama-index-embeddings-huggingface
    # pip install llama-index-embeddings-instructor
    from llama_index.core import VectorStoreIndex

    # Create the HuggingFace retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    HuggingFace_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()

    return HuggingFace_retriever, t2 - t1


