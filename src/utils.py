import random
import time
import numpy as np
from openai import RateLimitError
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()


def call_llm_with_retry(func, max_retries: int = 5, initial_delay: int = 4):
    """
    wrapper for llm to handle rate limits
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            print(f"Rate limit hit, retrying in {delay}s...")
            time.sleep(delay + random.uniform(0, 0.5))  # small jitter
            delay *= 2  # exponential backoff
    raise Exception("Max retries exceeded")


def query_vector_store(query_embedding: np.float16, vstore_client: QdrantClient):
    query_result = vstore_client.query_points(
        collection_name=os.environ["COLLECTION"],
        query=query_embedding,
        with_vectors=True,
        with_payload=True,
        limit=int(os.environ["TOP_K"]),  # we can limit the results
        score_threshold=float(os.environ["SCORE_THRESHOLD"]),
    )
    return query_result


def create_context_from_vdb(query_result: Any) -> str:
    context = ""
    context += (
        "\n".join([x.payload["text"] for x in query_result.points])
        if query_result.points != []
        else context
    )
    return context
