from collections import defaultdict
from functools import reduce
import random
import time
import numpy as np
from openai import RateLimitError
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from typing import Any
from qdrant_client.models import Filter, FieldCondition, MatchValue
import tiktoken
from typing import List
from itertools import combinations


load_dotenv()

# encoder
enc = tiktoken.encoding_for_model("gpt-4")


def reducer(acc, item):
    key, val = item
    acc[key] = acc.get(key, 0) + val
    return acc


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


def query_vector_store_points(query_embedding: np.float16, vstore_client: QdrantClient):
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
    context = (
        context + "\n".join([x.payload["text"] for x in query_result.points])
        if query_result.points != []
        else context
    )

    return context


# this adds more chunks until the input context is full and we can return it to the user
def add_more_chunks(query_result, context, max_window):
    for p in query_result.points:
        new_total_context = context + " " + p.payload["text"]
        if len(enc.encode(new_total_context)) > max_window:
            break
        else:
            context = new_total_context
    return context.strip()


def query_for_groups(
    qclient,
    collection_name,
    query_embedding,
    top_k,
    score_threshold,
    metadata_key="source",
):
    query_result = qclient.query_points_groups(
        collection_name=collection_name,
        query=query_embedding,
        with_vectors=True,
        with_payload=True,
        limit=int(top_k),  # we can limit the results
        score_threshold=float(
            score_threshold
        ),  # we can also set a minumum score for returning results
        group_by=metadata_key,  # the id to group the results by (i use the document source)
    )
    return query_result


def query_all_chunks_from_doc_winner(
    qclient, top_k, doc_winner, collection_name, query_embedding
):
    query_result = qclient.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_vectors=True,
        with_payload=True,
        limit=int(top_k),  # we can limit the results
        # score_threshold = , # we can also set a minumum score for returning results
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(
                        value=doc_winner,
                    ),
                )
            ]
        ),
    )
    return query_result


def sliding_window_on_text(
    paragraph_text: str,
    window_size: int = 100,
    slide: int = 90,
) -> List[List[str]]:
    """
    Function to create overlaps in the text

    The function will process window_size items
    then move by slide value and process again window_size items.
    This way we create overlaps and we can maintain  some of the contextual information
    between sentences

    Args:
        paragraph_text: the text given
        window_size: how many elements each chunk will include
        slide: defines how much we move forward (how much information we maintain)
    Returns:
        total_tokens: The overlapping chunks of words  produced(nested list)
    """

    # setting close window and slide we make sure we dont start from the begining of a large sequence
    paragraph_text_list = re.split(r"\s+", paragraph_text.strip())
    total_tokens = []

    i = 0
    while i < len(paragraph_text_list):
        # Get the window from i to i + window_size
        window = paragraph_text_list[i : i + window_size]
        total_tokens.append(window)

        # Slide the window and repeat
        i += slide

    return total_tokens


def split_overlap_window_recurse(
    chunk_text: str, max_tokens: int, window: int, slide: int
) -> List[List[str]]:
    """
    Function that recurses to produce chunks with token count less than a max length given

    This is implemented to give the option to run all the llm extracting steps
    (extract properties, extract parties) in smaller overlapping chunks of the
    page text if the whole page text does not fit in the max_context set to max_tokens.
    Args:
        chunk_text: the text given to split
        max_tokens: user defined max context to send to llm
        window: how many items each chunk can include
        slide: defines how much information we maintain while sliding
    Returns:
        overlapping_chunks: the final list of lists of overlapping sentences
    """
    if window <= 0 or slide <= 0:
        return [[chunk_text]]

    overlapping_chunks = sliding_window_on_text(chunk_text, window, slide)
    for x in overlapping_chunks:
        token_count = len(enc.encode(x[0]))
        if token_count > max_tokens:
            print(f"token count {token_count} is large, should split into smaller")
            return split_overlap_window_recurse(
                chunk_text, max_tokens, window - 10, slide - 10
            )

    return overlapping_chunks


def find_best_source_and_decide(query_result, top_score_diff_thres=0.07):
    # case of not enough top score (score<threshold), dont know
    if len(query_result.groups) == 0:
        return ""

    # case of only one group returne, we have clear source document winner
    # we can aggregate more doc chunks from it
    if len(query_result.groups) == 1:
        return query_result.groups[0].id

    # case of many groups returned (only compare the top-k results)
    differences = map(
        lambda x: abs(x[0] - x[1]),
        combinations((x.hits[0].score for x in query_result.groups), 2),
    )
    # no_winner = all(x<=0.07 for x in differences)

    # if difference of top group with second top group above threshold, then top group winner
    if list(differences)[0] >= top_score_diff_thres:
        return query_result.groups[0].id
    else:
        # no winner
        return ""
