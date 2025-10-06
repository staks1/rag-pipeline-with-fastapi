from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()


class Embeddingmodel:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = BGEM3FlagModel(
            os.environ["EMBEDDER_MODEL"], use_fp16=True, devices="cpu"
        )

    def embed_query(self, query: str) -> np.float16:
        if not self.model:
            raise RuntimeError("Model is not loaded")
        # here we need to add caching to avoid redundant calculations
        query_embedding = self.model.encode(query)["dense_vecs"]
        return query_embedding
