from pydantic import BaseModel, Field
from bson.objectid import ObjectId


class Query(BaseModel):
    question: str


class Ragresult(BaseModel):
    answer: str


class Mongocompatmodel(BaseModel):
    # the default_factory is called in python so no serialization is needed for the id
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Pydantic Models for db
class Averagemetrics(BaseModel):
    avg_latency: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_cache_hit_rate: float
    avg_top_k_score: float
    # add the totals
    total_completion_tokens: float
    total_latency: float
    total_prompt_tokens: float
    total_topk_score: float
    total_cache_hit_rate: float
    total: int


class Basecost(BaseModel):
    price_per_1_input_token: int
    price_per_1_output_token: int


class PerquerymetricsPost(Mongocompatmodel):
    total_llm_input_cost: float
    total_llm_output_cost: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    # query_title: str


class Perquerymetricscreate(BaseModel):
    total_llm_input_cost: float
    total_llm_output_cost: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    # query_title: str = None


# this is a model used to calculate the metrics for a query
# that need to be used for the running average calculation
class Perqueryaveragemetrics(BaseModel):
    completion_tokens: float
    latency: float
    prompt_tokens: float
    topk_score: float
    cache_hit_rate: float
