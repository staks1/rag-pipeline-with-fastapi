from fastapi import Depends, FastAPI
from contextlib import asynccontextmanager
from openai import OpenAI
from qdrant_client import QdrantClient
import uvicorn
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic_models import (
    Query,
    Ragresult,
    Averagemetrics,
    Perquerymetricscreate,
    Basecost,
    PerquerymetricsPost,
)
from models import Embeddingmodel
import os
from utils import (
    call_llm_with_retry,
    query_for_groups,
    find_best_source_and_decide,
    query_all_chunks_from_doc_winner,
)
from prompts import user_prompt, system_prompt
from database import get_database, get_collection, get_databases

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)


# load .env
load_dotenv()


# initialize the class
embedding_model = Embeddingmodel()

# create our clients
llm_client = OpenAI()
vstore_client = QdrantClient(url=os.environ["VECTOR_STORE"])


# 1 endpoint for post questions (user rag)
# send to vector database --> receive top k --> send to llm --> receive result

# 2 endpoint for metrics
# database or txt file that is updated with all requests metrics

# 3 endpoint for cost calculation
app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # probably set up client when application starts

    # we load the model only when app starts to avoid doing heavy calculations beforehand
    db, client = get_database()
    app.state.db = db
    app.state.client = client
    embedding_model.load_model()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="RAG Pipeline",
    summary="RAg Pipeline for museum customer questions. The user can ask questions and receive answers.",
    version="1.0",
)


@app.post(
    "/querypost",
)
async def calculate_and_store_query_cost(query_cost: Perquerymetricscreate):
    """Function to calculate and insert the query into the database"""
    # TODO : maybe add option to use the "query string/description" as id
    # to store in the database of metrics along with the query
    query_metric = PerquerymetricsPost(**query_cost.model_dump())

    await app.state.db["per_query_metrics"].insert_one(
        query_metric.model_dump(by_alias=True)
    )


# i also defined a query index based get request for the average metrics
# in case we need to grab by id
# similarly we could do for the base_costs
# @app.get("/getmetrics/{id}", response_model=Averagemetrics)
# async def get_the_avg_metrics(id: str):

#     avg_metric = await app.state.db["average_metrics"].find_one({"_id": id})
#     if avg_metric is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
#     return avg_metric


@app.get("/get_avgmetrics", response_model=Averagemetrics)
async def get_the_avg_metrics():
    all_avg_metrics = []
    avg_metric = app.state.db["average_metrics"].find({})
    if avg_metric is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    async for x in avg_metric:
        all_avg_metrics.append(x)
    return all_avg_metrics[0]


@app.get("/get_base_costs", response_model=Basecost)
async def get_the_base_costs():
    all_base_costs = []
    base_cost = app.state.db["base_cost"].find({})
    if base_cost is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    async for x in base_cost:
        all_base_costs.append(x)
    return all_base_costs[0]


@app.post("/qa", response_model=Ragresult)
def rag_questions(query: Query) -> str:
    # this is a sync function since it calls some sync functions
    # we need to get the question string and send it to the query endpoint of vector store
    # create the embedding of the query using our model
    query_embedding = embedding_model.embed_query(query.question)

    # query to get the document group
    group_result = query_for_groups(
        vstore_client,
        os.environ["COLLECTION"],
        query_embedding,
        top_k=os.environ["GTOP_K"],
        metadata_key=os.environ["METADATA_FILTER"],
        score_threshold=os.environ["GSCORE_THRESHOLD"],
    )

    # get the best group (if clear winner) or return dont know
    doc_winner = find_best_source_and_decide(
        group_result, top_score_diff_thres=float(os.environ["GSCORE_DIFF_THRESHOLD"])
    )

    # TODO : assumption that we give full confidence to our knowledge base
    # so if no point is returned with high similarity we return directly i do not know
    # no llm call this way , this is a design choice we could
    # as well call llm with some kind of instruction "the anwser will not be precise, please give more info"
    if doc_winner == "":
        return Ragresult(
            answer="I do not know this, you should provide more information or be more concise regarding your topic!"
        )

    else:
        total_context = query_all_chunks_from_doc_winner(
            vstore_client,
            os.environ["PTOP_K"],
            doc_winner,
            os.environ["COLLECTION"],
            query_embedding,
        )
        print(f"-----new context----- \n: {total_context}")

        # calculate the cost of input + output

        # Now we have the enriched context we can ask again the llm adding the question and the context
        completion = call_llm_with_retry(
            lambda: llm_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(query, total_context)},
                ],
                temperature=0.1,
                top_p=0.95,
            )
        )
        return Ragresult(answer=completion.choices[0].message.model_dump()["content"])


@app.get("/cost", response_class=JSONResponse)
async def get_total_cost():
    # TODO : well actually this just needs to return the database (maybe a mongo instance)
    # that has the total cost of requests until now
    return {"endpoint": "cost"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=20000)
