from fastapi import FastAPI
from contextlib import asynccontextmanager
from openai import OpenAI
from qdrant_client import QdrantClient
import uvicorn
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic_models import Query, Ragresult
from models import Embeddingmodel
import os
from utils import call_llm_with_retry, query_vector_store, create_context_from_vdb
from prompts import user_prompt, system_prompt

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
# ambiguous, i am guessing one endpoint that uses all default cost values to calculate the final result
app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # probably set up client when application starts

    # we load the model only when app starts to avoid doing heavy calculations beforehand
    embedding_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/qa", response_model=Ragresult)
def rag_questions(query: Query) -> str:
    # this is a sync function since it calls some sync functions
    # we need to get the question string and send it to the query endpoint of vector store
    # create the embedding of the query using our model
    query_embedding = embedding_model.embed_query(query.question)
    # send the query to the vector store
    query_result = query_vector_store(query_embedding, vstore_client)
    # get the tex extracted from vector database (assumption we get the full paragraph)
    new_context = create_context_from_vdb(query_result)
    # this should probably be sent to a new endpoint that handles the llm part
    # TODO : assumption that we give full confidence to our knowledge base
    # so if no point is returned with high similarity we return directly i do not know
    # no llm call this way , this is a design choice we could
    # as well call llm with some kind of instruction "the anwser will not be precise, please give more info"
    if new_context == "":
        return Ragresult(
            answer="I do not know, you should provide more information please"
        )
    completion = call_llm_with_retry(
        lambda: llm_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(query, new_context)},
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
