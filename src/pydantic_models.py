from pydantic import BaseModel


class Query(BaseModel):
    question: str


class Ragresult(BaseModel):
    answer: str
