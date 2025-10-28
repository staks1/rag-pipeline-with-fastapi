import os
from typing import List
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)

from dotenv import load_dotenv


async def get_databases(cl: AsyncIOMotorClient) -> List[str]:
    res = await cl.list_database_names()
    return res


load_dotenv()


def get_database() -> AsyncIOMotorDatabase:
    given_db = os.environ["DB_CHOICE"]
    db_client_path = os.environ["DB_PATH"]
    db_client = AsyncIOMotorClient(db_client_path)
    return db_client[given_db], db_client


def get_collection(
    given_db: AsyncIOMotorDatabase, collection_name: str
) -> AsyncIOMotorCollection:
    return given_db[collection_name]
