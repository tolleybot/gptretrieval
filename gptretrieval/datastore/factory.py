from datastore.datastore import DataStore
import os


async def get_datastore() -> DataStore:
    datastore = os.environ.get("DATASTORE")
    assert datastore is not None

    match datastore:
        case "milvusbook":
            from datastore.providers.milvus_book_datastore import MilvusBookDataStore

            return MilvusBookDataStore()
        case "milvussource":
            from datastore.providers.milvus_source_datastore import (
                MilvusSourceDataStore,
            )

            return MilvusSourceDataStore()
        case _:
            raise ValueError(f"Unsupported vector database: {datastore}")
