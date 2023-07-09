import os
from uuid import uuid4

MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION") or "c" + uuid4().hex
MILVUS_HOST = os.environ.get("MILVUS_HOST") or "localhost"
MILVUS_PORT = os.environ.get("MILVUS_PORT") or 19530
MILVUS_USER = os.environ.get("MILVUS_USER")
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD")
MILVUS_USE_SECURITY = False if MILVUS_PASSWORD is None else True

MILVUS_INDEX_PARAMS = os.environ.get("MILVUS_INDEX_PARAMS")
MILVUS_SEARCH_PARAMS = os.environ.get("MILVUS_SEARCH_PARAMS")
MILVUS_CONSISTENCY_LEVEL = os.environ.get("MILVUS_CONSISTENCY_LEVEL")

UPSERT_BATCH_SIZE = 20
OUTPUT_DIM = 1536
EMBEDDING_FIELD = "embedding"

from typing import Dict, List, Optional
from pymilvus import (
    Collection,
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    MilvusException,
)

from ...models.models import QueryResult, QueryWithEmbedding, DocumentMetadataFilter
from ...datastore.datastore import DataStore


class Required:
    pass


# The fields names that we are going to be storing within Milvus, the field declaration for schema creation, and the default value
SCHEMA_V2 = [
    (
        EMBEDDING_FIELD,
        FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=OUTPUT_DIM),
        Required,
    ),
    (
        "text",
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        Required,
    ),
    (
        "document_id",
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=65535),
        "",
    ),
    (
        "source_id",
        FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=65535),
        "",
    ),
    (
        "id",
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=65535,
            is_primary=True,
        ),
        "",
    ),
    (
        "source",
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
        "",
    ),
    ("url", FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=65535), ""),
    ("created_at", FieldSchema(name="created_at", dtype=DataType.INT64), -1),
    (
        "author",
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=65535),
        "",
    ),
]


class MilvusDataStore(DataStore):
    def __init__(
        self,
        create_new: Optional[bool] = False,
        consistency_level: str = os.environ.get("MILVUS_CONSISTENCY_LEVEL", "Bounded"),
        milvus_collection: str = os.environ.get("MILVUS_COLLECTION"),
        milvus_host: str = os.environ.get("MILVUS_HOST") or "localhost",
        milvus_port: int = int(os.environ.get("MILVUS_PORT") or 19530),
        milvus_user: Optional[str] = os.environ.get("MILVUS_USER"),
        milvus_password: Optional[str] = os.environ.get("MILVUS_PASSWORD"),
        milvus_use_security: bool = False
        if os.environ.get("MILVUS_PASSWORD") is None
        else True,
        milvus_index_params: Optional[str] = os.environ.get("MILVUS_INDEX_PARAMS"),
        milvus_search_params: Optional[str] = os.environ.get("MILVUS_SEARCH_PARAMS"),
        upsert_batch_size: int = 20,
        output_dim: int = 1536,
        embedding_field: str = "embedding",
        schema: List = SCHEMA_V2,
    ):
        """Create a Milvus DataStore

        The Milvus Datastore allows for storing your indexes and metadata within a Milvus instance.

        Args:
                                        create_new (Optional[bool], optional): Whether to overwrite if collection already exists. Defaults to True.
                                        consistency_level(str, optional): Specify the collection consistency level.
                                                                                                                                                                                                                                                                                                                                        Defaults to "Bounded" for search performance.
                                                                                                                                                                                                                                                                                                                                        Set to "Strong" in test cases for result validation.
        """
        self.create_new = create_new
        self.consistency_level = consistency_level
        self.milvus_collection = milvus_collection
        # assert if milvus_colleciton is None
        if self.milvus_collection is None:
            raise ValueError("milvus_collection must be specified")
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.milvus_user = milvus_user
        self.milvus_password = milvus_password
        self.milvus_use_security = milvus_use_security
        self.milvus_index_params = milvus_index_params
        self.milvus_search_params = milvus_search_params
        self.upsert_batch_size = upsert_batch_size
        self.output_dim = output_dim
        self.embedding_field = embedding_field

        self.index_params = milvus_index_params
        self.search_params = milvus_search_params
        self.col = None
        self.alias = ""
        self.schema = schema

        self._initialize()

    def _initialize(self):
        """Initialize the Milvus DataStore, override if you need to do something special"""
        self._create_connection()
        self._create_collection(self.milvus_collection, self.create_new)  # type: ignore
        self._create_index()

    def _get_schema(self):
        """Get the schema for the Milvus collection"""
        return self.schema

    def insert(self, chunks, batch_size=UPSERT_BATCH_SIZE):
        """inserts data into the milvus collection"""
        pass

    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """Query the Milvus collection for the given queries"""
        pass

    def _get_filter(self, filter: DocumentMetadataFilter) -> Optional[str]:
        """Converts a DocumentMetdataFilter to the expression that Milvus takes.

        Args:
                filter (DocumentMetadataFilter): The Filter to convert to Milvus expression.

        Returns:
                Optional[str]: The filter if valid, otherwise None.
        """
        filters = []
        # Go through all the fields and their values
        for field, value in filter.dict().items():
            # Check if the Value is empty
            if value is not None:
                # Convert start_date to int and add greater than or equal logic
                if field == "start_date":
                    filters.append(
                        "(created_at >= " + str(to_unix_timestamp(value)) + ")"
                    )
                # Convert end_date to int and add less than or equal logic
                elif field == "end_date":
                    filters.append(
                        "(created_at <= " + str(to_unix_timestamp(value)) + ")"
                    )
                # Convert Source to its string value and check equivalency
                elif field == "source":
                    filters.append("(" + field + ' == "' + str(value.value) + '")')
                # Check equivalency of rest of string fields
                else:
                    filters.append("(" + field + ' == "' + str(value) + '")')
        # Join all our expressions with `and``
        return " and ".join(filters)

    def _create_connection(self):
        """
        Create the initial connection to the milvus datasetore
        """
        try:
            self.alias = ""
            # Check if the connection already exists
            for x in connections.list_connections():
                addr = connections.get_connection_addr(x[0])
                if (
                    x[1]
                    and ("address" in addr)
                    and (
                        addr["address"]
                        == "{}:{}".format(self.milvus_host, self.milvus_port)
                    )
                ):
                    self.alias = x[0]
                    print(
                        "Reuse connection to Milvus server '{}:{}' with alias '{:s}'".format(
                            self.milvus_host, self.milvus_port, self.alias
                        )
                    )
                    break

            # Connect to the Milvus instance using the passed in Environment variables
            if len(self.alias) == 0:
                self.alias = uuid4().hex
                connections.connect(
                    alias=self.alias,
                    host=self.milvus_host,
                    port=self.milvus_port,
                    user=self.milvus_user,  # type: ignore
                    password=self.milvus_password,  # type: ignore
                    secure=self.milvus_use_security,
                )
                print(
                    "Create connection to Milvus server '{}:{}' with alias '{:s}'".format(
                        self.milvus_host, self.milvus_port, self.alias
                    )
                )
        except Exception as e:
            print(
                "Failed to create connection to Milvus server '{}:{}', error: {}".format(
                    self.milvus_host, self.milvus_port, e
                )
            )

    def _connect_to_collection(self, collection_name: str):
        """used to just connect to an existin collection"""
        self.col = Collection(collection_name, using=self.alias)

    def _create_collection(self, collection_name, create_new: bool) -> None:
        """Create a collection based on environment and passed in variables.

        Args:
                                        create_new (bool): Whether to overwrite if collection already exists.
        """
        try:
            # If the collection exists and create_new is True, drop the existing collection
            if utility.has_collection(collection_name, using=self.alias) and create_new:
                utility.drop_collection(collection_name, using=self.alias)

            # Check if the collection doesnt exist
            if utility.has_collection(collection_name, using=self.alias) is False:
                # If it doesnt exist use the field params from init to create a new schem
                schema = [field[1] for field in self.schema]
                schema = CollectionSchema(schema)
                # Use the schema to create a new collection
                self.col = Collection(
                    collection_name,
                    schema=schema,
                    using=self.alias,
                    consistency_level=self._consistency_level,
                )
                print(
                    f"Create Milvus collection '{collection_name}' and consistency level {self._consistency_level}"
                )
            else:
                # If the collection exists, point to it
                self.col = Collection(collection_name, using=self.alias)  # type: ignore
                # Which sechma is used

                print(f"Milvus collection '{collection_name}' already exists")
        except Exception as e:
            print(f"Failed to create collection '{collection_name}', error: {e}")

    def _create_index(self):
        try:
            # If no index on the collection, create one
            if len(self.col.indexes) == 0:
                if self.index_params is not None:
                    # Convert the string format to JSON format parameters passed by MILVUS_INDEX_PARAMS
                    self.index_params = json.loads(self.index_params)
                    print("Create Milvus index: {}".format(self.index_params))
                    # Create an index on the 'embedding' field with the index params found in init
                    self.col.create_index(
                        EMBEDDING_FIELD, index_params=self.index_params
                    )
                else:
                    # If no index param supplied, to first create an HNSW index for Milvus
                    try:
                        i_p = {
                            "metric_type": "IP",
                            "index_type": "HNSW",
                            "params": {"M": 8, "efConstruction": 64},
                        }
                        print(
                            "Attempting creation of Milvus '{}' index".format(
                                i_p["index_type"]
                            )
                        )
                        self.col.create_index(EMBEDDING_FIELD, index_params=i_p)
                        self.index_params = i_p
                        print(
                            "Creation of Milvus '{}' index successful".format(
                                i_p["index_type"]
                            )
                        )
                    # If create fails, most likely due to being Zilliz Cloud instance, try to create an AutoIndex
                    except MilvusException:
                        print("Attempting creation of Milvus default index")
                        i_p = {
                            "metric_type": "IP",
                            "index_type": "AUTOINDEX",
                            "params": {},
                        }
                        self.col.create_index(EMBEDDING_FIELD, index_params=i_p)
                        self.index_params = i_p
                        print("Creation of Milvus default index successful")
            # If an index already exists, grab its params
            else:
                # How about if the first index is not vector index?
                for index in self.col.indexes:
                    idx = index.to_dict()
                    if idx["field"] == EMBEDDING_FIELD:
                        print("Index already exists: {}".format(idx))
                        self.index_params = idx["index_param"]
                        break

            self.col.load()

            if self.search_params is not None:
                # Convert the string format to JSON format parameters passed by MILVUS_SEARCH_PARAMS
                self.search_params = json.loads(self.search_params)
            else:
                # The default search params
                metric_type = "IP"
                if "metric_type" in self.index_params:
                    metric_type = self.index_params["metric_type"]
                default_search_params = {
                    "IVF_FLAT": {"metric_type": metric_type, "params": {"nprobe": 10}},
                    "IVF_SQ8": {"metric_type": metric_type, "params": {"nprobe": 10}},
                    "IVF_PQ": {"metric_type": metric_type, "params": {"nprobe": 10}},
                    "HNSW": {"metric_type": metric_type, "params": {"ef": 10}},
                    "RHNSW_FLAT": {"metric_type": metric_type, "params": {"ef": 10}},
                    "RHNSW_SQ": {"metric_type": metric_type, "params": {"ef": 10}},
                    "RHNSW_PQ": {"metric_type": metric_type, "params": {"ef": 10}},
                    "IVF_HNSW": {
                        "metric_type": metric_type,
                        "params": {"nprobe": 10, "ef": 10},
                    },
                    "ANNOY": {"metric_type": metric_type, "params": {"search_k": 10}},
                    "AUTOINDEX": {"metric_type": metric_type, "params": {}},
                }
                # Set the search params
                self.search_params = default_search_params[
                    self.index_params["index_type"]
                ]
            print(f"Milvus search parameters: {self.search_params}")
        except Exception as e:
            print("Failed to create index, error: {e}")
