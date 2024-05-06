from typing import List
import asyncio

from milvus_base_datastore import (
    MilvusDataStore,
    SCHEMA_V2,
)

try:
    from ...services.openai import get_embeddings
    from ...services.classification import (
        classify_code,
        classify_question,
        select_partition,
    )
except ImportError:
    print("Failed to import openai and classification services")


UPSERT_BATCH_SIZE = 20
EMBEDDING_FIELD = "embedding"
EF_VALUE = 1000

from ...models.models import (
    QueryResult,
    QueryWithEmbedding,
    DocumentMetadataFilter,
    DocumentChunkWithScore,
    DocumentChunkMetadata,
    Source,
)


# Used to create embeddings from source code
class MilvusSrcDataStore(MilvusDataStore):
    def _initialize(self):
        """A Milvous datastore which specializes in source code"""
        self._create_connection()
        self._create_collection(self.milvus_collection, self.create_new)  # type: ignore
        self._create_index()
        self.use_classification = True

    def get_count(self):
        return self.col.num_entities

    def flush(self):
        self.col.flush()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings(texts)

    def insert(self, chunks, batch_size=UPSERT_BATCH_SIZE, partition: str = None):
        """inserts data into the milvus collection"""
        # If chunks is a single dictionary, convert it to a list of dictionaries
        if isinstance(chunks, dict):
            chunks = [chunks]

        # Convert the keys of the dictionary to a list of field names
        fields = [field[1].name for field in SCHEMA_V2]

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Initialize a dictionary to hold the field values
            data = {field: [] for field in fields}

            # Convert each dictionary in batch to the correct format
            for chunk in batch:
                # Add the values of the dictionary to the corresponding lists in the data dictionary
                for field in fields:
                    data[field].append(chunk.get(field, "None"))

            # Convert the data dictionary to a list of lists
            data = list(data.values())

            if partition:
                partitions = self.col.partitions
                partition_names = [p.name for p in partitions]
                if partition not in partition_names:
                    # Create the partition
                    self.col.create_partition(partition_name=partition)

            # Insert the data into the collection
            self.col.insert(data, partition_name=partition)

    async def _query(
        self,
        queries: List[QueryWithEmbedding],
        top_k: int = None,
        partitions: List[str] = None,
    ) -> List[QueryResult]:
        """Query the QueryWithEmbedding against the MilvusDocumentSearch

        Search the embedding and its filter in the collection.

        Args:
                        queries (List[QueryWithEmbedding]): The list of searches to perform.

        Returns:
                        List[QueryResult]: Results for each search.
        """

        # Async to perform the query, adapted from pinecone implementation
        async def _single_query(
            query: QueryWithEmbedding,
            max_attempts: int = 1,
            partitions: List[str] = None,
        ) -> QueryResult:
            try:
                if self.use_classification:
                    question_label = classify_question(query.query)
            except Exception as e:
                print(f"Failed to classify question, error: {e}")
                return QueryResult(query=query.query, results=[])

            for _ in range(max_attempts):
                try:
                    filter = None
                    # Set the filter to expression that is valid for Milvus
                    if query.filter is not None:
                        # Either a valid filter or None will be returned
                        filter = self._get_filter(query.filter)

                    # Perform our search
                    top_k_ = query.top_k if top_k is None else top_k

                    # check partitions, None will search everything so filter out all
                    if partitions is not None:
                        if "all" in partitions:
                            partitions = None
                        else:
                            partitions = select_partition(query.query, partitions)

                    # The 'ef' parameter in Milvus search queries stands for "size of the dynamic candidate list"
                    # and is crucial for controlling the trade-off between search accuracy and performance.
                    self.search_params["params"]["ef"] = EF_VALUE

                    res = self.col.search(
                        data=[query.embedding],
                        anns_field=EMBEDDING_FIELD,
                        param=self.search_params,
                        limit=top_k_,
                        expr=filter,
                        output_fields=[field[0] for field in self._get_schema()[1:]],
                        partitions=partitions,
                    )

                    # Results that will hold our DocumentChunkWithScores
                    results = []
                    # Parse every result for our search
                    for hit in res[0]:  # type: ignore
                        # The distance score for the search result, falls under DocumentChunkWithScore
                        score = hit.score
                        # Our metadata info, falls under DocumentChunkMetadata
                        metadata = {}
                        # Grab the values that correspond to our fields, ignore pk and embedding.
                        for x in [field[0] for field in self._get_schema()[1:]]:
                            metadata[x] = hit.entity.get(x)

                        source = metadata.pop("source")
                        # Text falls under the DocumentChunk
                        text = metadata.pop("text")
                        # Id falls under the DocumentChunk
                        ids = metadata.pop("id")

                        # if the resonse is not relvant, skip it
                        if self.use_classification:
                            code_relevance = classify_code(
                                code=text,
                                question=query.query,
                                question_label=question_label,
                            )
                            if code_relevance["function_args"]["code_label"] == 0:
                                continue

                        chunk = DocumentChunkWithScore(
                            id=ids,
                            score=score,
                            text=source + ": " + text,
                            metadata=DocumentChunkMetadata(**metadata),
                        )
                        results.append(chunk)

                    if results:
                        return QueryResult(query=query.query, results=results)

                except Exception as e:
                    print(f"Failed to query, error: {e}")

                return QueryResult(query=query.query, results=[])

        max_attempts = 1 if not self.use_classification else 3
        results: List[QueryResult] = await asyncio.gather(
            *[
                _single_query(query, max_attempts=max_attempts, partitions=partitions)
                for query in queries
            ]
        )
        return results

    def _query_synch(
        self,
        queries: List[QueryWithEmbedding],
        top_k: int = None,
        partitions: List[str] = None,
    ) -> List[QueryResult]:
        """Query the QueryWithEmbedding against the MilvusDocumentSearch

        Search the embedding and its filter in the collection.

        Args:
            queries (List[QueryWithEmbedding]): The list of searches to perform.

        Returns:
            List[QueryResult]: Results for each search.
        """

        def _single_query(
            query: QueryWithEmbedding,
            max_attempts: int = 1,
            partitions: List[str] = None,
        ) -> QueryResult:
            try:
                if self.use_classification:
                    question_label = classify_question(query.query)
            except Exception as e:
                print(f"Failed to classify question, error: {e}")
                return QueryResult(query=query.query, results=[])

            for _ in range(max_attempts):
                try:
                    filter = None
                    if query.filter is not None:
                        filter = self._get_filter(query.filter)

                    top_k_ = query.top_k if top_k is None else top_k
                    # check partitions, None will search everything so filter out all
                    if partitions is not None:
                        if "all" in partitions:
                            partitions = None
                        else:
                            partitions = select_partition(
                                question=query.query, partitions=partitions
                            )

                    self.search_params["params"]["ef"] = EF_VALUE

                    res = self.col.search(
                        data=[query.embedding],
                        anns_field=EMBEDDING_FIELD,
                        param=self.search_params,
                        limit=top_k_,
                        # expr=filter,
                        output_fields=[field[0] for field in self._get_schema()[1:]],
                        partitions=partitions,
                    )

                    results = []
                    for hit in res[0]:
                        score = hit.score
                        metadata = {}
                        for x in [field[0] for field in self._get_schema()[1:]]:
                            metadata[x] = hit.entity.get(x)

                        source = metadata.pop("source")
                        text = metadata.pop("text")
                        ids = metadata.pop("id")

                        if self.use_classification:
                            code_relevance = classify_code(
                                code=text,
                                question=query.query,
                                question_label=question_label,
                            )
                            if code_relevance["function_args"]["code_label"] == 0:
                                continue

                        chunk = DocumentChunkWithScore(
                            id=ids,
                            score=score,
                            text=source + ": " + text,
                            metadata=DocumentChunkMetadata(**metadata),
                        )
                        results.append(chunk)

                    if results:
                        return QueryResult(query=query.query, results=results)

                except Exception as e:
                    print(f"Failed to query, error: {e}")

            return QueryResult(query=query.query, results=[])

        max_attempts = 1 if not self.use_classification else 3
        results = [
            _single_query(query, max_attempts=max_attempts, partitions=partitions)
            for query in queries
        ]
        return results
