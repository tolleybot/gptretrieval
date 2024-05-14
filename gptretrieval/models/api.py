from ..models.models import Document, DocumentMetadataFilter, Query, QueryResult
from pydantic import BaseModel
from typing import List, Optional, Dict


class UpsertRequest(BaseModel):
    documents: List[Document]


class UpsertResponse(BaseModel):
    ids: List[str]


class QueryRequest(BaseModel):
    queries: List[Query]


class Partitions(BaseModel):
    partitions: Dict[str, Dict[str, str]] = None


class QueryResponse(BaseModel):
    state: str
    result: Optional[List[QueryResult]] = None
    error: Optional[str] = None


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filter: Optional[DocumentMetadataFilter] = None
    delete_all: Optional[bool] = False


class DeleteResponse(BaseModel):
    success: bool
