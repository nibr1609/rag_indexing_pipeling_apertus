#!/usr/bin/env python3
"""
Remote Embedding Service Client
Provides a LlamaIndex-compatible embedding interface for a remote embedding service.
"""

import httpx
from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from pydantic import BaseModel


class Chunk(BaseModel):
    """Chunk model for the embedding service."""
    chunk_id: str
    text: str


class RemoteEmbedding(BaseEmbedding):
    """
    Remote embedding model that sends text to an external embedding service.

    Args:
        service_url: URL of the embedding service endpoint
        timeout: Request timeout in seconds (default: 300)
    """

    service_url: str = Field(description="URL of the embedding service")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")

    def __init__(
        self,
        service_url: str,
        timeout: float = 300.0,
        **kwargs: Any
    ):
        super().__init__(service_url=service_url, timeout=timeout, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "RemoteEmbedding"

    def _call_embedding_service(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Call the remote embedding service with a list of chunks.

        Args:
            chunks: List of Chunk objects to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        # Convert chunks to dict format for JSON serialization
        chunks_data = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]

        # Make synchronous HTTP request
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.service_url,
                json=chunks_data
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Embedding service error {response.status_code}: {response.text}"
                )

            response_data = response.json()

            if len(response_data) != len(chunks):
                raise RuntimeError(
                    f"Mismatch: received {len(response_data)} embeddings for {len(chunks)} chunks"
                )

            # Extract just the embedding vectors from the response
            # The service returns [{"chunk_id": "...", "chunk_embedding": [...]}, ...]
            # We need to extract just the "chunk_embedding" field which contains the embedding vectors
            embeddings = [item["chunk_embedding"] for item in response_data]

            return embeddings

    async def _acall_embedding_service(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Async version of _call_embedding_service.

        Args:
            chunks: List of Chunk objects to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        # Convert chunks to dict format for JSON serialization
        chunks_data = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]

        # Make async HTTP request
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.service_url,
                json=chunks_data
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Embedding service error {response.status_code}: {response.text}"
                )

            response_data = response.json()

            if len(response_data) != len(chunks):
                raise RuntimeError(
                    f"Mismatch: received {len(response_data)} embeddings for {len(chunks)} chunks"
                )

            # Extract just the embedding vectors from the response
            # The service returns [{"chunk_id": "...", "chunk_embedding": [...]}, ...]
            # We need to extract just the "chunk_embedding" field which contains the embedding vectors
            embeddings = [item["chunk_embedding"] for item in response_data]

            return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a single query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        chunk = Chunk(chunk_id="query", text=query)
        embeddings = self._call_embedding_service([chunk])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        chunk = Chunk(chunk_id="text", text=text)
        embeddings = self._call_embedding_service([chunk])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Create chunks with sequential IDs
        chunks = [
            Chunk(chunk_id=f"text_{i}", text=text)
            for i, text in enumerate(texts)
        ]
        return self._call_embedding_service(chunks)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Async version of _get_query_embedding.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        chunk = Chunk(chunk_id="query", text=query)
        embeddings = await self._acall_embedding_service([chunk])
        return embeddings[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Async version of _get_text_embedding.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        chunk = Chunk(chunk_id="text", text=text)
        embeddings = await self._acall_embedding_service([chunk])
        return embeddings[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of _get_text_embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Create chunks with sequential IDs
        chunks = [
            Chunk(chunk_id=f"text_{i}", text=text)
            for i, text in enumerate(texts)
        ]
        return await self._acall_embedding_service(chunks)
