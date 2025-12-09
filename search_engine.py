"""
Semantic Search Engine
End-to-end search pipeline combining embeddings, vector storage, and retrieval.
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from embeddings import EmbeddingService
from vector_store import (
    VectorStoreFactory,
    VectorStoreBase,
    Document,
    SearchResult,
    generate_document_id
)


@dataclass
class SearchQuery:
    """Represents a search query with metadata"""
    text: str
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0


@dataclass
class SearchResponse:
    """Complete search response with results and metadata"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    embedding_time_ms: float


class SemanticSearchEngine:
    """
    Full-stack semantic search engine with document management,
    embedding generation, and vector similarity search.
    """

    def __init__(
        self,
        vector_provider: str = "faiss",
        embedding_provider: str = "mock",
        dimension: int = 1536,
        **kwargs
    ):
        """
        Initialize the semantic search engine.

        Args:
            vector_provider: "faiss" or "pinecone"
            embedding_provider: "openai", "sentence-transformers", or "mock"
            dimension: Embedding dimension
            **kwargs: Additional provider arguments
        """
        self.dimension = dimension

        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            provider=embedding_provider,
            dimension=dimension,
            **kwargs.get("embedding_kwargs", {})
        )

        # Initialize vector store
        self.vector_store = VectorStoreFactory.create(
            provider=vector_provider,
            dimension=dimension,
            **kwargs.get("vector_kwargs", {})
        )

        # Track indexed documents
        self.document_count = 0
        self.index_stats = {
            "total_indexed": 0,
            "total_searches": 0,
            "avg_search_time_ms": 0
        }

    def index_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Document:
        """
        Index a single document.

        Args:
            content: Document text content
            metadata: Optional metadata dictionary
            document_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Indexed Document object
        """
        # Generate ID if not provided
        doc_id = document_id or generate_document_id(content)

        # Generate embedding
        embedding = self.embedding_service.embed(content)

        # Create document
        doc = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Add to vector store
        self.vector_store.add_documents([doc])
        self.document_count += 1
        self.index_stats["total_indexed"] += 1

        return doc

    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Index multiple documents in batch.

        Args:
            documents: List of dicts with 'content' and optional 'metadata', 'id'

        Returns:
            Number of documents indexed
        """
        # Generate embeddings in batch
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_service.embed_batch(contents)

        # Create Document objects
        docs = []
        for doc_dict, embedding in zip(documents, embeddings):
            doc = Document(
                id=doc_dict.get("id", generate_document_id(doc_dict["content"])),
                content=doc_dict["content"],
                embedding=embedding,
                metadata=doc_dict.get("metadata", {})
            )
            docs.append(doc)

        # Add to vector store
        added = self.vector_store.add_documents(docs)
        self.document_count += added
        self.index_stats["total_indexed"] += added

        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> SearchResponse:
        """
        Perform semantic search.

        Args:
            query: Search query text
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns:
            SearchResponse with results and metadata
        """
        import time

        # Generate query embedding
        embed_start = time.perf_counter()
        query_embedding = self.embedding_service.embed(query)
        embed_time = (time.perf_counter() - embed_start) * 1000

        # Search vector store
        search_start = time.perf_counter()
        results = self.vector_store.search(query_embedding, top_k=top_k)
        search_time = (time.perf_counter() - search_start) * 1000

        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]

        # Update stats
        self.index_stats["total_searches"] += 1
        total_searches = self.index_stats["total_searches"]
        prev_avg = self.index_stats["avg_search_time_ms"]
        self.index_stats["avg_search_time_ms"] = (
            (prev_avg * (total_searches - 1) + search_time) / total_searches
        )

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=round(search_time, 2),
            embedding_time_ms=round(embed_time, 2)
        )

    def delete_documents(self, document_ids: List[str]) -> int:
        """Delete documents by ID"""
        deleted = self.vector_store.delete(document_ids)
        self.document_count -= deleted
        return deleted

    def get_stats(self) -> Dict:
        """Get comprehensive engine statistics"""
        return {
            "engine": {
                "document_count": self.document_count,
                "dimension": self.dimension,
                **self.index_stats
            },
            "vector_store": self.vector_store.get_stats(),
            "embedding_service": self.embedding_service.get_stats()
        }

    def export_index(self, filepath: str):
        """Export search index statistics to JSON"""
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


class KnowledgeBase:
    """
    Higher-level knowledge base built on semantic search.
    Adds categorization, tagging, and structured queries.
    """

    def __init__(self, **engine_kwargs):
        """Initialize knowledge base with search engine"""
        self.engine = SemanticSearchEngine(**engine_kwargs)
        self.categories: Dict[str, int] = {}
        self.tags: Dict[str, int] = {}

    def add_knowledge(
        self,
        content: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> Document:
        """
        Add knowledge entry.

        Args:
            content: Knowledge content
            category: Optional category
            tags: Optional list of tags
            source: Optional source reference

        Returns:
            Indexed Document
        """
        metadata = {}

        if category:
            metadata["category"] = category
            self.categories[category] = self.categories.get(category, 0) + 1

        if tags:
            metadata["tags"] = tags
            for tag in tags:
                self.tags[tag] = self.tags.get(tag, 0) + 1

        if source:
            metadata["source"] = source

        return self.engine.index_document(content, metadata)

    def query(
        self,
        question: str,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> SearchResponse:
        """
        Query the knowledge base.

        Args:
            question: Natural language question
            category: Optional category filter
            top_k: Maximum results

        Returns:
            SearchResponse with relevant knowledge
        """
        response = self.engine.search(question, top_k=top_k)

        # Filter by category if specified
        if category:
            response.results = [
                r for r in response.results
                if r.document.metadata.get("category") == category
            ]
            response.total_results = len(response.results)

        return response

    def get_categories(self) -> Dict[str, int]:
        """Get all categories with document counts"""
        return self.categories.copy()

    def get_tags(self) -> Dict[str, int]:
        """Get all tags with document counts"""
        return self.tags.copy()

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            **self.engine.get_stats(),
            "categories": self.categories,
            "tags": self.tags
        }


if __name__ == "__main__":
    # Demo usage
    print("=== Semantic Search Engine Demo ===\n")

    # Create engine with mock embeddings and FAISS
    engine = SemanticSearchEngine(
        vector_provider="faiss",
        embedding_provider="mock",
        dimension=384
    )

    # Sample documents
    documents = [
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"category": "AI", "source": "intro"}
        },
        {
            "content": "Neural networks are computing systems inspired by biological neural networks in the brain.",
            "metadata": {"category": "AI", "source": "deep-learning"}
        },
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"category": "Programming", "source": "basics"}
        },
        {
            "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "metadata": {"category": "Data Science", "source": "overview"}
        },
        {
            "content": "Vector databases store embeddings for efficient similarity search at scale.",
            "metadata": {"category": "Infrastructure", "source": "databases"}
        }
    ]

    # Index documents
    print("Indexing documents...")
    indexed = engine.index_documents(documents)
    print(f"Indexed {indexed} documents\n")

    # Perform searches
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Best programming language for beginners"
    ]

    for query in queries:
        print(f"Query: '{query}'")
        response = engine.search(query, top_k=2)

        print(f"  Found {response.total_results} results in {response.search_time_ms}ms")
        for result in response.results:
            print(f"  #{result.rank} (score: {result.score:.4f}): {result.document.content[:60]}...")
        print()

    # Show stats
    print("Engine Stats:")
    print(json.dumps(engine.get_stats(), indent=2))
