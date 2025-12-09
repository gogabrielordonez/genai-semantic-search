"""
Vector Store Abstraction Layer
Unified interface for Pinecone (cloud) and FAISS (local) vector databases.
Enables seamless switching between providers for semantic search applications.
"""

import os
import json
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

# Optional imports
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class Document:
    """Represents a document with its embedding and metadata"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass
class SearchResult:
    """Result from a semantic search query"""
    document: Document
    score: float
    rank: int

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "content": self.document.content,
            "metadata": self.document.metadata
        }


class VectorStoreBase(ABC):
    """Abstract base class for vector store implementations"""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]) -> int:
        """Delete documents by ID"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        pass


class FAISSVectorStore(VectorStoreBase):
    """
    Local vector store using Facebook AI Similarity Search (FAISS).
    Optimized for development and small-to-medium scale deployments.
    """

    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension (1536 for OpenAI, 768 for many others)
            index_path: Optional path to load existing index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.index_path = index_path

        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)

        # Store document metadata separately (FAISS only stores vectors)
        self.documents: Dict[int, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.current_index = 0

        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents with embeddings to FAISS index"""
        added = 0

        for doc in documents:
            if doc.embedding is None:
                continue

            # Convert to numpy array
            embedding = np.array([doc.embedding], dtype=np.float32)

            # Add to FAISS index
            self.index.add(embedding)

            # Store document metadata
            self.documents[self.current_index] = doc
            self.id_to_index[doc.id] = self.current_index
            self.current_index += 1
            added += 1

        return added

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents using L2 distance"""
        if self.index.ntotal == 0:
            return []

        # Convert query to numpy array
        query = np.array([query_embedding], dtype=np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break

            doc = self.documents.get(idx)
            if doc:
                # Convert L2 distance to similarity score (0-1)
                # Lower distance = higher similarity
                similarity = 1 / (1 + dist)

                results.append(SearchResult(
                    document=doc,
                    score=similarity,
                    rank=rank + 1
                ))

        return results

    def delete(self, document_ids: List[str]) -> int:
        """
        Delete documents by ID.
        Note: FAISS doesn't support direct deletion, so we rebuild the index.
        """
        deleted = 0
        remaining_docs = []

        for idx, doc in self.documents.items():
            if doc.id in document_ids:
                deleted += 1
            else:
                remaining_docs.append(doc)

        if deleted > 0:
            # Rebuild index without deleted documents
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            self.id_to_index = {}
            self.current_index = 0
            self.add_documents(remaining_docs)

        return deleted

    def get_stats(self) -> Dict:
        """Get FAISS index statistics"""
        return {
            "provider": "FAISS",
            "type": "local",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatL2"
        }

    def save_index(self, path: str):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, f"{path}.faiss")

        metadata = {
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
            "id_to_index": self.id_to_index,
            "current_index": self.current_index,
            "dimension": self.dimension
        }

        with open(f"{path}.meta.json", 'w') as f:
            json.dump(metadata, f)

    def _load_index(self, path: str):
        """Load FAISS index and metadata from disk"""
        if os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")

        if os.path.exists(f"{path}.meta.json"):
            with open(f"{path}.meta.json", 'r') as f:
                metadata = json.load(f)
                self.id_to_index = metadata.get("id_to_index", {})
                self.current_index = metadata.get("current_index", 0)


class PineconeVectorStore(VectorStoreBase):
    """
    Cloud-native vector store using Pinecone.
    Optimized for production scale with managed infrastructure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "semantic-search",
        dimension: int = 1536,
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
            dimension: Embedding dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not installed. Run: pip install pinecone-client")

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required")

        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Create or connect to index
        self._init_index()

    def _init_index(self):
        """Initialize or connect to Pinecone index"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(self.index_name)

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to Pinecone index"""
        vectors = []

        for doc in documents:
            if doc.embedding is None:
                continue

            vectors.append({
                "id": doc.id,
                "values": doc.embedding,
                "metadata": {
                    "content": doc.content[:1000],  # Pinecone metadata limit
                    **doc.metadata
                }
            })

        if vectors:
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

        return len(vectors)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents in Pinecone"""
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        results = []
        for rank, match in enumerate(response.matches):
            doc = Document(
                id=match.id,
                content=match.metadata.get("content", ""),
                metadata={k: v for k, v in match.metadata.items() if k != "content"}
            )

            results.append(SearchResult(
                document=doc,
                score=match.score,
                rank=rank + 1
            ))

        return results

    def delete(self, document_ids: List[str]) -> int:
        """Delete documents from Pinecone"""
        self.index.delete(ids=document_ids)
        return len(document_ids)

    def get_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        stats = self.index.describe_index_stats()

        return {
            "provider": "Pinecone",
            "type": "cloud",
            "total_vectors": stats.total_vector_count,
            "dimension": self.dimension,
            "metric": self.metric,
            "namespaces": stats.namespaces
        }


class VectorStoreFactory:
    """Factory for creating vector store instances"""

    @staticmethod
    def create(
        provider: str = "faiss",
        dimension: int = 1536,
        **kwargs
    ) -> VectorStoreBase:
        """
        Create a vector store instance.

        Args:
            provider: "faiss" or "pinecone"
            dimension: Embedding dimension
            **kwargs: Provider-specific arguments

        Returns:
            VectorStoreBase instance
        """
        if provider.lower() == "faiss":
            return FAISSVectorStore(dimension=dimension, **kwargs)
        elif provider.lower() == "pinecone":
            return PineconeVectorStore(dimension=dimension, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")


def generate_document_id(content: str) -> str:
    """Generate unique document ID from content hash"""
    return hashlib.md5(content.encode()).hexdigest()[:12]


if __name__ == "__main__":
    # Demo usage with FAISS
    print("=== Vector Store Demo ===\n")

    # Create FAISS store
    store = VectorStoreFactory.create("faiss", dimension=4)

    # Create sample documents with mock embeddings
    docs = [
        Document(
            id=generate_document_id("Machine learning basics"),
            content="Machine learning is a subset of AI that enables computers to learn from data.",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"category": "AI", "source": "tutorial"}
        ),
        Document(
            id=generate_document_id("Deep learning intro"),
            content="Deep learning uses neural networks with multiple layers to learn complex patterns.",
            embedding=[0.15, 0.25, 0.35, 0.45],
            metadata={"category": "AI", "source": "tutorial"}
        ),
        Document(
            id=generate_document_id("Python programming"),
            content="Python is a versatile programming language popular in data science and web development.",
            embedding=[0.5, 0.4, 0.3, 0.2],
            metadata={"category": "Programming", "source": "guide"}
        ),
    ]

    # Add documents
    added = store.add_documents(docs)
    print(f"Added {added} documents to FAISS\n")

    # Search
    query_embedding = [0.12, 0.22, 0.32, 0.42]
    results = store.search(query_embedding, top_k=2)

    print("Search Results:")
    for result in results:
        print(f"  #{result.rank} (score: {result.score:.4f})")
        print(f"     {result.document.content[:60]}...")
        print()

    # Get stats
    stats = store.get_stats()
    print(f"Store Stats: {json.dumps(stats, indent=2)}")
