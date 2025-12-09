"""
Embedding Service
Unified interface for generating text embeddings using various providers.
Supports OpenAI, Sentence Transformers, and mock embeddings for testing.
"""

import os
import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np

# Optional imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    tokens_used: Optional[int] = None


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension"""
        pass


class OpenAIEmbeddings(EmbeddingProviderBase):
    """
    OpenAI embedding provider using text-embedding-3-small or ada-002.
    Production-grade embeddings with high semantic accuracy.
    """

    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Run: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.model = model
        self.dimension = self.MODELS.get(model, 1536)
        self.client = openai.OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return EmbeddingResult(
            text=text,
            embedding=response.data[0].embedding,
            model=self.model,
            dimension=self.dimension,
            tokens_used=response.usage.total_tokens
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        results = []
        for i, data in enumerate(response.data):
            results.append(EmbeddingResult(
                text=texts[i],
                embedding=data.embedding,
                model=self.model,
                dimension=self.dimension,
                tokens_used=response.usage.total_tokens // len(texts)
            ))

        return results

    def get_dimension(self) -> int:
        return self.dimension


class SentenceTransformerEmbeddings(EmbeddingProviderBase):
    """
    Local embedding provider using Sentence Transformers.
    Free, fast, and privacy-preserving (no API calls).
    """

    MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers not installed. Run: pip install sentence-transformers")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.MODELS.get(model_name, 384)

    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)

        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model=self.model_name,
            dimension=self.dimension
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        results = []
        for i, embedding in enumerate(embeddings):
            results.append(EmbeddingResult(
                text=texts[i],
                embedding=embedding.tolist(),
                model=self.model_name,
                dimension=self.dimension
            ))

        return results

    def get_dimension(self) -> int:
        return self.dimension


class MockEmbeddings(EmbeddingProviderBase):
    """
    Mock embedding provider for testing without API costs.
    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.model_name = "mock-embeddings"

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding from text hash"""
        # Use hash to seed random generator for reproducibility
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))

        # Generate normalized embedding
        embedding = np.random.randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    def embed(self, text: str) -> EmbeddingResult:
        """Generate mock embedding"""
        return EmbeddingResult(
            text=text,
            embedding=self._generate_mock_embedding(text),
            model=self.model_name,
            dimension=self.dimension,
            tokens_used=len(text.split())
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate mock embeddings for batch"""
        return [self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        return self.dimension


class EmbeddingService:
    """
    High-level embedding service with caching and provider abstraction.
    """

    def __init__(
        self,
        provider: str = "mock",
        cache_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize embedding service.

        Args:
            provider: "openai", "sentence-transformers", or "mock"
            cache_enabled: Enable in-memory embedding cache
            **kwargs: Provider-specific arguments
        """
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, List[float]] = {}
        self.stats = {"hits": 0, "misses": 0, "total_tokens": 0}

        # Initialize provider
        if provider == "openai":
            self.provider = OpenAIEmbeddings(**kwargs)
        elif provider == "sentence-transformers":
            self.provider = SentenceTransformerEmbeddings(**kwargs)
        else:
            self.provider = MockEmbeddings(**kwargs)

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, text: str) -> List[float]:
        """Get embedding for text (with caching)"""
        cache_key = self._cache_key(text)

        if self.cache_enabled and cache_key in self.cache:
            self.stats["hits"] += 1
            return self.cache[cache_key]

        self.stats["misses"] += 1
        result = self.provider.embed(text)

        if result.tokens_used:
            self.stats["total_tokens"] += result.tokens_used

        if self.cache_enabled:
            self.cache[cache_key] = result.embedding

        return result.embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if self.cache_enabled and cache_key in self.cache:
                self.stats["hits"] += 1
                results.append((i, self.cache[cache_key]))
            else:
                self.stats["misses"] += 1
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get embeddings for uncached texts
        if uncached_texts:
            new_results = self.provider.embed_batch(uncached_texts)
            for idx, result in zip(uncached_indices, new_results):
                if result.tokens_used:
                    self.stats["total_tokens"] += result.tokens_used
                if self.cache_enabled:
                    self.cache[self._cache_key(result.text)] = result.embedding
                results.append((idx, result.embedding))

        # Sort by original index and return embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.provider.get_dimension()

    def get_stats(self) -> Dict:
        """Get service statistics"""
        hit_rate = 0
        total = self.stats["hits"] + self.stats["misses"]
        if total > 0:
            hit_rate = self.stats["hits"] / total

        return {
            "provider": type(self.provider).__name__,
            "dimension": self.get_dimension(),
            "cache_size": len(self.cache),
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"],
            "hit_rate": round(hit_rate * 100, 2),
            "total_tokens": self.stats["total_tokens"]
        }

    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()


if __name__ == "__main__":
    # Demo usage
    print("=== Embedding Service Demo ===\n")

    # Use mock embeddings for demo
    service = EmbeddingService(provider="mock", dimension=384)

    texts = [
        "Machine learning is transforming industries.",
        "Deep learning uses neural networks.",
        "Python is great for data science."
    ]

    print("Generating embeddings...")
    embeddings = service.embed_batch(texts)

    for text, emb in zip(texts, embeddings):
        print(f"  '{text[:40]}...' -> [{emb[0]:.4f}, {emb[1]:.4f}, ... ] (dim={len(emb)})")

    print(f"\nStats: {service.get_stats()}")

    # Test caching
    print("\nTesting cache...")
    _ = service.embed(texts[0])  # Should hit cache
    print(f"After re-embedding: {service.get_stats()}")
