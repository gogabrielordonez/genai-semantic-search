# GenAI Semantic Search

Full-stack semantic search application with vector database integration. Built with Python, Streamlit, and production-ready vector stores.

## Features

- **Semantic Search**: Find documents by meaning, not just keywords
- **Multiple Vector Stores**: Seamless switching between FAISS (local) and Pinecone (cloud)
- **Flexible Embeddings**: Support for OpenAI, Sentence Transformers, or mock embeddings
- **Interactive UI**: Streamlit dashboard with search, indexing, and analytics
- **Production Ready**: Caching, batch processing, and comprehensive statistics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
├─────────────────────────────────────────────────────────────┤
│                Semantic Search Engine                        │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Embedding       │    │ Vector Store Abstraction        │ │
│  │ Service         │    │ ┌───────────┐ ┌───────────────┐ │ │
│  │ • OpenAI        │    │ │   FAISS   │ │   Pinecone    │ │ │
│  │ • Sentence-     │    │ │  (local)  │ │   (cloud)     │ │ │
│  │   Transformers  │    │ └───────────┘ └───────────────┘ │ │
│  │ • Mock          │    └─────────────────────────────────┘ │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gogabrielordonez/genai-semantic-search.git
cd genai-semantic-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

For OpenAI embeddings:
```bash
export OPENAI_API_KEY="your-api-key"
```

For Pinecone vector store:
```bash
export PINECONE_API_KEY="your-api-key"
```

### Run the Application

```bash
streamlit run app.py
```

## Usage

### 1. Initialize Engine
Select your preferred vector store (FAISS or Pinecone) and embedding provider in the sidebar, then click "Initialize Engine".

### 2. Index Documents
- **Single Document**: Add individual documents with optional metadata
- **Bulk Upload**: Upload JSON files with multiple documents
- **Sample Data**: Load pre-built AI/ML sample documents for testing

### 3. Search
Enter natural language queries to find semantically similar documents. Results include similarity scores and metadata.

### 4. Analytics
View real-time statistics including document counts, search latency, and cache performance.

## API Reference

### SemanticSearchEngine

```python
from search_engine import SemanticSearchEngine

# Initialize with FAISS and mock embeddings (for testing)
engine = SemanticSearchEngine(
    vector_provider="faiss",
    embedding_provider="mock",
    dimension=384
)

# Index documents
engine.index_document("Your document text", {"category": "AI"})

# Search
results = engine.search("What is machine learning?", top_k=5)
for result in results.results:
    print(f"Score: {result.score:.4f} - {result.document.content}")
```

### KnowledgeBase

```python
from search_engine import KnowledgeBase

# Higher-level interface with categorization
kb = KnowledgeBase(vector_provider="faiss", embedding_provider="sentence-transformers")

# Add knowledge entries
kb.add_knowledge(
    content="Machine learning enables computers to learn from data.",
    category="AI",
    tags=["ml", "fundamentals"]
)

# Query with optional category filter
results = kb.query("How do computers learn?", category="AI")
```

### Vector Store Direct Access

```python
from vector_store import VectorStoreFactory, Document

# Create store
store = VectorStoreFactory.create("faiss", dimension=1536)

# Add documents with embeddings
doc = Document(
    id="doc1",
    content="Sample content",
    embedding=[0.1, 0.2, ...],  # Your embedding vector
    metadata={"source": "api"}
)
store.add_documents([doc])

# Search
results = store.search(query_embedding, top_k=10)
```

## Performance

| Metric | FAISS (Local) | Pinecone (Cloud) |
|--------|---------------|------------------|
| Index Speed | ~1000 docs/sec | ~500 docs/sec |
| Search Latency | <10ms | ~50-100ms |
| Scalability | Millions of vectors | Billions of vectors |
| Cost | Free | Pay-per-use |

## Tech Stack

- **Frontend**: Streamlit
- **Vector Stores**: FAISS, Pinecone
- **Embeddings**: OpenAI text-embedding-3-small, Sentence Transformers
- **Visualization**: Plotly
- **Language**: Python 3.9+

## Project Structure

```
genai-semantic-search/
├── app.py              # Streamlit UI
├── search_engine.py    # Core search engine
├── embeddings.py       # Embedding service providers
├── vector_store.py     # Vector store abstraction
├── requirements.txt    # Dependencies
└── README.md
```

## License

MIT License - see LICENSE file for details.

## Author

Gabriel Ordonez - [gabrielordonez.com](https://gabrielordonez.com)
