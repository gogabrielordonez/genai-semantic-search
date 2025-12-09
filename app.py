"""
GenAI Semantic Search - Streamlit Application
Full-stack application for semantic search with Pinecone/FAISS.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from search_engine import SemanticSearchEngine, KnowledgeBase
from vector_store import generate_document_id


# Page config
st.set_page_config(
    page_title="Semantic Search | Gabriel Ordonez",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .search-result {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #0f3460;
        margin-bottom: 1rem;
        border-left: 4px solid #00e5ff;
    }
    .score-badge {
        background: linear-gradient(135deg, #3d5afe 0%, #00e5ff 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card {
        background: rgba(61, 90, 254, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(61, 90, 254, 0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00e5ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        text-transform: uppercase;
    }
    .tag {
        background: rgba(0, 229, 255, 0.1);
        color: #00e5ff;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "kb" not in st.session_state:
        st.session_state.kb = None
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def initialize_engine(vector_provider: str, embedding_provider: str):
    """Initialize or reinitialize the search engine"""
    dimension = 384 if embedding_provider == "sentence-transformers" else 1536

    st.session_state.engine = SemanticSearchEngine(
        vector_provider=vector_provider,
        embedding_provider=embedding_provider,
        dimension=dimension
    )

    st.session_state.kb = KnowledgeBase(
        vector_provider=vector_provider,
        embedding_provider=embedding_provider,
        dimension=dimension
    )

    st.session_state.initialized = True


def render_sidebar():
    """Render sidebar with configuration"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")

        # Vector store selection
        st.subheader("Vector Store")
        vector_provider = st.selectbox(
            "Provider",
            ["faiss", "pinecone"],
            help="FAISS: Local, free. Pinecone: Cloud, scalable."
        )

        # Embedding selection
        st.subheader("Embeddings")
        embedding_provider = st.selectbox(
            "Provider",
            ["mock", "sentence-transformers", "openai"],
            help="Mock: Testing. Sentence-Transformers: Free/local. OpenAI: Production."
        )

        # Initialize button
        st.markdown("---")
        if st.button("🚀 Initialize Engine", type="primary", use_container_width=True):
            with st.spinner("Initializing..."):
                initialize_engine(vector_provider, embedding_provider)
            st.success("Engine initialized!")

        # Show status
        if st.session_state.initialized:
            st.markdown("---")
            st.markdown("### Status")
            stats = st.session_state.engine.get_stats()
            st.markdown(f"**Documents:** {stats['engine']['document_count']}")
            st.markdown(f"**Vector Store:** {stats['vector_store']['provider']}")
            st.markdown(f"**Embeddings:** {stats['embedding_service']['provider']}")

        # About section
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        **GenAI Semantic Search**

        Full-stack application demonstrating:
        - 🔍 Semantic similarity search
        - 📊 Vector database integration
        - 🧠 Embedding generation
        - 📈 Real-time analytics

        Built with Python & Streamlit.
        """)

        return vector_provider, embedding_provider


def render_search_tab():
    """Render the main search interface"""
    st.header("🔍 Semantic Search")

    if not st.session_state.initialized:
        st.info("👈 Initialize the engine in the sidebar to get started.")
        return

    # Search input
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Search query",
            placeholder="Ask a question or enter search terms...",
            label_visibility="collapsed"
        )

    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5, label_visibility="collapsed")

    # Search button
    if st.button("🔍 Search", type="primary") or query:
        if query:
            with st.spinner("Searching..."):
                response = st.session_state.engine.search(query, top_k=top_k)

            # Store in history
            st.session_state.search_history.append({
                "query": query,
                "results": len(response.results),
                "time_ms": response.search_time_ms
            })

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Results Found", response.total_results)
            with col2:
                st.metric("Search Time", f"{response.search_time_ms}ms")
            with col3:
                st.metric("Embedding Time", f"{response.embedding_time_ms}ms")

            st.markdown("---")

            # Display results
            if response.results:
                for result in response.results:
                    st.markdown(f"""
                    <div class="search-result">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="color: #a0a0b0;">#{result.rank}</span>
                            <span class="score-badge">Score: {result.score:.4f}</span>
                        </div>
                        <p style="color: #ffffff; margin-bottom: 0.5rem;">{result.document.content}</p>
                        <div>
                            {"".join([f'<span class="tag">{k}: {v}</span>' for k, v in result.document.metadata.items()])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No results found. Try a different query or add more documents.")


def render_index_tab():
    """Render document indexing interface"""
    st.header("📥 Index Documents")

    if not st.session_state.initialized:
        st.info("👈 Initialize the engine in the sidebar to get started.")
        return

    tab1, tab2, tab3 = st.tabs(["Single Document", "Bulk Upload", "Sample Data"])

    with tab1:
        st.subheader("Add Single Document")

        content = st.text_area(
            "Document content",
            height=150,
            placeholder="Enter the document text to index..."
        )

        col1, col2 = st.columns(2)
        with col1:
            category = st.text_input("Category (optional)")
        with col2:
            tags = st.text_input("Tags (comma-separated)")

        if st.button("➕ Index Document"):
            if content:
                with st.spinner("Indexing..."):
                    metadata = {}
                    if category:
                        metadata["category"] = category
                    if tags:
                        metadata["tags"] = [t.strip() for t in tags.split(",")]

                    doc = st.session_state.engine.index_document(content, metadata)
                st.success(f"Document indexed with ID: {doc.id}")
            else:
                st.warning("Please enter document content.")

    with tab2:
        st.subheader("Bulk Upload")
        st.markdown("Upload a JSON file with documents in this format:")
        st.code("""[
    {"content": "Document text...", "metadata": {"category": "Tech"}},
    {"content": "Another document...", "metadata": {"source": "blog"}}
]""")

        uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

        if uploaded_file and st.button("📤 Upload & Index"):
            try:
                documents = json.load(uploaded_file)
                with st.spinner(f"Indexing {len(documents)} documents..."):
                    indexed = st.session_state.engine.index_documents(documents)
                st.success(f"Successfully indexed {indexed} documents!")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        st.subheader("Load Sample Data")
        st.markdown("Load pre-built sample documents to test the search functionality.")

        if st.button("📚 Load AI/ML Sample Data"):
            sample_docs = [
                {"content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.", "metadata": {"category": "AI", "topic": "ML Basics"}},
                {"content": "Deep learning is a type of machine learning based on artificial neural networks with multiple layers that progressively extract higher-level features from raw input.", "metadata": {"category": "AI", "topic": "Deep Learning"}},
                {"content": "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages.", "metadata": {"category": "AI", "topic": "NLP"}},
                {"content": "Convolutional Neural Networks (CNNs) are deep learning models primarily used for image recognition and computer vision tasks.", "metadata": {"category": "AI", "topic": "Computer Vision"}},
                {"content": "Transformers are a type of neural network architecture that uses self-attention mechanisms, forming the basis of models like BERT and GPT.", "metadata": {"category": "AI", "topic": "Transformers"}},
                {"content": "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models to produce more accurate and grounded responses.", "metadata": {"category": "AI", "topic": "RAG"}},
                {"content": "Vector databases are specialized databases designed to store and query high-dimensional vectors, enabling efficient similarity search.", "metadata": {"category": "Infrastructure", "topic": "Databases"}},
                {"content": "Embeddings are dense vector representations of data that capture semantic meaning, enabling similarity comparisons.", "metadata": {"category": "AI", "topic": "Embeddings"}},
                {"content": "Fine-tuning is the process of taking a pre-trained model and training it further on a specific dataset for a particular task.", "metadata": {"category": "AI", "topic": "Training"}},
                {"content": "Prompt engineering is the practice of designing and optimizing prompts to effectively communicate with large language models.", "metadata": {"category": "AI", "topic": "Prompting"}},
            ]

            with st.spinner("Loading sample data..."):
                indexed = st.session_state.engine.index_documents(sample_docs)
            st.success(f"Loaded {indexed} sample documents!")


def render_analytics_tab():
    """Render analytics dashboard"""
    st.header("📊 Analytics Dashboard")

    if not st.session_state.initialized:
        st.info("👈 Initialize the engine in the sidebar to get started.")
        return

    stats = st.session_state.engine.get_stats()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['engine']['document_count']}</div>
            <div class="metric-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['engine']['total_searches']}</div>
            <div class="metric-label">Searches</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['engine']['avg_search_time_ms']:.1f}ms</div>
            <div class="metric-label">Avg Search Time</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['embedding_service']['hit_rate']}%</div>
            <div class="metric-label">Cache Hit Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Search History")
        if st.session_state.search_history:
            history_df = pd.DataFrame(st.session_state.search_history)
            history_df["search_num"] = range(1, len(history_df) + 1)

            fig = px.line(
                history_df,
                x="search_num",
                y="time_ms",
                title="Search Latency Over Time",
                markers=True
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
                xaxis_title="Search #",
                yaxis_title="Time (ms)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Perform searches to see latency trends.")

    with col2:
        st.subheader("System Stats")

        # Create stats breakdown
        stats_data = [
            {"Component": "Vector Store", "Value": stats['vector_store']['provider']},
            {"Component": "Total Vectors", "Value": stats['vector_store']['total_vectors']},
            {"Component": "Embedding Provider", "Value": stats['embedding_service']['provider']},
            {"Component": "Dimension", "Value": stats['engine']['dimension']},
            {"Component": "Cache Size", "Value": stats['embedding_service']['cache_size']},
        ]

        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # Raw stats
    with st.expander("📋 Raw Statistics"):
        st.json(stats)


def main():
    """Main application entry point"""
    init_session_state()

    st.title("🔍 GenAI Semantic Search")
    st.markdown("*Full-stack semantic search with vector databases*")

    vector_provider, embedding_provider = render_sidebar()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📥 Index", "📊 Analytics"])

    with tab1:
        render_search_tab()

    with tab2:
        render_index_tab()

    with tab3:
        render_analytics_tab()


if __name__ == "__main__":
    main()
