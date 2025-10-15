# Pytest configuration and fixtures
import pytest
import tempfile
import os
from app.vectorstore import vector_store

@pytest.fixture(autouse=True)
def reset_vector_store():
    """Reset the vector store before each test to ensure clean state"""
    # Clear the global vector store
    vector_store.documents = []
    vector_store.index = None
    
    # Recreate the index if FAISS is available
    try:
        import faiss
        vector_store.index = faiss.IndexFlatIP(vector_store.dimension)
    except ImportError:
        pass
    
    yield
    
    # Clean up after test
    vector_store.documents = []
    vector_store.index = None

@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample text file for testing purposes. It contains some basic information about machine learning and artificial intelligence.")
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            'content': "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            'source': 'ml_basics.txt_chunk_0'
        },
        {
            'content': "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
            'source': 'deep_learning.txt_chunk_0'
        },
        {
            'content': "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
            'source': 'nlp_intro.txt_chunk_0'
        }
    ]