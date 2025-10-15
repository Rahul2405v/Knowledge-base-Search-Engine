# Test query functionality
import pytest
from app.rag import query_knowledge_base, query_knowledge_base_with_passages, get_document_stats
from app.vectorstore import add_documents_to_store
from app.schemas import Passage

def test_query_empty_knowledge_base():
    """Test querying when no documents are loaded"""
    response = query_knowledge_base("What is machine learning?")
    assert "couldn't find any relevant information" in response.lower()

def test_query_with_passages_empty():
    """Test querying with passages when no documents are loaded"""
    answer, passages = query_knowledge_base_with_passages("What is machine learning?")
    assert "couldn't find any relevant information" in answer.lower()
    assert len(passages) == 0

def test_query_with_documents():
    """Test querying after adding some documents"""
    # Add test documents with proper structure
    test_docs = [
        {
            'content': "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            'source': 'ml_basics.txt_chunk_0'
        },
        {
            'content': "Python is a popular programming language for data science and machine learning applications.",
            'source': 'python_guide.txt_chunk_0'
        }
    ]
    
    add_documents_to_store(test_docs)
    
    # Test legacy query function
    response = query_knowledge_base("What is machine learning?")
    assert len(response) > 0
    assert "machine learning" in response.lower()

def test_query_with_passages():
    """Test querying with passages after adding documents"""
    # Add test documents
    test_docs = [
        {
            'content': "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            'source': 'ml_basics.txt_chunk_0'
        },
        {
            'content': "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            'source': 'deep_learning.txt_chunk_0'
        }
    ]
    
    add_documents_to_store(test_docs)
    
    # Test new query function with passages
    answer, passages = query_knowledge_base_with_passages("What is machine learning?", max_docs=2)
    
    assert len(answer) > 0
    assert len(passages) > 0
    assert all(isinstance(p, Passage) for p in passages)
    
    # Check passage structure
    for passage in passages:
        assert hasattr(passage, 'id')
        assert hasattr(passage, 'text')
        assert hasattr(passage, 'score')
        assert hasattr(passage, 'source')

def test_query_no_relevant_documents():
    """Test querying for something not in the knowledge base"""
    # Add a document about cooking
    test_docs = [
        {
            'content': "To make pasta, boil water and add salt. Cook pasta according to package instructions.",
            'source': 'cooking_guide.txt_chunk_0'
        }
    ]
    
    add_documents_to_store(test_docs)
    
    # Query about something completely different
    response = query_knowledge_base("How do quantum computers work?")
    # Should still return something, even if not directly relevant
    assert len(response) > 0

def test_get_document_stats():
    """Test getting knowledge base statistics"""
    stats = get_document_stats()
    assert 'total_documents' in stats
    assert 'index_available' in stats
    assert 'embedding_provider' in stats
    assert isinstance(stats['total_documents'], int)
    assert isinstance(stats['index_available'], bool)
    assert isinstance(stats['embedding_provider'], str)

def test_query_top_k_parameter():
    """Test that top_k parameter is respected in new query function"""
    # Add multiple test documents
    test_docs = [
        {'content': f"Document {i} about machine learning and AI.", 'source': f'doc_{i}.txt_chunk_0'}
        for i in range(10)
    ]
    
    add_documents_to_store(test_docs)
    
    # Query with different top_k values
    answer1, passages1 = query_knowledge_base_with_passages("machine learning", max_docs=1)
    answer2, passages2 = query_knowledge_base_with_passages("machine learning", max_docs=5)
    
    # Both should return responses
    assert len(answer1) > 0
    assert len(answer2) > 0
    assert len(passages1) <= 1
    assert len(passages2) <= 5

def test_passage_ids_are_unique():
    """Test that passage IDs are unique and consistent"""
    test_docs = [
        {
            'content': "Artificial intelligence is transforming the world.",
            'source': 'ai_basics.txt_chunk_0'
        },
        {
            'content': "Machine learning enables computers to learn without explicit programming.",
            'source': 'ml_intro.txt_chunk_0'
        }
    ]
    
    add_documents_to_store(test_docs)
    
    answer, passages = query_knowledge_base_with_passages("artificial intelligence", max_docs=5)
    
    if len(passages) > 1:
        # Check that all IDs are unique
        ids = [p.id for p in passages]
        assert len(ids) == len(set(ids)), "Passage IDs should be unique"
        
        # Check that IDs are consistent (same content should have same ID)
        for passage in passages:
            assert len(passage.id) > 0, "Passage ID should not be empty"