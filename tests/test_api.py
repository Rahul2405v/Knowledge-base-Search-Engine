# Test API endpoints
import pytest
from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Knowledge Base Search Backend"}

def test_query_endpoint_empty():
    """Test query endpoint with empty knowledge base"""
    response = client.post(
        "/query",
        json={"q": "What is machine learning?", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "passages" in data
    assert isinstance(data["passages"], list)

def test_query_endpoint_invalid_request():
    """Test query endpoint with invalid request"""
    response = client.post(
        "/query",
        json={"invalid_field": "test"}
    )
    assert response.status_code == 422  # Validation error

def test_ingest_endpoint_text_file():
    """Test ingesting a text file via API"""
    # Create a test text file
    text_content = "This is a test document for API ingestion."
    text_file = io.BytesIO(text_content.encode('utf-8'))
    
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "inserted" in data
    assert "message" in data
    assert data["inserted"] == 1

def test_ingest_endpoint_unsupported_file():
    """Test ingesting an unsupported file type"""
    # Create a test file with unsupported extension
    content = "This is not a supported file type."
    file_obj = io.BytesIO(content.encode('utf-8'))
    
    response = client.post(
        "/ingest",
        files={"file": ("test.docx", file_obj, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    
    assert response.status_code == 400
    assert "Only PDF and TXT files are supported" in response.json()["detail"]

def test_query_after_ingestion():
    """Test querying after ingesting a document"""
    # First ingest a document
    text_content = "Machine learning is a subset of artificial intelligence."
    text_file = io.BytesIO(text_content.encode('utf-8'))
    
    ingest_response = client.post(
        "/ingest",
        files={"file": ("ml_info.txt", text_file, "text/plain")}
    )
    assert ingest_response.status_code == 200
    
    # Then query for information
    query_response = client.post(
        "/query",
        json={"q": "What is machine learning?", "top_k": 3}
    )
    
    assert query_response.status_code == 200
    data = query_response.json()
    assert len(data["answer"]) > 0
    # Should find the relevant document
    assert len(data["passages"]) > 0

def test_query_response_structure():
    """Test that query response has correct structure"""
    response = client.post(
        "/query",
        json={"q": "test query", "top_k": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check answer field
    assert "answer" in data
    assert isinstance(data["answer"], str)
    
    # Check passages field
    assert "passages" in data
    assert isinstance(data["passages"], list)
    
    # If there are passages, check their structure
    for passage in data["passages"]:
        assert "id" in passage
        assert "text" in passage
        assert "score" in passage
        assert "source" in passage
        assert isinstance(passage["id"], str)
        assert isinstance(passage["text"], str)
        assert isinstance(passage["score"], (int, float))

def test_ingest_response_structure():
    """Test that ingest response has correct structure"""
    text_content = "Test document for structure validation."
    text_file = io.BytesIO(text_content.encode('utf-8'))
    
    response = client.post(
        "/ingest",
        files={"file": ("structure_test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "inserted" in data
    assert "message" in data
    assert isinstance(data["inserted"], int)
    assert isinstance(data["message"], str)
    assert data["inserted"] >= 0

def test_query_with_default_top_k():
    """Test query with default top_k parameter"""
    response = client.post(
        "/query",
        json={"q": "test query"}  # No top_k specified
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "passages" in data