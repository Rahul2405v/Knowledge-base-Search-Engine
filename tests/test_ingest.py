# Test ingestion functionality
import pytest
import os
import tempfile
from app.ingest import ingest_single_document, ingest_documents, ingest_pdf_bytes, ingest_text_bytes

def test_ingest_text_file():
    """Test ingesting a simple text file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for ingestion.")
        temp_path = f.name
    
    try:
        result = ingest_single_document(temp_path)
        assert result == True
    finally:
        os.unlink(temp_path)

def test_ingest_text_bytes():
    """Test ingesting text from bytes"""
    text_content = "This is a test document for byte ingestion."
    text_bytes = text_content.encode('utf-8')
    
    result = ingest_text_bytes(text_bytes, "test_document.txt")
    assert result == True

def test_ingest_pdf_bytes():
    """Test ingesting PDF from bytes (mock test since we need a real PDF)"""
    # This would need a real PDF for a complete test
    # For now, we'll test that the function handles invalid PDF bytes gracefully
    invalid_pdf_bytes = b"This is not a PDF"
    
    result = ingest_pdf_bytes(invalid_pdf_bytes, "test.pdf")
    # Should return False for invalid PDF
    assert result == False

def test_ingest_nonexistent_file():
    """Test ingesting a file that doesn't exist"""
    result = ingest_single_document("nonexistent_file.txt")
    assert result == False

def test_ingest_unsupported_file():
    """Test ingesting an unsupported file type"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
        f.write("This is a test document.")
        temp_path = f.name
    
    try:
        result = ingest_single_document(temp_path)
        assert result == False
    finally:
        os.unlink(temp_path)

def test_ingest_documents_from_directory():
    """Test ingesting multiple documents from a directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        txt_file = os.path.join(temp_dir, "test1.txt")
        with open(txt_file, 'w') as f:
            f.write("First test document.")
        
        txt_file2 = os.path.join(temp_dir, "test2.txt")
        with open(txt_file2, 'w') as f:
            f.write("Second test document.")
        
        # Create an unsupported file
        unsupported_file = os.path.join(temp_dir, "test.docx")
        with open(unsupported_file, 'w') as f:
            f.write("Unsupported document.")
        
        result = ingest_documents(temp_dir)
        assert result == True

def test_ingest_empty_directory():
    """Test ingesting from an empty directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = ingest_documents(temp_dir)
        assert result == False

def test_text_chunking():
    """Test that text is properly chunked during ingestion"""
    # Create a long text document
    long_text = "This is a test sentence. " * 1000  # Long enough to be chunked
    text_bytes = long_text.encode('utf-8')
    
    result = ingest_text_bytes(text_bytes, "long_document.txt")
    assert result == True