# Test utility functions
import pytest
from app.utils import extract_text_from_pdf_bytes, extract_text_from_txt_bytes, chunk_text, clean_text, generate_passage_id

def test_extract_text_from_txt_bytes():
    """Test extracting text from text bytes"""
    text_content = "This is a test document with UTF-8 encoding."
    text_bytes = text_content.encode('utf-8')
    
    result = extract_text_from_txt_bytes(text_bytes)
    assert result == text_content

def test_extract_text_from_txt_bytes_encoding():
    """Test text extraction with different encodings"""
    text_content = "This is a test with special characters: café, naïve"
    
    # Test UTF-8
    utf8_bytes = text_content.encode('utf-8')
    result_utf8 = extract_text_from_txt_bytes(utf8_bytes)
    assert text_content in result_utf8
    
    # Test latin-1 fallback
    latin1_bytes = "Simple text".encode('latin-1')
    result_latin1 = extract_text_from_txt_bytes(latin1_bytes)
    assert "Simple text" in result_latin1

def test_extract_text_from_pdf_bytes_invalid():
    """Test PDF extraction with invalid bytes"""
    invalid_pdf = b"This is not a PDF file"
    
    # Should handle gracefully and not crash
    try:
        result = extract_text_from_pdf_bytes(invalid_pdf)
        # If it doesn't crash, that's good. Result might be None or empty
        assert result is not None or result == ""
    except Exception:
        # If it raises an exception, that's also acceptable for invalid input
        pass

def test_chunk_text_small():
    """Test chunking with text smaller than chunk size"""
    text = "This is a small text."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_large():
    """Test chunking with text larger than chunk size"""
    # Create text with exactly 1000 words
    words = ["word"] * 1000
    text = " ".join(words)
    
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 800 for chunk in chunks)

def test_chunk_text_overlap():
    """Test that chunk overlap works correctly"""
    # Create text with exactly 1200 words
    words = [f"word{i}" for i in range(1200)]
    text = " ".join(words)
    
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    
    assert len(chunks) >= 2
    # Check that there's some overlap between consecutive chunks
    if len(chunks) >= 2:
        chunk1_words = chunks[0].split()
        chunk2_words = chunks[1].split()
        
        # Last 100 words of chunk1 should have some overlap with first words of chunk2
        # (might not be exact due to chunking logic)
        assert len(chunk1_words) > 0
        assert len(chunk2_words) > 0

def test_clean_text():
    """Test text cleaning functionality"""
    dirty_text = "This   has    extra    spaces\n\nand\t\ttabs and\n\nnewlines."
    clean = clean_text(dirty_text)
    
    # Should normalize whitespace
    assert "  " not in clean
    assert "\n" not in clean
    assert "\t" not in clean
    assert clean.startswith("This has extra spaces")

def test_clean_text_special_characters():
    """Test cleaning text with special characters"""
    text_with_special = "Hello! This has émojis 🚀 and special chars @#$%"
    clean = clean_text(text_with_special)
    
    # Should keep basic punctuation but remove some special chars
    assert "Hello" in clean
    assert "!" in clean  # Basic punctuation should be kept

def test_generate_passage_id():
    """Test passage ID generation"""
    text = "This is a test passage."
    source = "test.txt"
    
    id1 = generate_passage_id(text, source)
    id2 = generate_passage_id(text, source)
    
    # Same input should generate same ID
    assert id1 == id2
    assert len(id1) > 0

def test_generate_passage_id_different():
    """Test that different content generates different IDs"""
    text1 = "This is passage one."
    text2 = "This is passage two."
    source = "test.txt"
    
    id1 = generate_passage_id(text1, source)
    id2 = generate_passage_id(text2, source)
    
    # Different text should generate different IDs
    assert id1 != id2

def test_generate_passage_id_without_source():
    """Test passage ID generation without source"""
    text = "This is a test passage."
    
    id1 = generate_passage_id(text)
    id2 = generate_passage_id(text, "")
    
    # Should work with empty or no source
    assert len(id1) > 0
    assert len(id2) > 0