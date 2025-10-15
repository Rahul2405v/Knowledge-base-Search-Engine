import io
import hashlib
import re
from typing import List
from pathlib import Path

# PDF extraction: uses PyPDF2
import PyPDF2


def extract_text_from_pdf_bytes(b: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(b))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap, preserving line structure for structured files"""
    # For structured files like requirements.txt, preserve line breaks
    if any(pattern in text for pattern in ['>=', '==', 'import ', 'def ', 'class ', '#']):
        # Split by lines first for better structure
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for line in lines:
            line_words = len(line.split())
            
            # If adding this line would exceed chunk_size, save current chunk
            if current_word_count + line_words > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # Keep overlap (last few lines)
                overlap_lines = current_chunk[-max(1, overlap//50):] if overlap > 0 else []
                current_chunk = overlap_lines + [line]
                current_word_count = sum(len(l.split()) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_word_count += line_words
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Default word-based chunking for regular text
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text.strip()


def generate_passage_id(text: str, source: str = "") -> str:
    """Generate unique ID for a text passage"""
    content = f"{source}:{text}"
    return hashlib.md5(content.encode()).hexdigest()


def extract_text_from_txt_bytes(b: bytes) -> str:
    """Extract text from text file bytes"""
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return b.decode('latin-1')
        except UnicodeDecodeError:
            return b.decode('utf-8', errors='ignore')