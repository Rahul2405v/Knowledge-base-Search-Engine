# Document ingestion helpers
import os
from typing import List
from app.utils import extract_text_from_pdf_bytes, extract_text_from_txt_bytes, chunk_text, clean_text
from app.embeddings import get_embeddings
from app.vectorstore import add_documents_to_store

def ingest_documents(data_dir: str = "example_data") -> bool:
    """Ingest all documents from the data directory"""
    try:
        documents = []
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            
            if filename.endswith('.pdf'):
                with open(filepath, 'rb') as f:
                    pdf_bytes = f.read()
                text = extract_text_from_pdf_bytes(pdf_bytes)
            elif filename.endswith('.txt'):
                with open(filepath, 'rb') as f:
                    txt_bytes = f.read()
                text = extract_text_from_txt_bytes(txt_bytes)
            else:
                continue
            
            if text:
                # Clean the text
                cleaned_text = clean_text(text)
                
                # Split into chunks
                chunks = chunk_text(cleaned_text)
                
                # Create documents for each chunk
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': f"{filename}_chunk_{i}"
                    })
        
        if documents:
            add_documents_to_store(documents)
            return True
        return False
        
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        return False

def ingest_single_document(filepath: str) -> bool:
    """Ingest a single document"""
    try:
        if filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                pdf_bytes = f.read()
            text = extract_text_from_pdf_bytes(pdf_bytes)
        elif filepath.endswith('.txt'):
            with open(filepath, 'rb') as f:
                txt_bytes = f.read()
            text = extract_text_from_txt_bytes(txt_bytes)
        else:
            return False
            
        if text:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Split into chunks
            chunks = chunk_text(cleaned_text)
            
            # Create documents for each chunk
            documents = []
            filename = os.path.basename(filepath)
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': f"{filename}_chunk_{i}"
                })
            
            add_documents_to_store(documents)
            return True
        return False
        
    except Exception as e:
        print(f"Error ingesting document: {e}")
        return False

def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> bool:
    """Ingest a PDF from bytes (for file uploads)"""
    try:
        text = extract_text_from_pdf_bytes(pdf_bytes)
        if text:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Split into chunks
            chunks = chunk_text(cleaned_text)
            
            # Create documents for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': f"{filename}_chunk_{i}"
                })
            
            add_documents_to_store(documents)
            return True
        return False
        
    except Exception as e:
        print(f"Error ingesting PDF bytes: {e}")
        return False

def ingest_text_bytes(text_bytes: bytes, filename: str) -> bool:
    """Ingest text from bytes (for file uploads)"""
    try:
        text = extract_text_from_txt_bytes(text_bytes)
        if text:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Split into chunks
            chunks = chunk_text(cleaned_text)
            
            # Create documents for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': f"{filename}_chunk_{i}"
                })
            
            add_documents_to_store(documents)
            return True
        return False
        
    except Exception as e:
        print(f"Error ingesting text bytes: {e}")
        return False