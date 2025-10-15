# FAISS vector store management
import os
import pickle
from typing import List, Dict, Any
import numpy as np
from app.embeddings import get_embeddings
from app.utils import generate_passage_id

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class VectorStore:
    def __init__(self, store_path: str = "vector_store"):
        self.store_path = store_path
        self.index = None
        self.documents = []
        self.dimension = 384  # Default for sentence-transformers all-MiniLM-L6-v2
        
        if FAISS_AVAILABLE:
            self.load_store()
        else:
            print("FAISS not available, using simple similarity search")
    
    def load_store(self):
        """Load existing vector store or create new one"""
        index_path = os.path.join(self.store_path, "index.faiss")
        docs_path = os.path.join(self.store_path, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Loaded vector store with {len(self.documents)} documents")
                # Sanity-check: if index and documents counts mismatch, rebuild index from documents
                try:
                    idx_count = int(self.index.ntotal)
                except Exception:
                    idx_count = None

                if idx_count is None or idx_count != len(self.documents):
                    print(f"\u26a0\ufe0f FAISS index size ({idx_count}) does not match documents ({len(self.documents)}). Rebuilding index from documents...")
                    self._rebuild_index_from_documents()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.documents = []

    def _rebuild_index_from_documents(self):
        """Rebuild FAISS index using current self.documents."""
        if not FAISS_AVAILABLE:
            return

        if not self.documents:
            # create empty index with default dimension
            self.index = faiss.IndexFlatIP(self.dimension)
            return

        try:
            texts = [doc['content'] for doc in self.documents]
            all_embeddings = get_embeddings(texts)
            all_embeddings_array = np.array(all_embeddings).astype('float32')
            # update dimension and create index
            self.dimension = all_embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(all_embeddings_array)
            self.index.add(all_embeddings_array)
            print(f"Rebuilt FAISS index with {len(self.documents)} documents and dimension {self.dimension}")
        except Exception as e:
            print(f"Failed to rebuild FAISS index: {e}")
            # fallback to empty index
            self.index = faiss.IndexFlatIP(self.dimension)
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        if not documents:
            return
        
        texts = [doc['content'] for doc in documents]
        embeddings = get_embeddings(texts)

        # Extend documents first so any rebuild uses full set
        self.documents.extend(documents)

        if FAISS_AVAILABLE and self.index is not None:
            try:
                # Normalize embeddings for cosine similarity
                embeddings_array = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)

                # If embedding dimension changed, rebuild index from all documents
                if embeddings_array.shape[1] != self.dimension:
                    print(f"Embedding dimension changed ({self.dimension} -> {embeddings_array.shape[1]}). Rebuilding index from all documents...")
                    self._rebuild_index_from_documents()
                else:
                    # Safe to just add the new embeddings
                    self.index.add(embeddings_array)
            except Exception as e:
                print(f"Error adding documents to FAISS index: {e}. Attempting to rebuild index.")
                self._rebuild_index_from_documents()

        self.save_store()
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.documents:
            return []
        
        query_embedding = get_embeddings([query])[0]
        
        if FAISS_AVAILABLE and self.index is not None:
            return self._faiss_search(query_embedding, k)
        else:
            return self._simple_search(query_embedding, k)
    
    def _faiss_search(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Search using FAISS index"""
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, min(k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Guard against invalid indices (mismatches between index and documents)
            if idx is None:
                continue
            try:
                idx_int = int(idx)
            except Exception:
                continue

            if idx_int < 0:
                continue

            if idx_int >= len(self.documents):
                # Skip invalid index and warn
                print(f"\u26a0\ufe0f Received FAISS index {idx_int} >= documents length {len(self.documents)}; skipping")
                continue

            doc = self.documents[idx_int].copy()
            doc['id'] = generate_passage_id(doc['content'], doc.get('source', ''))
            doc['score'] = float(score)
            results.append(doc)
        
        return results
    
    def _simple_search(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Simple cosine similarity search without FAISS"""
        query_array = np.array(query_embedding)
        similarities = []
        
        for i, doc in enumerate(self.documents):
            doc_embedding = get_embeddings([doc['content']])[0]
            doc_array = np.array(doc_embedding)
            
            # Cosine similarity
            similarity = np.dot(query_array, doc_array) / (np.linalg.norm(query_array) * np.linalg.norm(doc_array))
            similarities.append((similarity, i))
        
        # Sort by similarity and get top k
        similarities.sort(reverse=True)
        results = []
        
        for similarity, idx in similarities[:k]:
            doc = self.documents[idx].copy()
            doc['id'] = generate_passage_id(doc['content'], doc.get('source', ''))
            doc['score'] = float(similarity)
            results.append(doc)
        
        return results
    
    def save_store(self):
        """Save vector store to disk"""
        os.makedirs(self.store_path, exist_ok=True)
        
        if FAISS_AVAILABLE and self.index is not None:
            index_path = os.path.join(self.store_path, "index.faiss")
            faiss.write_index(self.index, index_path)
        
        docs_path = os.path.join(self.store_path, "documents.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)

# Global vector store instance
vector_store = VectorStore()

def add_documents_to_store(documents: List[Dict[str, Any]]):
    """Add documents to the global vector store"""
    vector_store.add_documents(documents)

def search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search documents in the global vector store"""
    return vector_store.search(query, k)