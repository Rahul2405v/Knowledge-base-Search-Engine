
"""Embedding provider with multiple backends and a simple fallback.

This module exposes `get_embeddings(texts)` and creates a global
`embedding_provider` instance.
"""

import os
import re
import hashlib
from typing import List
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


class EmbeddingProvider:
    def __init__(self):
        self.openai_client = None
        self.sentence_model = None
        self.groq_client = None

        # Initialize sentence-transformers model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading sentence-transformers model...")
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                print(" Sentence transformers model loaded successfully!")
            except Exception as e:
                print(f" Failed to load sentence-transformers: {e}")


        if GROQ_AVAILABLE:
            groq_api_key =  os.getenv("GROQ_API_KEY")
            if groq_api_key:
                try:
                    self.groq_client = groq.Groq(api_key=groq_api_key)
                    print(" Groq client initialized")
                except Exception as e:
                    print(f" Failed to initialize Groq client: {e}")

        if not any([self.sentence_model, self.openai_client, self.groq_client]):
            print(" No embedding or LLM providers available, using simple fallback")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts using the best available provider."""
        if self.sentence_model:
            print(" Using sentence-transformers for embeddings")
            return self._get_sentence_embeddings(texts)
        if OPENAI_AVAILABLE and self.openai_client:
            print(" Using OpenAI for embeddings")
            return self._get_openai_embeddings(texts)
        # Groq doesn't provide embeddings in this project; fall back to simple
        print(" Using simple fallback embeddings")
        return self._get_simple_embeddings(texts)

    def _get_sentence_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.sentence_model.encode(texts)
        return embeddings.tolist()

    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                resp = self.openai_client.embeddings.create(input=text, model="text-embedding-3-small")
                embeddings.append(resp.data[0].embedding)
            except Exception as e:
                print(f" OpenAI embedding error: {e}")
                embeddings.append(self._get_simple_embeddings([text])[0])
        return embeddings

    def _get_simple_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Very small hash-based embedding fallback (deterministic)."""
        embeddings = []
        for text in texts:
            words = re.findall(r"\b\w+\b", (text or "").lower())
            vec = [0.0] * 384
            if words:
                for word in words:
                    h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                    for i in range(3):
                        pos = (h + i * 37) % 384
                        vec[pos] += 1.0
                # text-level features
                th = int(hashlib.md5((text or "").encode()).hexdigest(), 16)
                for i in range(min(10, len(words))):
                    pos = (th + i * 41) % 384
                    vec[pos] += 0.5
                mag = sum(x * x for x in vec) ** 0.5
                if mag > 0:
                    vec = [x / mag for x in vec]
            else:
                vec[0] = 1.0
            embeddings.append(vec)
        return embeddings


# Global provider instance and convenience wrapper
embedding_provider = EmbeddingProvider()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_provider.get_embeddings(texts)
