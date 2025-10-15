"""Retrieval + generation orchestration.

Provides:
- query_knowledge_base_with_passages(query, max_docs, use_llm) -> (answer, passages)
- query_knowledge_base(query, max_docs)
- get_document_stats()
"""
from typing import List, Dict, Any, Tuple
from app.vectorstore import search_documents, vector_store
from app.embeddings import embedding_provider
from app.embeddings import get_embeddings
import numpy as np
import os
from app.schemas import Passage
import re


def query_knowledge_base_with_passages(query: str, max_docs: int = 5, use_llm: bool = None) -> Tuple[str, List[Passage]]:
    """Query the knowledge base and return answer with passages."""
    relevant_docs = search_documents(query, k=max_docs)

    if not relevant_docs:
        return ("I couldn't find any relevant information in the knowledge base to answer your question.", [])

    # Compute similarity scores using sentence-transformers if available, else fallback embeddings
    try:
        texts = [d.get('content', '') for d in relevant_docs]
        # Use sentence-transformers model directly if present for speed/consistency
        if getattr(embedding_provider, 'sentence_model', None):
            q_emb = embedding_provider.sentence_model.encode([query])[0]
            d_embs = embedding_provider.sentence_model.encode(texts)
        else:
            embs = get_embeddings([query] + texts)
            q_emb = np.array(embs[0])
            d_embs = [np.array(e) for e in embs[1:]]

        q_arr = np.array(q_emb)
        # normalize
        q_norm = np.linalg.norm(q_arr)
        for i, d_emb in enumerate(d_embs):
            d_arr = np.array(d_emb)
            denom = q_norm * np.linalg.norm(d_arr)
            score = float(np.dot(q_arr, d_arr) / denom) if denom > 0 else 0.0
            relevant_docs[i]['score'] = score
    except Exception as e:
        # If any error computing embeddings, leave scores as-is (may be 0)
        print(f"Error computing similarity scores: {e}")

    passages = []
    for doc in relevant_docs:
        passages.append(Passage(id=str(doc.get('id', '')), text=doc.get('content', ''), score=float(doc.get('score', 0.0)), source=doc.get('source')))

    answer = _generate_answer(query, relevant_docs, use_llm=use_llm)
    return answer, passages


def query_knowledge_base(query: str, max_docs: int = 3) -> str:
    answer, _ = query_knowledge_base_with_passages(query, max_docs)
    return answer


def _generate_answer(query: str, relevant_docs: List[Dict[str, Any]], use_llm: bool = None) -> str:
    """Choose generation strategy: LLM (Groq/OpenAI) if available or heuristic fallback."""
    # If client requested LLM and we have one, prefer that path
    if use_llm:
        if getattr(embedding_provider, 'groq_client', None):
            return _generate_with_groq(query, relevant_docs)
        if getattr(embedding_provider, 'openai_client', None):
            return _generate_with_openai(query, relevant_docs)

    # Otherwise prefer LLM if available, else heuristic
    if getattr(embedding_provider, 'openai_client', None):
        return _generate_with_openai(query, relevant_docs)
    if getattr(embedding_provider, 'groq_client', None):
        return _generate_with_groq(query, relevant_docs)

    return _generate_simple_response(query, relevant_docs)


def _generate_with_openai(query: str, docs: List[Dict[str, Any]]) -> str:
    """Use the OpenAI client wrapped by embedding_provider to produce a concise answer."""
    try:
        # Prepare context
        context = ''
        for i, d in enumerate(docs[:8], 1):
            context += f"Document {i} (Source: {d.get('source','Unknown')}):\n{d.get('content','')}\n\n"

        prompt = f"""You are a precise assistant. Use ONLY the provided CONTEXT to answer the question. If the answer cannot be found in the context, reply exactly: \"I couldn't find information to answer that question.\"\n\nCONTEXT:\n{context}\nQUESTION: {query}\n\nInstructions: Provide a one-line concise answer if available, then a short explanation and Sources: <sources>."""

        client = embedding_provider.openai_client
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role':'system','content':'You are a precise, factual assistant.'},{'role':'user','content':prompt}],
            max_tokens=800,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        return answer + _append_sources(docs)
    except Exception as e:
        print(f"OpenAI generation error: {e}")
        return _generate_simple_response(query, docs)


def _generate_with_groq(query: str, docs: List[Dict[str, Any]]) -> str:
    try:
        context = ''
        for i, d in enumerate(docs[:8], 1):
            context += f"Document {i} (Source: {d.get('source','Unknown')}):\n{d.get('content','')}\n\n"

        prompt = f"""You are a precise assistant. Use ONLY the provided CONTEXT to answer the question. If the answer cannot be found in the context, reply exactly: \"I couldn't find information to answer that question.\"\n\nCONTEXT:\n{context}\nQUESTION: {query}\n\nInstructions: Provide a one-line concise answer if available, then a short explanation and Sources: <sources>."""

        client = embedding_provider.groq_client
        # Allow overriding Groq model via env var so users can change models without editing code
        model_name = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{'role':'system','content':'You are a precise, factual assistant.'},{'role':'user','content':prompt}],
                max_tokens=800,
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
            return answer + _append_sources(docs)
        except Exception as inner_e:
            # If the model is decommissioned or invalid, surface a clear message in logs
            print(f"Groq generation error: {inner_e}")
            # Re-raise to be caught by outer except so fallback behavior still applies
            raise
    except Exception as e:
        print(f"Groq generation error: {e}")
        # If Groq failed (for example model decommissioned), fall back to simple extractor
        return _generate_simple_response(query, docs)


def _append_sources(docs: List[Dict[str, Any]]) -> str:
    sources = list({d.get('source','Unknown') for d in docs})
    if not sources:
        return ''
    if len(sources) <= 3:
        return "\n\nSources: " + ', '.join(sources)
    return "\n\nSources: " + ', '.join(sources[:3]) + f" and {len(sources)-3} more"


def _generate_simple_response(query: str, docs: List[Dict[str, Any]]) -> str:
    """A conservative fallback that attempts heuristics then returns an excerpt."""
    # Heuristic extraction for common intents
    text_pool = '\n'.join([d.get('content','') for d in docs[:5]])
    q = query.lower()

    # contact extraction
    if any(k in q for k in ['email','contact','phone','tel']):
        m = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text_pool)
        if m:
            return f"{m.group(0)}\n\nSources: {docs[0].get('source','Unknown')}"
        m2 = re.search(r'\b\d{10,}\b', text_pool)
        if m2:
            return f"{m2.group(0)}\n\nSources: {docs[0].get('source','Unknown')}"

    # where / studying extraction
    if any(k in q for k in ['where','studying','study','which college','which university']):
        for line in text_pool.splitlines():
            lw = line.lower()
            if any(w in lw for w in ['institute','university','college','school','vit','vellore']):
                return f"{line.strip()}\n\nSources: {docs[0].get('source','Unknown')}"

    # fallback: return a short excerpt from best doc
    best = max(docs, key=lambda x: x.get('score',0))
    excerpt = best.get('content','')[:800]
    return f"Based on the available documents, here's the most relevant information I found:\n\n{excerpt}...\n\nSources: {best.get('source','Unknown')}"


def get_document_stats() -> Dict[str, Any]:
    return {
        'total_documents': len(vector_store.documents),
        'index_available': vector_store.index is not None,
        'embedding_provider': (
            'OpenAI' if getattr(embedding_provider, 'openai_client', None) else
            'Groq' if getattr(embedding_provider, 'groq_client', None) else
            'SentenceTransformers' if getattr(embedding_provider, 'sentence_model', None) else
            'Fallback'
        )
    }
# app/rag.py
from app.embeddings import embedding_provider 
from typing import List, Dict, Tuple
 # must be top-level
from app.schemas import Passage

# -------------------------
# MAIN FUNCTION
# -------------------------

def query_knowledge_base_with_llm(query: str, documents: List[Dict[str, Any]]) -> Tuple[str, List[Passage]]:
    """
    Send entire documents to LLM to answer any query.
    Returns:
        answer (str) - concise answer extracted by LLM
        passages (List[Passage]) - all documents sent for reference
    """
    if not documents:
        return "No documents provided to answer the question.", []

    # Convert to Passage objects and ensure ID is string
    passages = [
        Passage(
            id=str(doc.get("id", i)),  # <-- ID must be string
            text=doc.get("content", ""),
            score=doc.get("score", 0),
            source=doc.get("source")
        )
        for i, doc in enumerate(documents)
    ]

    # Prepare structured JSON payload for LLM
    payload = {
        "query": query,
        "documents": [
            {"id": p.id, "text": p.text, "source": p.source, "score": p.score} for p in passages
        ]
    }

    # Call LLM
    answer = _generate_with_llm(payload)

    return answer, passages


# -------------------------
# LLM GENERATION
# -------------------------
def _generate_with_llm(payload: Dict[str, Any]) -> str:
    """
    Send structured JSON with entire documents to LLM (Groq preferred, OpenAI fallback)
    and get the concise factual answer.
    """
    query = payload["query"]
    documents = payload["documents"]

    # Prepare readable context for the model
    context = ""
    for doc in documents:
        context += f"Document ID: {doc['id']} | Source: {doc.get('source', 'Unknown')} | Score: {doc.get('score', 0)}\n"
        context += f"{doc['text']}\n\n"

    # Strict prompt instructing the LLM to only use the provided context
    prompt = f"""
You are a precise AI assistant. Use ONLY the provided documents to answer the question.
Do NOT add any information that is not present in the documents.
If the answer cannot be found, reply exactly: "I couldn't find information to answer that question."

CONTEXT (ALL DOCUMENTS):
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Give a one-line concise answer first (if available)
- Followed by a short explanation (2-3 sentences)
- Cite sources in the format: Sources: <source1>, <source2>
Answer:
"""

    try:
        # Prefer Groq LLM if available
        if hasattr(embedding_provider, "groq_client") and embedding_provider.groq_client:
            response = embedding_provider.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a precise and factual assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            return answer

        # Fallback to OpenAI
        if embedding_provider.openai_client:
            response = embedding_provider.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise and factual assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            return answer

        # No LLM available
        return "No LLM client is available to generate the answer."

    except Exception as e:
        print(f"LLM generation error: {e}")
        return "An error occurred while generating the answer from the LLM."

# -------------------------
# DOCUMENT STATS (OPTIONAL)
# -------------------------
def get_document_stats(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_documents": len(documents),
        "embedding_provider": (
            "OpenAI" if hasattr(embedding_provider, "openai_client") and embedding_provider.openai_client else
            "Groq" if hasattr(embedding_provider, "groq_client") and embedding_provider.groq_client else
            "None"
        )
    }
