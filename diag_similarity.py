import numpy as np
from app.embeddings import embedding_provider, get_embeddings
from app.vectorstore import vector_store
import sys

print('Embedding provider status:')
print(' sentence_model:', bool(getattr(embedding_provider, 'sentence_model', None)))
print(' openai_client:', bool(getattr(embedding_provider, 'openai_client', None)))
print(' groq_client:', bool(getattr(embedding_provider, 'groq_client', None)))

if not vector_store.documents:
    print('No documents in vector_store.documents')
    sys.exit(0)

# pick first doc and a sample query
doc = vector_store.documents[0]
print('\nFirst document source:', doc.get('source'))
print('First document snippet:', doc.get('content','')[:200])

query = 'Where is Rahul studying?'

# compute embeddings
try:
    emb_doc = get_embeddings([doc.get('content','')])[0]
    emb_query = get_embeddings([query])[0]
except Exception as e:
    print('Error computing embeddings:', e)
    raise

arr_doc = np.array(emb_doc)
arr_query = np.array(emb_query)

print('\nEmbeddings shapes and types:')
print(' doc shape:', arr_doc.shape, 'dtype:', arr_doc.dtype)
print(' query shape:', arr_query.shape, 'dtype:', arr_query.dtype)

# norms
norm_doc = np.linalg.norm(arr_doc)
norm_query = np.linalg.norm(arr_query)
print(' norms: doc=', norm_doc, ' query=', norm_query)

# dot product and cosine
dot = float(np.dot(arr_doc, arr_query))
cos = dot / (norm_doc * norm_query) if norm_doc>0 and norm_query>0 else None
print(' dot:', dot)
print(' cosine:', cos)

# If FAISS is available, check index
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

print('\nFAISS_AVAILABLE =', FAISS_AVAILABLE)
if FAISS_AVAILABLE and getattr(vector_store, 'index', None) is not None:
    idx = vector_store.index
    print('index ntotal:', getattr(idx, 'ntotal', None))
    # prepare query_vector
    qv = np.array([arr_query]).astype('float32')
    try:
        faiss.normalize_L2(qv)
    except Exception as e:
        print('faiss.normalize_L2 error:', e)
    try:
        scores, indices = idx.search(qv, min(5, len(vector_store.documents)))
        print('faiss search scores:', scores)
        print('faiss search indices:', indices)
    except Exception as e:
        print('faiss index search error:', e)

# Also compute simple search scores against stored docs by recomputing embeddings for each stored doc
print('\nSimple similarity (recomputing embeddings for each stored doc):')
results = []
for i, d in enumerate(vector_store.documents[:5]):
    try:
        emb = get_embeddings([d.get('content','')])[0]
    except Exception as e:
        print('error embedding doc', i, e); emb = [0.0]*len(arr_query)
    arr = np.array(emb)
    n = np.linalg.norm(arr)
    s = float(np.dot(arr_query, arr) / (norm_query * n)) if n>0 else 0.0
    results.append((i, s, d.get('source')))

for r in results:
    print(' doc idx', r[0], 'score', r[1], 'source', r[2])

print('\nDone')
