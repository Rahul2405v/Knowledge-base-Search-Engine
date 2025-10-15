import traceback
import app.embeddings as emb
import app.rag as rag

print('Embedding provider status:')
print(' sentence_model:', bool(getattr(emb.embedding_provider,'sentence_model',None)))
print(' openai_client:', bool(getattr(emb.embedding_provider,'openai_client',None)))
print(' groq_client:', bool(getattr(emb.embedding_provider,'groq_client',None)))

docs = [{'content':'This is a small test document. Rahul studied at Vellore Institute of Technology (VIT-AP).','source':'test.pdf','score':1.0}]

if getattr(emb.embedding_provider,'groq_client',None):
    try:
        print('\nTesting Groq generation...')
        ans = rag._generate_with_groq('Where is Rahul studying?', docs)
        print('Groq answer:', ans)
    except Exception as e:
        print('Groq exception:')
        traceback.print_exc()


if getattr(emb.embedding_provider,'openai_client',None):
    try:
        print('\nTesting OpenAI generation...')
        ans = rag._generate_with_openai('Where is Rahul studying?', docs)
        print('OpenAI answer:', ans)
    except Exception as e:
        print('OpenAI exception:')
        traceback.print_exc()

print('\nDone')
