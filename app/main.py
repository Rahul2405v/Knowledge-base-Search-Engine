# FastAPI app (entrypoint)
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import QueryRequest, QueryResponse, IngestResponse, Passage
from app.rag import get_document_stats, query_knowledge_base_with_llm, query_knowledge_base_with_passages
from app.ingest import ingest_pdf_bytes, ingest_text_bytes
from app.vectorstore import vector_store  # to fetch all documents

# Load environment variables
load_dotenv()

app = FastAPI(title="Knowledge Base Search Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Knowledge Base Search Backend"}

@app.get("/favicon.ico")
async def favicon():
    """Return a simple response for favicon requests to avoid 404s"""
    return {"message": "No favicon available"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base with a question using LLM or fallback retrieval. Returns passages with match scores."""
    try:
        top_k = getattr(request, "top_k", 5)
        use_llm = getattr(request, "use_llm", None)

        # Use the retrieval function that computes passages and scores
        answer, passages = query_knowledge_base_with_passages(request.q, max_docs=top_k, use_llm=use_llm)

        return QueryResponse(answer=answer, passages=passages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF or text file"""
    try:
        contents = await file.read()

        if file.filename.endswith(".pdf"):
            success = ingest_pdf_bytes(contents, file.filename)
        elif file.filename.endswith(".txt"):
            success = ingest_text_bytes(contents, file.filename)
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

        if success:
            return IngestResponse(inserted=1, message=f"Successfully ingested {file.filename}")
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest file")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/stats")
async def stats():
    """Return simple stats about the vector store for debugging ingestion issues"""
    try:
        # Use vector_store.documents for stats
        stats = get_document_stats(vector_store.documents)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {e}")
