print(" LOADING CORRECT app.py FILE")

from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from slowapi import Limiter
from slowapi.util import get_remote_address
import faiss
import numpy as np
import time
from fastapi.responses import RedirectResponse
app = FastAPI(title="RAG-Based Question Answering API")
limiter = Limiter(key_func=get_remote_address)


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SIMILARITY_THRESHOLD = 0.3


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


def chunk_text(text: str):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
    return chunks

def process_document(file: UploadFile):
    text = file.file.read().decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    index.add(np.array(embeddings))
    documents.extend(chunks)

def retrieve_chunks(query: str, top_k: int):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)
        if similarity >= SIMILARITY_THRESHOLD:
            results.append(documents[idx])
    return results


@app.post("/upload")
def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_document, file)
    return {"message": "Document ingestion started"}

@app.post("/ask")
@limiter.limit("5/minute")
def ask_question(request: QuestionRequest):
    start_time = time.time()
    chunks = retrieve_chunks(request.question, request.top_k)

    if not chunks:
        return {"answer": "Not available in the document"}

    context = "\n".join(chunks)
    latency = time.time() - start_time

    return {
        "answer": context[:300] + "...",
        "latency_seconds": round(latency, 2)
    }