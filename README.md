RAG-Based Question Answering System
Objective
Build an API-based system that allows users to upload documents and ask questions using a Retrieval-Augmented Generation (RAG) approach.

Tech Stack
FastAPI
Sentence-Transformers
FAISS
Python
System Workflow# RAG-Based Question Answering System
Project Overview
This project implements a Retrieval-Augmented Generation (RAG) based Question Answering system using FastAPI, Sentence Transformers, and FAISS.

The system allows users to upload documents, process them into embeddings, store them in a vector database, and ask questions that are answered using relevant retrieved document chunks.

The API provides interactive documentation using Swagger UI.

Architecture Overview
High-level workflow:

User uploads a document using the API
Document ingestion runs as a background task
Text is chunked into overlapping segments
Each chunk is converted into vector embeddings
Embeddings are stored in a FAISS vector index
User submits a question
Relevant chunks are retrieved using similarity search
An answer is generated from the retrieved context
Chunking Strategy
Chunk Size: 500 words
Chunk Overlap: 100 words

Reason for choosing this chunk size:

Large enough to preserve semantic meaning
Small enough to avoid noisy embeddings
Overlap helps reduce information loss at chunk boundaries
This strategy balances retrieval accuracy and performance.

Retrieval Mechanism
Embedding model: all-MiniLM-L6-v2

Vector database: FAISS (IndexFlatL2)

Similarity score calculation:

similarity = 1 / (1 + distance)

Only chunks above a similarity threshold of 0.3 are considered relevant

Observed Retrieval Failure Case
Failure Case: When a user asks a vague or generic question unrelated to the uploaded document.

Example: Document: Technical article
Question: "Tell me everything"

Result: No chunks meet the similarity threshold, and the system returns: "Not available in the document"

This behavior prevents hallucinated or irrelevant answers.

Metric Tracked
Latency

Latency is measured for every /ask request. It includes:

Query embedding time
Vector similarity search time
Response generation time
Latency is returned in the API response as: latency_seconds

Technology Stack
Backend: FastAPI
Embeddings: Sentence Transformers
Vector Store: FAISS
Validation: Pydantic
Rate Limiting: SlowAPI
Server: Uvicorn

API Endpoints
Root Endpoint:

Automatically redirects to /docs
Upload Document

Endpoint: POST /upload

Description: Uploads a document and processes it in the background.

Request Type: multipart/form-data

Response: { "message": "Document ingestion started" }

Ask Question

Endpoint: POST /ask

Request Body: { "question": "string", "top_k": 3 }

Response: { "answer": "string", "latency_seconds": 0.42 }

API Documentation
Interactive Swagger UI is available at: http://127.0.0.1:8000/docs

The root URL (/) redirects automatically to this page.

Setup Instructions
Step 1: Clone the Repository

git clone cd rag-qa-system

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Run the Application

uvicorn app:app --reload

Step 4: Open Swagger UI

http://127.0.0.1:8000/docs

requirements.txt Example
fastapi uvicorn sentence-transformers faiss-cpu slowapi pydantic numpy

Evaluation Checklist Alignment
Document upload implemented Chunking strategy implemented Vector store implemented Similarity search implemented Background ingestion implemented Rate limiting implemented Request validation implemented Latency metric tracked Clear system explanation provided

Conclusion
This project demonstrates a clean and practical implementation of a Retrieval-Augmented Generation based Question Answering system. The design focuses on simplicity, correctness, and explainability, following real-world API development practices.

User uploads a document
Document is processed in the background
Text is chunked into smaller parts
Chunks are converted into embeddings
Embeddings are stored in FAISS
User asks a question
Relevant chunks are retrieved
Answer is generated using retrieved context
Chunking Strategy
Chunk size of 500 words with 100-word overlap was chosen to balance semantic context and retrieval precision.

Retrieval Failure Case
When the user asks a vague question not present in the document, the system may retrieve weakly related chunks. This is handled using a similarity threshold.

Metric Tracked
End-to-end latency is tracked to measure system performance.

Run Instructions

pip install -r requirements.txt
uvicorn app:app --reload
