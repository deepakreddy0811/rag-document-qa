"""
main.py — FastAPI service that exposes the RAG system over HTTP.

Endpoints:
  POST /upload   -> upload a document, build the index
  POST /ask      -> ask a question against the uploaded document

Run it with:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for an interactive UI.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.rag import Retriever, load_text, answer_question

app = FastAPI(title="RAG Document Q&A")

# Simple in-memory state. For one user / one document this is fine and easy
# to explain. (In production you'd store indexes per-user in a database or
# a managed vector DB — a good thing to mention as a "next step".)
STATE: dict[str, Retriever] = {}


class Question(BaseModel):
    question: str
    k: int = 4


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Save the uploaded file, extract text, and build the FAISS index."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    text = load_text(tmp_path)
    STATE["retriever"] = Retriever.from_text(text)
    Path(tmp_path).unlink(missing_ok=True)  # clean up the temp file

    n_chunks = len(STATE["retriever"].chunks)
    return {"message": f"Indexed '{file.filename}' into {n_chunks} chunks."}


@app.post("/ask")
async def ask(payload: Question):
    """Answer a question against the currently indexed document."""
    retriever = STATE.get("retriever")
    if retriever is None:
        return {"error": "Upload a document first via /upload."}

    # To use a real LLM, import default_llm from app.llm and pass it here:
    #   from app.llm import default_llm
    #   answer = answer_question(retriever, payload.question, llm=default_llm, k=payload.k)
    answer = answer_question(retriever, payload.question, llm=None, k=payload.k)
    return {"question": payload.question, "answer": answer}


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}
