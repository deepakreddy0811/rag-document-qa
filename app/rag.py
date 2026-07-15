"""
rag.py — Core Retrieval-Augmented Generation logic.

The pipeline, end to end:
  1. Load a document (PDF or text) and split it into overlapping chunks.
  2. Turn each chunk into a vector (embedding) using a sentence-transformer.
  3. Store those vectors in a FAISS index for fast similarity search.
  4. At query time: embed the question, find the most similar chunks,
     stuff them into a prompt, and ask the LLM to answer using only that context.

This file is intentionally small and readable so you can explain every line.
"""

from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ----------------------------------------------------------------------
# 1. Document loading
# ----------------------------------------------------------------------
def load_text(file_path: str) -> str:
    """Read a .pdf or .txt file and return its full text as one string."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 2. Chunking
# ----------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of ~chunk_size words with `overlap` words of
    shared context between neighbours.

    Why overlap? A sentence answering the question might sit on the boundary
    between two chunks. Overlap means it isn't cut in half and lost.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but leave an overlap
    return chunks


# ----------------------------------------------------------------------
# 3 + 4. The vector store (embeddings + FAISS index + search)
# ----------------------------------------------------------------------
@dataclass
class Retriever:
    """Holds the embedding model, the FAISS index, and the original chunks."""

    model: SentenceTransformer
    index: faiss.Index
    chunks: list[str]

    @classmethod
    def from_text(cls, text: str, model_name: str = "all-MiniLM-L6-v2") -> "Retriever":
        """Build a retriever from raw text: chunk -> embed -> index."""
        model = SentenceTransformer(model_name)
        chunks = chunk_text(text)

        # Embed every chunk. normalize_embeddings=True lets us use inner product
        # as cosine similarity (a clean, standard similarity measure).
        embeddings = model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        # IndexFlatIP = exact search using inner product. "Flat" means it
        # compares against every vector — perfect for small/medium corpora.
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return cls(model=model, index=index, chunks=chunks)

    def search(self, question: str, k: int = 4) -> list[str]:
        """Return the k chunks most similar to the question."""
        q_emb = self.model.encode(
            [question], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, k)
        return [self.chunks[i] for i in indices[0] if i != -1]


# ----------------------------------------------------------------------
# 5. Generation — assemble the prompt and call the LLM
# ----------------------------------------------------------------------
def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Combine retrieved context and the question into one grounded prompt."""
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def answer_question(retriever: Retriever, question: str, llm=None, k: int = 4) -> str:
    """
    Full RAG step: retrieve context, build the prompt, generate an answer.

    `llm` is any function that takes a prompt string and returns a string.
    Keeping it pluggable means you can swap in OpenAI, a local model, or a
    fake one for testing — without touching the retrieval code.
    """
    context_chunks = retriever.search(question, k=k)
    prompt = build_prompt(question, context_chunks)

    if llm is None:
        # No LLM configured: return the prompt so the pipeline is still
        # demonstrable offline. Plug a real model in via app/llm.py.
        return (
            "[No LLM configured. Below is the grounded prompt that would be "
            f"sent to the model.]\n\n{prompt}"
        )
    return llm(prompt)
