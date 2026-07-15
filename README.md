# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) system for question answering over your own documents. Upload a PDF or text file, ask questions about it, and get answers grounded in the document's actual content.

Built with **Python, FastAPI, FAISS, and sentence-transformers**.

---

## Why

A language model on its own has two problems: it doesn't know anything about your private documents, and it will confidently invent answers when it doesn't know something. RAG addresses both by retrieving relevant passages from a real source first, then constraining the model to answer from that retrieved context.

---

## How it works

```
Document → Chunk → Embed → FAISS index
                                 ↓
Question → Embed → Similarity search → Top-k chunks → Prompt → LLM → Answer
```

1. **Load** a PDF or text file.
2. **Chunk** it into ~500-word passages with 50 words of overlap.
3. **Embed** each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2`.
4. **Index** the vectors in FAISS for fast similarity search.
5. **Retrieve** the top-k most relevant chunks for a given question.
6. **Generate** an answer from those chunks as grounded context.

---

## Project structure

```
rag-document-qa/
├── app/
│   ├── rag.py           # chunking, embedding, FAISS index, retrieval, prompt
│   ├── llm.py           # swappable wrapper around the language model
│   └── main.py          # FastAPI server (/upload and /ask)
├── demo.py              # CLI demo, runs without an API key
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# CLI demo (downloads the embedding model on first run)
python demo.py

# Or run the API
uvicorn app.main:app --reload   # http://127.0.0.1:8000/docs
```

To generate real answers, install `openai`, set `OPENAI_API_KEY`, and pass `default_llm` into `answer_question` (see the comment in `app/main.py`).

---

## API

**`POST /upload`** — upload a document and build the index.

```bash
curl -F "file=@mydoc.pdf" http://127.0.0.1:8000/upload
# {"message": "Indexed 'mydoc.pdf' into 42 chunks."}
```

**`POST /ask`** — ask a question against the indexed document.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "k": 4}'
```

---

## Design decisions

**Chunking with overlap.** Retrieval is more precise on smaller passages, but a hard split risks cutting a relevant sentence in half at a chunk boundary. A 50-word overlap keeps boundary-spanning context intact.

**`IndexFlatIP` for the vector index.** Vectors are normalized at encode time, so inner product is equivalent to cosine similarity. `Flat` means exact search across every vector — the right default at this scale, where recall matters more than the milliseconds an approximate index would save. At millions of chunks, HNSW or IVF would be the trade-off to make.

**Top-k as the tuning knob.** `k` controls how many chunks reach the model. Higher `k` means more context but more tokens and more opportunity for irrelevant passages to dilute the answer. It's exposed per-request rather than hardcoded.

**The LLM behind a function boundary.** Retrieval and generation are separate concerns, so the model call lives in its own module behind a plain `(prompt) -> str` interface. Swapping providers touches one file; passing `llm=None` returns the assembled prompt, which makes the retrieval half testable on its own.

**Grounded prompting.** The prompt instructs the model to answer only from the supplied context and to say so when the answer isn't there. Retrieval narrows what the model sees; the prompt constrains what it does with it.

---

## Roadmap

- Persist embeddings in a vector database (Chroma, Pinecone, Weaviate) instead of rebuilding the index per run
- Return citations mapping each answer back to its source chunks
- Add retrieval and answer-quality evaluation
- Support multiple documents with per-user indexes
- Switch to an approximate index (HNSW) for larger corpora
