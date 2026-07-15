# RAG Document Q&A

A small Retrieval-Augmented Generation (RAG) system. Upload a document, ask
questions about it, and get answers grounded in the document's content.

Built with **Python, FastAPI, FAISS, and sentence-transformers**.

---

## What it does

1. **Load** a PDF or text file.
2. **Chunk** it into overlapping passages.
3. **Embed** each chunk into a vector using a sentence-transformer model.
4. **Index** the vectors in FAISS for fast similarity search.
5. **Retrieve** the most relevant chunks for a question.
6. **Generate** an answer by giving those chunks to an LLM as grounded context.

---

## Project structure

```
rag-document-qa/
├── app/
│   ├── rag.py      # core: chunking, embedding, FAISS index, retrieval, prompt
│   ├── llm.py      # thin, swappable wrapper around the language model
│   └── main.py     # FastAPI server (/upload and /ask endpoints)
├── demo.py         # run the pipeline from the CLI, no API key needed
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the offline demo (downloads the embedding model on first run)
python demo.py

# 3. Or run the web API
uvicorn app.main:app --reload
# then open http://127.0.0.1:8000/docs
```

To use a real LLM for answers, install `openai`, set `OPENAI_API_KEY`, and
follow the one-line comment in `app/main.py` to pass `default_llm` into
`answer_question`.

---

## How it works (the parts you should be able to explain)

**Why RAG instead of just asking the LLM?**
LLMs don't know about your private documents and can hallucinate. RAG grounds
the answer in real retrieved text, so responses are accurate and traceable to
a source.

**Why chunk with overlap?**
Embedding models have a limited context, and retrieval is more precise on
smaller passages. Overlap (here, 50 words) keeps sentences that straddle a
chunk boundary from being split and lost.

**What is an embedding?**
A vector of numbers representing the *meaning* of text. Texts with similar
meaning land close together in vector space, which is what makes semantic
search possible. Here we use `all-MiniLM-L6-v2` — small, fast, 384 dimensions.

**Why FAISS, and why `IndexFlatIP`?**
FAISS is a library for fast vector similarity search. `IndexFlatIP` does an
exact search using inner product; combined with normalized vectors that's
equivalent to cosine similarity. "Flat" = compare against everything, which is
ideal for small-to-medium corpora. For millions of vectors you'd switch to an
approximate index like `IVF` or `HNSW` — a good trade-off to mention.

**What does `k` control?**
How many chunks we retrieve and feed to the LLM. Higher `k` gives more context
but costs more tokens and can add noise. It's the main accuracy/latency knob.

**Why is the LLM call in its own file?**
Separation of concerns. Retrieval and generation are independent, so swapping
OpenAI for a local model touches exactly one file and nothing else breaks.

---

## Things I'd improve next (good interview talking points)

- Store embeddings in a persistent vector DB (Chroma, Pinecone, Weaviate)
  instead of rebuilding the index in memory each run.
- Add **citations**: return which chunks an answer came from.
- Add an **evaluation** step to measure retrieval quality and answer accuracy.
- Support multiple documents and per-user indexes.
- Swap the flat index for an approximate one (HNSW) at larger scale.
