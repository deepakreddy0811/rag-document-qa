"""
demo.py — Run the whole RAG pipeline from the command line, no server needed.

This is the fastest way to SEE the pipeline work and to understand it before
an interview. It uses a tiny built-in sample document so it runs with no
API key (it prints the grounded prompt instead of calling a paid model).

Run:
    python demo.py
"""

from app.rag import Retriever, answer_question

SAMPLE_DOC = """
The Apollo program was a series of human spaceflight missions undertaken by NASA.
Apollo 11 was the first mission to land humans on the Moon, in July 1969.
Neil Armstrong was the first person to walk on the lunar surface, followed by
Buzz Aldrin. Michael Collins remained in lunar orbit aboard the command module.
The program officially ended in 1972 after Apollo 17, the final crewed Moon landing.
"""


def main():
    print("Building retriever from sample document...\n")
    retriever = Retriever.from_text(SAMPLE_DOC)
    print(f"Document split into {len(retriever.chunks)} chunk(s).\n")

    questions = [
        "Who was the first person to walk on the Moon?",
        "When did the Apollo program end?",
        "What is the capital of France?",  # not in the doc -> tests grounding
    ]

    for q in questions:
        print("=" * 70)
        print(f"Q: {q}\n")
        retrieved = retriever.search(q, k=2)
        print(f"Retrieved {len(retrieved)} chunk(s). Top match starts with:")
        print(f'  "{retrieved[0][:80]}..."\n')
        # llm=None prints the grounded prompt so you can see exactly what
        # would be sent to the model. Wire in a real LLM via app/llm.py.
        answer = answer_question(retriever, q, llm=None, k=2)
        print(answer)
        print()


if __name__ == "__main__":
    main()
