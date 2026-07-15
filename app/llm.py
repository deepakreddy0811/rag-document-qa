"""
llm.py — A thin wrapper around the language model.

It's kept separate from rag.py on purpose: retrieval and generation are
different concerns. This is the only file you touch to change model providers,
which is a clean design decision you can point to in an interview.
"""

import os


def openai_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI's chat API. Requires:
        pip install openai
        export OPENAI_API_KEY=sk-...
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature -> factual, less creative answers
    )
    return response.choices[0].message.content


# Swap this line to change providers. For example, point it at a local
# Ollama model, Anthropic, or a HuggingFace pipeline. The rest of the app
# doesn't care which one you use.
default_llm = openai_llm
