SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context provided below. Follow these rules strictly:
1. Answer only using information found in the context.
2. If the context does not contain the answer, say "I don't have enough
   information to answer that" instead of guessing.
3. Do not use any outside knowledge, even if you know the answer.
4. Keep answers short and to the point.
5. If you use information from the context, you don't need to cite it
   explicitly, just answer naturally.
"""


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    user_message = f"""Context:
{context}
Question: {query}
Answer the question using only the context above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages