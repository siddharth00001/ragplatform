from openai import OpenAI
from src.ragapp.config import settings

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are a helpful assistant answering questions about mutual fund factsheets.
Answer ONLY using the provided context. If the context does not contain the answer,
say "I don't have enough information to answer that." Do not use outside knowledge."""


def generate_answer(query:str,context_chunks:list[str])-> str:
    
    context=  "\n\n---\n\n".join(context_chunks)
    user_message  = f"Context:\n{context}\n\nQuestion:{query}"
    
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages= [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content":user_message}
        ]
    )
    return response.choices[0].message.content
