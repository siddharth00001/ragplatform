from src.ragapp.config import settings
from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)


response = client.chat.completions.create(
    model=settings.llm_model,
    messages=[{"role":"user","content":"Greet someone in one sentence"}]
)

print(response.choices[0].message.content)