from openai import OpenAI
from src.ragapp.config import settings

client = OpenAI(api_key=settings.openai_api_key)

EMBED_MODEL = "text-embedding-3-small"

def embed_texts(texts:list[str])-> list[list[float]]:
    """
    Embed a list of text chunks into list of vectors.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    
    return [item.embedding for item in response.data]

def embed_query(query:str)-> list[float]:
    """_summary_
    Converts a query string into a vector embedding.
    Args:
        query (str): Query asked by the user.

    Returns:
        list[float]: The vector embedding of the query.
    """
    return embed_texts([query])[0]