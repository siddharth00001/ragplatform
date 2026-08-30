from openai import OpenAI
from src.ragapp.config import settings

client = OpenAI(api_key=settings.openai_api_key)

EMBED_MODEL = "text-embedding-3-small"

def embed_texts(texts:list[str],batch_size:int=100)-> list[list[float]]:
    """
    Embed a list of text chunks into list of vectors.
    """
    all_vectors=[]
    for i in range(0,len(texts),batch_size):
        batch = texts[i : i+batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        all_vectors.extend(item.embedding for item in response.data)
    
    return all_vectors

def embed_query(query:str)-> list[float]:
    """_summary_
    Converts a query string into a vector embedding.
    Args:
        query (str): Query asked by the user.

    Returns:
        list[float]: The vector embedding of the query.
    """
    return embed_texts([query])[0]