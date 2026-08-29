import numpy as np
from src.ragapp.embedder import embed_texts,embed_query

def cosine_similarity(query_vec:np.ndarray,chunk_vecs:np.ndarray)-> np.ndarray:
    "Calculate the cosine similarity between a query vector and many chunk vectors."
    query_norm = query_vec/np.linalg.norm(query_vec)
    chunk_norm = chunk_vecs/np.linalg.norm(chunk_vecs,axis=1,keepdims=True)
    return (query_norm @ chunk_norm.T)

def retrieve(query:str,chunks:list[str],chunk_vecs: np.ndarray,top_k:int=3)->list[str]:
    """ Return the top_k chunks most semantically similar to the query based on cosine similarity.

    Args:
        query (str): query inputed by the user
        chunks (list[str]): List of text chnks to compare against the query
    """
    q_vec = np.array(embed_query(query))
    sims = cosine_similarity(q_vec,chunk_vecs)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]
