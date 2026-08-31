import chromadb 
from src.ragapp.embedder import embed_query,embed_texts


class VectorStore():
    
    def __init__(self, collection_name:str="factsheets",persist_dir:str="./chroma_db" ) -> None:
        
        self.client  = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add(self,texts:list[str],metadata:list[dict]=None)-> None:
        vectors = embed_texts(texts)
        vec_ids = [ str(i) for i in range(len(texts))]
        self.collection.add(
            ids=vec_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadata
        )
    
    def search(self,query:str,top_k:int=3)->list[str]:
        q_vec = embed_query(query=query)
        results = self.collection.query(query_embeddings=[q_vec],n_results=top_k)
        return results["documents"][0]
    
    def count(self):
        return self.collection.count()