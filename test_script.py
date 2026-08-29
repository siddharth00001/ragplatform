import numpy as np
from src.ragapp.loader import load_documents
from src.ragapp.chunker import chunk_text
from src.ragapp.embedder import embed_texts,embed_query
from src.ragapp.retriever import retrieve
from src.ragapp.generator import generate_answer


# DEMO TESTING OF THE CHUNKER FUNCTION

# demo = "Hello there this is a demo text to test the chunking facility"*20
# print(len(demo))
# chunks = chunk_text(demo,chunk_size=100,overlap=20)
# print(f"{len(chunks)} chunks created from the demo text")
# for i,chunk in enumerate(chunks):
#     print(f"Chunk {i}: {chunk[:15]}...{chunk[-15:]}")

## The real implementation would involve loading a document and chunking it
text = load_documents("./data/active-factsheet---july-2026.pdf")
chunks = chunk_text(text)
chunk_vecs = np.array(embed_texts(chunks))
# print(type(chunks),len(chunks))
# vectors = embed_texts(chunks[:3])
# print(len(vectors))
# print(f"each vector has {len(vectors[0])} dimensions")
# print(f"The first 5 number of the vector 0 are: {vectors[0][:5]}")

# # Embed a question, Confirm same dimensions
# q_vec = embed_query("What is the expense ratio?")
# print(f"query vector dimensions: {len(q_vec)}")



# Example usage
query = "What is the CEO's home address?"
results  = retrieve(query,chunks,chunk_vecs,top_k =3)
answer = generate_answer(query=query,context_chunks=results)
print(answer)