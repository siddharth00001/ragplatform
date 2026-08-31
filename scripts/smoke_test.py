from src.ragapp.loader import load_documents
from src.ragapp.chunker import chunk_text
from src.ragapp.vector_store import VectorStore

store = VectorStore()

if store.count() == 0:
    print(f"Initial {store.count()} chunks.")
    text_data = load_documents(path="./data/ABSL Factsheet_July 2026.pdf")
    chunked_text= chunk_text(text=text_data)
    store.add(chunked_text)
    print(f"Ingested {store.count()} chunks.")

else:
    print(f"Data is already present Inside the Vector Store, Chunk Counts {store.count()}")


results = store.search(query="What is the Expense ratio?", top_k=3)

for i,r in enumerate(results):
    print(f"Index-----\n{i} and -----\n Trimmed text-----\n{r[:120]} ")
