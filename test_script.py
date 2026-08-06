from src.ragapp.loader import load_documents

text = load_documents("./data/active-factsheet---july-2026.pdf")
print(f"The Number of character loaded from the document is :{len(text)}")
print(text[:500])