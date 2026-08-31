
# LDD Notes
add(texts: list[str], metadata: list[dict] = None) -> None
    Stores each text chunk with its vector (embedded internally) and optional metadata.

search(query: str, top_k: int = 3) -> list[str]   # (plus scores/metadata)
    Embeds the query, finds the top_k most similar stored chunks, returns their text.