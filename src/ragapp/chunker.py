def chunk_text(text:str,chunk_size:int=800,overlap:int=200)-> list[str]:
    """
    Chunk the text corpus into smaller pieces of specifed size with specified overlap.
    
    Args:
        text (str): The text corpus to be chunked.
        chunk_size (int, optional): The size of each chunk. Defaults to 800.
        overlap (int, optional): The number of overlapping characters between chunks. Defaults to 200."""
    if overlap >= chunk_size:
        raise ValueError("Overlap cannot be bigger than chunk size")
    chunks =[]
    start = 0
    text_length = len(text)
    while start < text_length:
        end  = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks