from pathlib import Path
from pypdf import PdfReader

def load_documents(path:str)-> str:
    "Load a document of any type and return its text content"
    suffix = Path(path).suffix.lower()
    
    if suffix == ".pdf":
        reader =  PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in [".txt",".md"]:
        return Path(path).read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    

    