from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.ragapp.loader import load_documents
from src.ragapp.chunker import chunk_text
from src.ragapp.embedder import embed_texts
from src.ragapp.retriever import retrieve
from src.ragapp.generator import generate_answer,generate_answer_stream


# A place to hold the state buult once at startup
state={}
@asynccontextmanager
async def lifespan(app:FastAPI):
    # --- Run once at a startup ----
    print("Loading and Embedding the Corpus")
    text = load_documents("./data/ABSL Factsheet_July 2026.pdf")
    chunked_text = chunk_text(text=text)
    state["chunks"] =chunked_text
    state["chunks_vecs"] = np.array(embed_texts(chunked_text))
    print(f"Ready : {len(chunked_text)} Chunks Embedded.")
    
    yield
    
    state.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    
    question:str

@app.post("/query")
def query(req:QueryRequest):
    chunks_ret = retrieve(req.question,state['chunks'],state["chunks_vecs"],top_k=3)
    answer = generate_answer(req.question,chunks_ret)
    return {"question":req.question,"answer":answer}

@app.post("/query/stream")
def query(req:QueryRequest):
    chunks_ret = retrieve(req.question,state['chunks'],state["chunks_vecs"],top_k=3)
    return StreamingResponse(
        generate_answer_stream(req.question,chunks_ret),
        media_type="text/plain",
    )