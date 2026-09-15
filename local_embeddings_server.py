from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
device = "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

class EmbeddingsRequest(BaseModel):
    model: str
    input: list[str]

@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    vecs = model.encode(req.input).tolist()
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": v,
            }
            for i, v in enumerate(vecs)
        ],
        "model": req.model or "all-MiniLM-L6-v2",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }