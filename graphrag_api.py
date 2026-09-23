import os
import sys
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GraphRAG Query Service")

PROJECT_ROOT = Path(
    os.getenv("GRAPHRAG_ROOT", Path(__file__).resolve().parent)
)


class QueryRequest(BaseModel):
    message: str


class QueryResponse(BaseModel):
    answer: str
    provider: str = "graphrag"
    sources: list[dict] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    command = [
        sys.executable,
        "-m",
        "graphrag",
        "query",
        "--root",
        str(PROJECT_ROOT),
        "--method",
        "local",
        request.message,
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "GraphRAG query failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            },
        )

    return QueryResponse(answer=result.stdout.strip())