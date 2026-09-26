import json
import os
import sys
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from schemas.abstention_schema import AbstentionResponse, AbstentionRequest, parse_abstention_output

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


@app.post("/graphrag-query", response_model=QueryResponse)
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
        timeout=300,
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

@app.post(
    "/graphrag/evaluate",
    response_model=AbstentionResponse,
)
def evaluate_abstention(
    request: AbstentionRequest,
) -> AbstentionResponse:
    command = [
        sys.executable,
        "-m",
        "graphrag",
        "query",
        "--root",
        str(PROJECT_ROOT),
        "--method",
        "local",
        request.query,
    ]

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "message": "GraphRAG query timed out",
                "timeout_seconds": 300,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            },
        ) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "GraphRAG abstention query failed",
                "returncode": result.returncode,
                "stderr": result.stderr,
            },
        )

    try:
        parsed = parse_abstention_output(result.stdout)

        return AbstentionResponse(
            query=request.query,
            abstention_status=parsed["abstention_status"],
            is_spam=parsed.get("is_spam"),
            answer=parsed.get("answer"),
            abstention_reason=parsed.get("abstention_reason"),
            retrieval_metadata={
                "method": "local",
                "stdout_chars": len(result.stdout),
            },
        )
    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
        ValidationError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "GraphRAG returned invalid abstention JSON",
                "error": str(exc),
                "stdout_tail": result.stdout[-4000:],
            },
        ) from exc