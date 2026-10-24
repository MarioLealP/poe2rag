"""poe2rag backend — hybrid retrieval + an LLM answer streamed over SSE
(/api/ask), and retrieval alone (/api/search, for Felt).

Env:
  POE2_DB           path to poe2.db            (default ./data/poe2.db)
  EMBED_MODEL       model2vec model name       (default minishlab/potion-retrieval-32M)
  LLM_BASE_URL      OpenAI-compatible endpoint (default http://felt-llm:8080/v1)
  LLM_API_KEY       key for that endpoint      (default none)
  LLM_MODEL         model id                   (default felt)
  LLM_TIMEOUT       seconds to wait for the LLM (default 600: felt-llm has one slot,
                    so a question can queue behind Felt's long prompts)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .retrieve import Hit, connect, search

DB = os.getenv("POE2_DB", "data/poe2.db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "minishlab/potion-retrieval-32M")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://felt-llm:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "felt")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))
TOP_K = int(os.getenv("TOP_K", "6"))

SYSTEM = (
    "You are a Path of Exile 2 assistant for players mid-game. Answer using ONLY "
    "the numbered passages below.\n"
    "Style:\n"
    "- Lead with the direct answer in one sentence. No preamble, no restating the "
    "question, no 'based on the passages'.\n"
    "- Keep it short — a few sentences or a tight bullet list. Bullets for "
    "numbers, sources, thresholds or steps.\n"
    "- Be practical: what the player actually does with this, gotchas, common "
    "mistakes — when the passages support it.\n"
    "- Give exact numbers when a passage has them. Never invent a formula or "
    "value that isn't there.\n"
    "- Cite each claim with its passage number, e.g. [2]. Don't add a trailing "
    "'see also [3][5]' line.\n"
    "- If the passages don't cover it, say so in one line. Path of Exile 1 "
    "mechanics do not apply."
)

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    from model2vec import StaticModel

    state["db"] = connect(DB)
    state["embed"] = StaticModel.from_pretrained(EMBED_MODEL)
    state["http"] = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=10))
    yield
    await state["http"].aclose()
    state["db"].close()


app = FastAPI(title="poe2rag", lifespan=lifespan)


class Ask(BaseModel):
    question: str


def embed_query(q: str) -> list[float]:
    return state["embed"].encode([q])[0].astype("float32").tolist()


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = f"{h.title}" + (f" — {h.section}" if h.section else "")
        blocks.append(f"[{i}] ({h.source}: {loc})\n{h.text}")
    return "\n\n".join(blocks) + f"\n\n---\nQuestion: {question}"


@app.get("/api/health")
async def health():
    n = state["db"].execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"ok": True, "chunks": n, "model": LLM_MODEL}


@app.post("/api/ask")
async def ask(body: Ask):
    hits = search(state["db"], embed_query(body.question), body.question, k=TOP_K)
    sources = [
        {"n": i, "source": h.source, "title": h.title, "section": h.section, "url": h.url}
        for i, h in enumerate(hits, 1)
    ]

    async def stream():
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        if not hits:
            yield f"event: token\ndata: {json.dumps('No matching wiki content found.')}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        payload = {
            "model": LLM_MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(body.question, hits)},
            ],
        }
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        async with state["http"].stream(
            "POST", f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
        ) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield f"event: token\ndata: {json.dumps(delta)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class Search(BaseModel):
    query: str
    k: int = TOP_K


@app.post("/api/search")
async def search_endpoint(body: Search):
    """Retrieval only - no LLM call. For Felt (felt-core poe2_wiki skill,
    WP-29): she is fully local, so she takes the passages and phrases the
    answer on her own model instead of /api/ask's cloud Gemini."""
    hits = search(state["db"], embed_query(body.query), body.query, k=max(1, min(body.k, 12)))
    return {
        "hits": [
            {"source": h.source, "title": h.title, "section": h.section, "url": h.url, "text": h.text}
            for h in hits
        ]
    }


_static = Path(__file__).resolve().parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
