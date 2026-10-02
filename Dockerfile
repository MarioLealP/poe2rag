# poe2rag backend — FastAPI + model2vec + sqlite-vec, serving the built UI.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HF_HOME=/models \
    POE2_DB=/data/poe2.db

WORKDIR /app

# deps first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group server

# bake the embedding model into the image so the container is offline-capable
RUN uv run python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-retrieval-32M')"

COPY server/ ./server/

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "--group", "server", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
