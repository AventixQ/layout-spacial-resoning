"""Embedding provider abstraction."""

import hashlib
import os
import re
from collections.abc import Callable

from openai import OpenAI

from layout_spatial_reasoning.config import load_env


Embedding = list[float]
EmbeddingFunction = Callable[[list[str]], list[Embedding]]

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str]) -> list[Embedding]:
    """Return deterministic local embeddings for development and tests.

    This is intentionally simple and API-free. Production embeddings can later
    be plugged into graph methods without changing the layout algorithm.
    """
    return [local_hash_embedding(text) for text in texts]


def embedding_function_from_env() -> EmbeddingFunction:
    """Return the configured embedding function.

    DEFAULT_EMBEDDING_PROVIDER=openai gives the thesis-aligned provider.
    DEFAULT_EMBEDDING_PROVIDER=local keeps tests and development API-free.
    """
    load_env()
    provider = os.environ.get("DEFAULT_EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai":
        return openai_embed_texts
    if provider == "local":
        return embed_texts
    raise ValueError("DEFAULT_EMBEDDING_PROVIDER must be either 'local' or 'openai'.")


def openai_embed_texts(texts: list[str]) -> list[Embedding]:
    """Generate embeddings with OpenAI, using text-embedding-3-small by default."""
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")

    model = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts)
    return [list(item.embedding) for item in response.data]


def local_hash_embedding(text: str, *, dimensions: int = 64) -> Embedding:
    """Embed text as a normalized hashed bag of tokens."""
    vector = [0.0] * dimensions
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dimensions
        vector[index] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]
