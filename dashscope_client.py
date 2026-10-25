"""Cliente mínimo y validado para los embeddings de DashScope."""

import math
import os
from collections.abc import Sequence
from typing import Literal

from dashscope import TextEmbedding

MODEL = "qwen3.7-text-embedding"
BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/api/v1"
DIMENSION = 1024
MAX_BATCH_SIZE = 20
REQUEST_TIMEOUT = 30
QUERY_INSTRUCTION = (
    "Given a research idea, retrieve relevant thesis, research publication, or project titles."
)
TextType = Literal["query", "document"]


class EmbeddingError(RuntimeError):
    """La API de embeddings no está disponible o devolvió una respuesta inválida."""


def embed_texts(texts: Sequence[str], *, text_type: TextType = "document") -> list[list[float]]:
    """Devuelve un vector por texto, validando el tamaño y la respuesta de DashScope."""
    if not texts:
        return []
    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"DashScope admite hasta {MAX_BATCH_SIZE} textos por lote")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Todos los textos deben ser cadenas no vacías")
    if text_type not in ("query", "document"):
        raise ValueError("text_type debe ser 'query' o 'document'")

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise EmbeddingError("Falta configurar DASHSCOPE_API_KEY en el servidor")

    try:
        response = TextEmbedding.call(
            model=MODEL,
            input=list(texts),
            api_key=api_key,
            base_address=BASE_URL,
            request_timeout=REQUEST_TIMEOUT,
            text_type=text_type,
            dimension=DIMENSION,
            instruct=QUERY_INSTRUCTION if text_type == "query" else None,
        )
    except Exception:
        raise EmbeddingError("No se pudo conectar con el servicio de embeddings") from None

    if response.status_code != 200:
        code = getattr(response, "code", None) or "sin código"
        raise EmbeddingError(f"DashScope respondió con HTTP {response.status_code} ({code})")

    output = response.output or {}
    embeddings = output.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbeddingError("DashScope devolvió una cantidad inesperada de vectores")

    embeddings.sort(key=lambda item: item.get("text_index", item.get("index", 0)))
    vectors = [item.get("embedding") for item in embeddings]
    if any(
        not isinstance(vector, list)
        or len(vector) != DIMENSION
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise EmbeddingError("DashScope devolvió un vector con dimensiones o valores inválidos")
    return vectors


def embed_query(text: str) -> list[float]:
    """Vectoriza una consulta de usuario con la instrucción de recuperación."""
    if not text.strip():
        raise ValueError("La consulta no puede estar vacía")
    return embed_texts([text], text_type="query")[0]
