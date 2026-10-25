"""Cliente mínimo y validado para los modelos de búsqueda de DashScope.

Habla HTTP con la biblioteca estándar, como openalex.py: el SDK oficial traía 29 paquetes
(fastapi, opentelemetry, tiktoken…) a la imagen para dos llamadas POST.
"""

import http.client
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Literal

MODEL = "qwen3.7-text-embedding"
RERANK_MODEL = "qwen3-rerank"
BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/api/v1"
DIMENSION = 1024
MAX_BATCH_SIZE = 20
REQUEST_TIMEOUT = 30
MAX_QUERY_CHARS = 500
MAX_RERANK_DOCUMENTS = 100
QUERY_INSTRUCTION = (
    "Given a research idea, retrieve relevant thesis, research publication, or project titles."
)
RERANK_INSTRUCTION = (
    "Rank research titles by how directly they address the user's research idea. "
    "Prefer titles about the same task, method, or subject; do not reward superficial word overlap."
)
TextType = Literal["query", "document"]
# Un 429 o un 5xx suelen pasar solos. Al construir el índice vale la pena esperar y
# reintentar (como openalex.py); con un visitante no: mejor la búsqueda por términos ya.
RETRY_CODES = {429, 500, 502, 503, 504}
BACKOFF = (1, 2, 4, 8)
# Las fallas quedan en el log del servidor (docker logs) con su código, nunca con la llave
# ni con la idea: así se distingue una llave vencida o sin saldo de una caída pasajera.
log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """DashScope no respondió, o respondió con un error HTTP."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class EmbeddingError(RuntimeError):
    """La API de embeddings no está disponible o devolvió una respuesta inválida."""


class RerankError(RuntimeError):
    """La API de reranking no está disponible o devolvió una respuesta inválida."""


def post(path: str, body: dict[str, Any], api_key: str, *, retries: int = 0) -> dict[str, Any]:
    """Un POST JSON a DashScope, con hasta `retries` reintentos si la falla es pasajera.

    La llave va en la cabecera, nunca en la URL ni en un error.
    """
    attempt = 0
    while True:
        try:
            return send(path, body, api_key)
        except ApiError as error:
            if not error.transient or attempt == retries:
                raise
            time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
            attempt += 1


def send(path: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Un solo intento de POST; convierte cada falla en ApiError."""
    request = urllib.request.Request(
        f"{BASE_URL}/{path}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:  # nosec B310
            reply = json.load(response)
    except urllib.error.HTTPError as error:
        # DashScope explica el error en el cuerpo; solo se conserva su código, como
        # "InvalidApiKey", que no repite nada de la petición.
        try:
            code = json.loads(error.read()).get("code") or "sin código"
        except (ValueError, AttributeError):
            code = "sin código"
        message = f"DashScope respondió con HTTP {error.code} ({code})"
        raise failure(path, message, transient=error.code in RETRY_CODES) from None
    except (OSError, http.client.HTTPException, ValueError) as error:
        # Las familias completas y no una lista: OSError abarca la red, los tiempos de espera
        # y TLS (ssl.SSLError); HTTPException, una respuesta rota o cortada. El nombre de la
        # clase basta para saber qué pasó, sin su mensaje.
        message = f"No se pudo conectar con DashScope ({type(error).__name__})"
        raise failure(path, message, transient=not isinstance(error, ValueError)) from None
    if not isinstance(reply, dict):
        raise failure(path, "DashScope devolvió una respuesta inesperada")
    return reply


def failure(path: str, message: str, *, transient: bool = False) -> ApiError:
    """Deja la falla en el log del servidor y la devuelve como ApiError."""
    log.warning("%s: %s", path.rsplit("/", 1)[-1], message)
    return ApiError(message, transient=transient)


def is_configured() -> bool:
    """Indica si la clave está disponible en el entorno del servidor."""
    return bool(os.environ.get("DASHSCOPE_API_KEY", "").strip())


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

    parameters: dict[str, Any] = {"text_type": text_type, "dimension": DIMENSION}
    if text_type == "query":
        parameters["instruct"] = QUERY_INSTRUCTION
    try:
        reply = post(
            "services/embeddings/text-embedding/text-embedding",
            {"model": MODEL, "input": {"texts": list(texts)}, "parameters": parameters},
            api_key,
            retries=len(BACKOFF) if text_type == "document" else 0,
        )
    except ApiError as error:
        raise EmbeddingError(str(error)) from None

    output = reply.get("output") or {}
    embeddings = output.get("embeddings") if isinstance(output, dict) else None
    if (
        not isinstance(embeddings, list)
        or len(embeddings) != len(texts)
        or not all(isinstance(item, dict) for item in embeddings)
    ):
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
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(f"La consulta no puede superar {MAX_QUERY_CHARS} caracteres")
    return embed_texts([text], text_type="query")[0]


def rerank_titles(query: str, titles: Sequence[str]) -> list[tuple[int, float]]:
    """Ordena títulos candidatos y devuelve sus índices originales y relevancia."""
    if not query.strip():
        raise ValueError("La consulta no puede estar vacía")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"La consulta no puede superar {MAX_QUERY_CHARS} caracteres")
    if not titles:
        return []
    if len(titles) > MAX_RERANK_DOCUMENTS:
        raise ValueError(f"El reranker admite hasta {MAX_RERANK_DOCUMENTS} títulos")
    if any(not isinstance(title, str) or not title.strip() for title in titles):
        raise ValueError("Todos los títulos deben ser cadenas no vacías")

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RerankError("Falta configurar DASHSCOPE_API_KEY en el servidor")

    try:
        reply = post(
            "services/rerank/text-rerank/text-rerank",
            {
                "model": RERANK_MODEL,
                "input": {"query": query, "documents": list(titles)},
                "parameters": {"top_n": len(titles), "instruct": RERANK_INSTRUCTION},
            },
            api_key,
        )
    except ApiError as error:
        raise RerankError(str(error)) from None

    output = reply.get("output") or {}
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list) or len(results) != len(titles):
        raise RerankError("DashScope devolvió una cantidad inesperada de resultados")

    ranked: list[tuple[int, float]] = []
    for result in results:
        index = result.get("index") if isinstance(result, dict) else None
        score = result.get("relevance_score") if isinstance(result, dict) else None
        if (
            not isinstance(index, int)
            or index < 0
            or index >= len(titles)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise RerankError("DashScope devolvió un ranking inválido")
        ranked.append((index, float(score)))
    if len({index for index, _ in ranked}) != len(titles):
        raise RerankError("DashScope devolvió índices duplicados")
    return ranked
