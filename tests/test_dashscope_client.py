"""El adaptador valida la llamada y la respuesta sin acceder a la red."""

import email.message
import http.client
import io
import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Any

import pytest

import dashscope_client

KEY = "secret-for-test"


class FakePost:
    """Reemplaza dashscope_client.post: anota cada llamada y responde lo que se le pida."""

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, Any], str]] = []
        self.retries: list[int] = []
        self.bases: list[str] = []

    def __call__(
        self,
        path: str,
        body: dict[str, Any],
        api_key: str,
        *,
        retries: int = 0,
        base: str = dashscope_client.BASE_URL,
    ) -> dict[str, Any]:
        self.calls.append((path, body, api_key))
        self.retries.append(retries)
        self.bases.append(base)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def embeddings(vectors: list[list[float]]) -> dict[str, Any]:
    # La API puede devolver los vectores en otro orden; text_index dice a cuál corresponde.
    items = [{"text_index": i, "embedding": v} for i, v in reversed(list(enumerate(vectors)))]
    return {"output": {"embeddings": items}, "usage": {"total_tokens": 7}}


def reranked(results: list[tuple[int, float]]) -> dict[str, Any]:
    items = [{"index": index, "relevance_score": score} for index, score in results]
    return {"output": {"results": items}}


def vector(value: float = 0.25) -> list[float]:
    return [value] * dashscope_client.DIMENSION


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakePost:
    monkeypatch.setenv("DASHSCOPE_API_KEY", KEY)
    fake = FakePost(embeddings([vector()]))
    monkeypatch.setattr(dashscope_client, "post", fake)
    return fake


def test_embed_documents_uses_server_key_and_document_mode(api: FakePost) -> None:
    assert dashscope_client.embed_texts(["título de tesis"]) == [vector()]
    ((path, body, key),) = api.calls
    assert key == KEY
    assert path == "services/embeddings/text-embedding/text-embedding"
    assert body["model"] == "qwen3.7-text-embedding"
    assert body["input"] == {"texts": ["título de tesis"]}
    assert body["parameters"] == {"text_type": "document", "dimension": 1024}


def test_embed_query_uses_query_mode_and_instruction(api: FakePost) -> None:
    assert dashscope_client.embed_query("idea de tesis") == vector()
    parameters = api.calls[0][1]["parameters"]
    assert parameters["text_type"] == "query"
    assert parameters["instruct"] == dashscope_client.QUERY_INSTRUCTION


def test_embed_texts_keeps_the_order_of_the_texts(api: FakePost) -> None:
    api.reply = embeddings([vector(0.1), vector(0.2)])
    assert dashscope_client.embed_texts(["a", "b"]) == [vector(0.1), vector(0.2)]


def test_embed_texts_skips_api_for_empty_batch(
    api: FakePost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    assert dashscope_client.embed_texts([]) == []
    assert not api.calls


@pytest.mark.parametrize("texts", [[" "], ["ok"] * (dashscope_client.MAX_BATCH_SIZE + 1)])
def test_embed_texts_rejects_invalid_batch(api: FakePost, texts: list[str]) -> None:
    with pytest.raises(ValueError):
        dashscope_client.embed_texts(texts)
    assert not api.calls


def test_embed_texts_requires_api_key(api: FakePost, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    with pytest.raises(dashscope_client.EmbeddingError, match="DASHSCOPE_API_KEY"):
        dashscope_client.embed_texts(["título"])
    assert not api.calls


def test_embed_texts_turns_api_errors_into_embedding_errors(api: FakePost) -> None:
    api.reply = dashscope_client.ApiError("DashScope respondió con HTTP 401 (InvalidApiKey)")
    with pytest.raises(dashscope_client.EmbeddingError, match="InvalidApiKey"):
        dashscope_client.embed_texts(["título"])


@pytest.mark.parametrize(
    "reply",
    [
        embeddings([[0.1, 0.2]]),
        embeddings([[float("nan")] * dashscope_client.DIMENSION]),
        {"output": {"embeddings": ["no es un objeto"]}},
        {"output": {"embeddings": [{"text_index": None, "embedding": vector()}]}},
        {"output": {"embeddings": [{"text_index": "0", "embedding": vector()}]}},
        {"output": {"embeddings": [{"text_index": True, "embedding": vector()}]}},
        {"output": {"embeddings": [{"text_index": 3, "embedding": vector()}]}},
        {"output": "sin vectores"},
        {},
    ],
)
def test_embed_texts_rejects_malformed_replies(api: FakePost, reply: dict[str, Any]) -> None:
    api.reply = reply
    with pytest.raises(dashscope_client.EmbeddingError):
        dashscope_client.embed_texts(["título"])


def test_rerank_titles_uses_server_key_and_preserves_api_order(api: FakePost) -> None:
    expected = [(1, 0.9), (0, 0.4)]
    api.reply = reranked(expected)
    assert dashscope_client.rerank_titles("idea de tesis", ["título A", "título B"]) == expected
    ((path, body, key),) = api.calls
    assert key == KEY
    assert path == "services/rerank/text-rerank/text-rerank"
    assert body["model"] == "qwen3-rerank"
    assert body["input"] == {"query": "idea de tesis", "documents": ["título A", "título B"]}
    assert body["parameters"]["top_n"] == 2
    assert body["parameters"]["instruct"] == dashscope_client.RERANK_INSTRUCTION


def test_rerank_titles_validates_document_limit_and_response(api: FakePost) -> None:
    with pytest.raises(ValueError):
        dashscope_client.rerank_titles(
            "idea", ["título"] * (dashscope_client.MAX_RERANK_DOCUMENTS + 1)
        )
    assert not api.calls

    api.reply = reranked([(0, 0.5), (0, 0.4)])
    with pytest.raises(dashscope_client.RerankError, match="duplicados"):
        dashscope_client.rerank_titles("idea", ["título A", "título B"])

    api.reply = {"output": {"results": ["no es un objeto", "tampoco"]}}
    with pytest.raises(dashscope_client.RerankError, match="ranking inválido"):
        dashscope_client.rerank_titles("idea", ["título A", "título B"])


def test_rerank_titles_requires_api_key(api: FakePost, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    with pytest.raises(dashscope_client.RerankError, match="DASHSCOPE_API_KEY"):
        dashscope_client.rerank_titles("idea", ["título"])
    assert not api.calls


# --- post(): la única función que toca la red --------------------------------


def http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        dashscope_client.BASE_URL, code, "error", email.message.Message(), io.BytesIO(body)
    )


def test_post_sends_the_key_in_a_header_not_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[urllib.request.Request, float]] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        sent.append((request, timeout))
        return io.BytesIO(b'{"output": {}}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert dashscope_client.post("services/x", {"model": "m"}, KEY) == {"output": {}}
    ((request, timeout),) = sent
    assert request.full_url == f"{dashscope_client.BASE_URL}/services/x"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {KEY}"
    assert KEY not in request.full_url
    assert json.loads(request.data) == {"model": "m"}  # type: ignore[arg-type]
    assert timeout == dashscope_client.REQUEST_TIMEOUT


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            http_error(401, b'{"code": "InvalidApiKey", "message": "' + KEY.encode() + b'"}'),
            "HTTP 401 (InvalidApiKey)",
        ),
        (http_error(502, b"<html>bad gateway</html>"), "HTTP 502 (sin código)"),
        (urllib.error.URLError("sin red"), "No se pudo conectar"),
        (TimeoutError(), "No se pudo conectar"),
        (http.client.IncompleteRead(b'{"output"'), "No se pudo conectar"),
        (ssl.SSLError("certificado"), "No se pudo conectar"),
        (http.client.BadStatusLine("HTTP/1.1 ???"), "No se pudo conectar"),
    ],
)
def test_post_reports_failures_without_the_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
    message: str,
) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        raise failure

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with (
        caplog.at_level(logging.WARNING, logger="dashscope_client"),
        pytest.raises(dashscope_client.ApiError) as error,
    ):
        dashscope_client.post("services/x", {"input": {"texts": ["idea privada"]}}, KEY)
    assert message in str(error.value)
    assert KEY not in str(error.value)
    # La falla queda en el log del servidor, sin la llave ni el contenido de la petición.
    assert message in caplog.text
    assert KEY not in caplog.text
    assert "idea privada" not in caplog.text
    # Sin la excepción original encadenada: un traceback tampoco la muestra.
    assert error.value.__cause__ is None and error.value.__suppress_context__


def test_only_index_titles_are_retried(api: FakePost) -> None:
    dashscope_client.embed_texts(["título de tesis"])
    dashscope_client.embed_query("idea de tesis")
    # Los títulos del índice esperan y reintentan; la idea de un visitante, no.
    assert api.retries == [len(dashscope_client.BACKOFF), 0]


def scripted_urlopen(
    monkeypatch: pytest.MonkeyPatch, replies: list[Exception | bytes]
) -> list[float]:
    """urlopen responde en orden; devuelve las esperas que pidió post()."""
    waits: list[float] = []
    monkeypatch.setattr(dashscope_client.time, "sleep", waits.append)

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return waits


def test_post_retries_passing_failures_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = scripted_urlopen(
        monkeypatch,
        [http_error(429, b"{}"), urllib.error.URLError("sin red"), b'{"output": {}}'],
    )
    assert dashscope_client.post("services/x", {}, KEY, retries=4) == {"output": {}}
    assert waits == [1, 2]


def test_post_does_not_retry_a_rejected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = scripted_urlopen(monkeypatch, [http_error(401, b'{"code": "InvalidApiKey"}')])
    with pytest.raises(dashscope_client.ApiError, match="InvalidApiKey"):
        dashscope_client.post("services/x", {}, KEY, retries=4)
    assert waits == []


def test_post_gives_up_after_the_last_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = scripted_urlopen(monkeypatch, [http_error(503, b"{}")] * 3)
    with pytest.raises(dashscope_client.ApiError, match="HTTP 503"):
        dashscope_client.post("services/x", {}, KEY, retries=2)
    assert waits == [1, 2]


def test_embed_texts_rejects_repeated_indices_instead_of_crossing_vectors(api: FakePost) -> None:
    api.reply = {
        "output": {
            "embeddings": [
                {"text_index": 0, "embedding": vector(0.2)},
                {"text_index": 0, "embedding": vector(0.1)},
            ]
        }
    }
    with pytest.raises(dashscope_client.EmbeddingError, match="índice válido"):
        dashscope_client.embed_texts(["a", "b"])


class BrokenBody(io.BytesIO):
    """Un cuerpo de error que se corta al leerlo."""

    def read(self, *args: Any) -> bytes:
        raise http.client.IncompleteRead(b'{"code"')


def test_post_survives_an_error_body_that_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = urllib.error.HTTPError(
        dashscope_client.BASE_URL, 503, "error", email.message.Message(), BrokenBody()
    )
    scripted_urlopen(monkeypatch, [failure])
    with pytest.raises(dashscope_client.ApiError, match=r"HTTP 503 \(sin código\)"):
        dashscope_client.post("services/x", {}, KEY)


def chat_reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def test_chat_json_asks_flash_for_json_without_thinking(api: FakePost) -> None:
    api.reply = chat_reply('{"resumen": "Trabaja visión por computadora."}')
    answer = dashscope_client.chat_json("Responde en JSON.", "Idea: robots")
    assert answer == {"resumen": "Trabaja visión por computadora."}
    ((path, body, key),) = api.calls
    assert (path, key, api.bases) == ("chat/completions", KEY, [dashscope_client.CHAT_URL])
    assert body["model"] == "qwen3.7-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["enable_thinking"] is False
    assert body["max_tokens"] == dashscope_client.CHAT_MAX_TOKENS
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "reply",
    [chat_reply("no es JSON"), chat_reply("[1, 2]"), {"choices": []}, {}],
)
def test_chat_json_rejects_anything_but_a_json_object(api: FakePost, reply: dict[str, Any]) -> None:
    api.reply = reply
    with pytest.raises(dashscope_client.ChatError):
        dashscope_client.chat_json("Responde en JSON.", "Idea")


def test_chat_json_turns_api_errors_into_chat_errors(api: FakePost) -> None:
    api.reply = dashscope_client.ApiError("DashScope respondió con HTTP 429 (Throttling)")
    with pytest.raises(dashscope_client.ChatError, match="429"):
        dashscope_client.chat_json("Responde en JSON.", "Idea")


def test_chat_json_requires_api_key(api: FakePost, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    with pytest.raises(dashscope_client.ChatError, match="DASHSCOPE_API_KEY"):
        dashscope_client.chat_json("Responde en JSON.", "Idea")
    assert not api.calls


def test_post_can_reach_the_chat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        sent.append(request.full_url)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    dashscope_client.post("chat/completions", {}, KEY, base=dashscope_client.CHAT_URL)
    assert sent == [f"{dashscope_client.HOST}/compatible-mode/v1/chat/completions"]
