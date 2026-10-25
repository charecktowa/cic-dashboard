"""El adaptador valida la llamada y la respuesta sin acceder a la red."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dashscope_client


def response(vectors: list[list[float]], *, status: int = 200) -> SimpleNamespace:
    embeddings = [
        {"text_index": index, "embedding": vector}
        for index, vector in reversed(list(enumerate(vectors)))
    ]
    return SimpleNamespace(
        status_code=status,
        code="InvalidApiKey" if status != 200 else None,
        output={"embeddings": embeddings},
    )


def rerank_response(results: list[tuple[int, float]], *, status: int = 200) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status,
        code="InvalidApiKey" if status != 200 else None,
        output=SimpleNamespace(
            results=[
                SimpleNamespace(index=index, relevance_score=score) for index, score in results
            ]
        ),
    )


def vector(value: float = 0.25) -> list[float]:
    return [value] * dashscope_client.DIMENSION


def test_embed_documents_uses_server_key_and_document_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-for-test")
    with patch.object(
        dashscope_client.TextEmbedding, "call", return_value=response([vector()])
    ) as call:
        assert dashscope_client.embed_texts(["título de tesis"]) == [vector()]

    kwargs = call.call_args.kwargs
    assert kwargs["api_key"] == "secret-for-test"
    assert kwargs["model"] == "qwen3.7-text-embedding"
    assert kwargs["text_type"] == "document"
    assert kwargs["base_address"] == dashscope_client.BASE_URL
    assert "instruct" not in kwargs or kwargs["instruct"] is None


def test_embed_query_uses_query_mode_and_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-for-test")
    with patch.object(
        dashscope_client.TextEmbedding, "call", return_value=response([vector()])
    ) as call:
        assert dashscope_client.embed_query("idea de tesis") == vector()

    assert call.call_args.kwargs["text_type"] == "query"
    assert call.call_args.kwargs["instruct"] == dashscope_client.QUERY_INSTRUCTION


def test_embed_texts_skips_api_for_empty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with patch.object(dashscope_client.TextEmbedding, "call") as call:
        assert dashscope_client.embed_texts([]) == []
    call.assert_not_called()


@pytest.mark.parametrize(
    "texts",
    [
        [" "],
        ["ok"] * (dashscope_client.MAX_BATCH_SIZE + 1),
    ],
)
def test_embed_texts_rejects_invalid_batch(texts: list[str]) -> None:
    with patch.object(dashscope_client.TextEmbedding, "call") as call, pytest.raises(ValueError):
        dashscope_client.embed_texts(texts)
    call.assert_not_called()


def test_embed_texts_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with (
        patch.object(dashscope_client.TextEmbedding, "call") as call,
        pytest.raises(dashscope_client.EmbeddingError, match="DASHSCOPE_API_KEY"),
    ):
        dashscope_client.embed_texts(["título"])
    call.assert_not_called()


def test_embed_texts_rejects_api_errors_without_exposing_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-for-test")
    with (
        patch.object(
            dashscope_client.TextEmbedding,
            "call",
            return_value=response([], status=401),
        ),
        pytest.raises(dashscope_client.EmbeddingError) as error,
    ):
        dashscope_client.embed_texts(["título"])
    assert "InvalidApiKey" in str(error.value)
    assert "secret-for-test" not in str(error.value)


def test_embed_texts_rejects_malformed_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-for-test")
    with (
        patch.object(
            dashscope_client.TextEmbedding,
            "call",
            return_value=response([[0.1, 0.2]]),
        ),
        pytest.raises(dashscope_client.EmbeddingError, match="dimensiones"),
    ):
        dashscope_client.embed_texts(["título"])


def test_rerank_titles_uses_server_key_and_preserves_api_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-for-test")
    expected = [(1, 0.9), (0, 0.4)]
    with patch.object(
        dashscope_client.TextReRank,
        "call",
        return_value=rerank_response(expected),
    ) as call:
        assert dashscope_client.rerank_titles("idea de tesis", ["título A", "título B"]) == expected

    kwargs = call.call_args.kwargs
    assert kwargs["api_key"] == "secret-for-test"
    assert kwargs["model"] == "qwen3-rerank"
    assert kwargs["query"] == "idea de tesis"
    assert kwargs["documents"] == ["título A", "título B"]
    assert kwargs["base_address"] == dashscope_client.BASE_URL


def test_rerank_titles_validates_document_limit_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-for-test")
    with patch.object(dashscope_client.TextReRank, "call") as call, pytest.raises(ValueError):
        dashscope_client.rerank_titles(
            "idea", ["título"] * (dashscope_client.MAX_RERANK_DOCUMENTS + 1)
        )
    call.assert_not_called()

    with (
        patch.object(
            dashscope_client.TextReRank,
            "call",
            return_value=rerank_response([(0, 0.5), (0, 0.4)]),
        ),
        pytest.raises(dashscope_client.RerankError, match="duplicados"),
    ):
        dashscope_client.rerank_titles("idea", ["título A", "título B"])


def test_rerank_titles_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with (
        patch.object(dashscope_client.TextReRank, "call") as call,
        pytest.raises(dashscope_client.RerankError, match="DASHSCOPE_API_KEY"),
    ):
        dashscope_client.rerank_titles("idea", ["título"])
    call.assert_not_called()
