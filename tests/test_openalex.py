"""openalex.py habla con la API sin red: urlopen se reemplaza por respuestas fijas."""

import io
import json
import urllib.request

import pytest

import openalex


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[urllib.request.Request]:
    """Las peticiones que openalex.get() manda; cada una recibe un JSON vacío."""
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO(json.dumps({}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return requests


def test_the_key_goes_in_a_header_not_the_url(
    sent: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "llave-de-prueba")
    openalex.get("/authors", search="Ana López")
    (request,) = sent
    assert request.get_header("Authorization") == "Bearer llave-de-prueba"
    assert "llave-de-prueba" not in request.full_url


def test_without_a_key_there_is_no_authorization(
    sent: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Así queda en .env.example: la variable existe pero vacía.
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    openalex.get("/authors", search="Ana López")
    (request,) = sent
    assert request.get_header("Authorization") is None
