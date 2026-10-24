"""openalex.py habla con la API sin red: urlopen se reemplaza por respuestas fijas."""

import email.message
import http.client
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pytest

import openalex


@dataclass
class FakeApi:
    """Respuestas en orden (un dict o una excepción), lo que openalex.get() mandó y un reloj.

    El reloj solo avanza con sleep(), así que las esperas no dependen del tiempo real.
    """

    replies: list[dict | Exception] = field(default_factory=list)
    requests: list[urllib.request.Request] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)
    now: float = 1000.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    fake = FakeApi()

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        fake.requests.append(request)
        reply = fake.replies.pop(0) if fake.replies else {}
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    # openalex usa el reloj falso en lugar del módulo time; sin pausa entre llamadas salvo
    # que una prueba la pida.
    monkeypatch.setattr(openalex, "time", fake)
    monkeypatch.setattr(openalex, "MIN_INTERVAL", 0)
    monkeypatch.setattr(openalex, "last_request", 0.0)
    return fake


def http_error(code: int, **headers: str) -> urllib.error.HTTPError:
    message = email.message.Message()
    for name, value in headers.items():
        message[name] = value
    return urllib.error.HTTPError(openalex.API, code, "error", message, None)


def test_the_key_goes_in_a_header_not_the_url(
    api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "llave-de-prueba")
    openalex.get("/authors", search="Ana López")
    (request,) = api.requests
    assert request.get_header("Authorization") == "Bearer llave-de-prueba"
    assert "llave-de-prueba" not in request.full_url


def test_without_a_key_there_is_no_authorization(
    api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Así queda en .env.example: la variable existe pero vacía.
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    openalex.get("/authors", search="Ana López")
    (request,) = api.requests
    assert request.get_header("Authorization") is None


@pytest.mark.parametrize(
    "failure",
    [
        http_error(429),
        http_error(503),
        urllib.error.URLError("sin red"),
        # La conexión se corta o se queda callada a media respuesta.
        http.client.IncompleteRead(b"{"),
        TimeoutError("lectura"),
        ConnectionResetError("cortada"),
    ],
)
def test_passing_failures_are_retried(api: FakeApi, failure: Exception) -> None:
    api.replies = [failure, {"results": []}]
    assert openalex.get("/authors") == {"results": []}
    assert len(api.requests) == 2
    assert api.sleeps == [openalex.BACKOFF[0]]


def test_retries_stop_after_the_backoff(api: FakeApi) -> None:
    api.replies = [http_error(503)] * (len(openalex.BACKOFF) + 1)
    with pytest.raises(urllib.error.HTTPError):
        openalex.get("/authors")
    assert len(api.requests) == len(openalex.BACKOFF) + 1
    assert api.sleeps == list(openalex.BACKOFF)


def test_a_bad_request_is_not_retried(api: FakeApi) -> None:
    api.replies = [http_error(404)]
    with pytest.raises(urllib.error.HTTPError):
        openalex.get("/authors/A0")
    assert len(api.requests) == 1
    assert not api.sleeps


def test_an_exhausted_daily_budget_stops_at_once(api: FakeApi) -> None:
    api.replies = [http_error(429, **{"X-RateLimit-Remaining": "0"})]
    with pytest.raises(SystemExit, match="presupuesto diario"):
        openalex.get("/authors")
    assert len(api.requests) == 1


def test_calls_are_spaced_out(api: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openalex, "MIN_INTERVAL", 0.2)
    openalex.get("/authors")
    openalex.get("/authors")
    # El reloj no avanzó entre las dos llamadas: la segunda espera el intervalo completo.
    assert api.sleeps == [pytest.approx(0.2)]


# Un candidato que match aceptaría como "auto" (nombre y afiliación al IPN): si queda en
# "revisar", es solo porque su id choca con el de otra fila.
CONFIDENT_LUIS = {
    "id": "https://openalex.org/A9",
    "display_name": "Luis Pérez",
    "last_known_institutions": [{"display_name": "Instituto Politécnico Nacional"}],
}


def test_match_keeps_rows_reviewed_by_hand(
    api: FakeApi, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pl.DataFrame({"person": ["Ana López", "Luis Pérez"], "lab": ["IA", "IA"]}).write_parquet(
        tmp_path / "people.parquet"
    )
    mapping = tmp_path / "openalex_authors.csv"
    mapping.write_text(
        "person,openalex_ids,display_name,works_count,institution,status\n"
        "Ana López,A9,Ana López,12,Instituto Politécnico Nacional,corregido\n"
        "Luis Pérez,A1,Luis Pérez,3,,auto\n"
    )
    monkeypatch.setattr(openalex, "DATA", tmp_path)
    monkeypatch.setattr(openalex, "MAPPING", mapping)
    # OpenAlex propone para Luis el mismo autor que Ana ya tiene corregido.
    api.replies = [{"results": [CONFIDENT_LUIS]}]

    openalex.match()

    rows = {row["person"]: row for row in openalex.read_mapping()}
    assert rows["Ana López"]["openalex_ids"] == "A9"
    assert rows["Ana López"]["status"] == "corregido"
    assert rows["Luis Pérez"]["status"] == "revisar"
    # Solo se buscó a Luis: la fila corregida no vuelve a consultarse.
    assert len(api.requests) == 1
    assert "Luis" in api.requests[0].full_url


def test_fetch_counts_every_work_by_venue(
    api: FakeApi, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mapping = tmp_path / "openalex_authors.csv"
    mapping.write_text(
        "person,openalex_ids,display_name,works_count,institution,status\n"
        "Ana López,A9,Ana López,12,Instituto Politécnico Nacional,auto\n"
    )
    monkeypatch.setattr(openalex, "DATA", tmp_path)
    monkeypatch.setattr(openalex, "MAPPING", mapping)
    api.replies = [
        {
            "works_count": 12,
            "cited_by_count": 40,
            "summary_stats": {"h_index": 3},
            "counts_by_year": [{"year": 2024, "works_count": 2, "cited_by_count": 5}],
        },
        {
            "group_by": [
                {"key": "https://openalex.org/S1", "key_display_name": "IEEE Access", "count": 7},
                {"key": "https://openalex.org/S2", "key_display_name": "", "count": 2},
            ]
        },
        {"results": []},
    ]

    openalex.fetch()

    venues = pl.read_parquet(tmp_path / "openalex_venues.parquet")
    assert venues.to_dicts() == [{"person": "Ana López", "venue": "IEEE Access", "works": 7}]
    grouped = next(r for r in api.requests if "group_by" in r.full_url)
    # Los trabajos de esta persona, menos las tesis, que nunca se publican.
    assert "author.id%3AA9" in grouped.full_url
    assert "dissertation" in grouped.full_url


def test_match_compares_each_id_of_a_split_profile(
    api: FakeApi, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pl.DataFrame({"person": ["Ana López", "Luis Pérez"], "lab": ["IA", "IA"]}).write_parquet(
        tmp_path / "people.parquet"
    )
    mapping = tmp_path / "openalex_authors.csv"
    # Ana está partida en dos perfiles de OpenAlex.
    mapping.write_text(
        "person,openalex_ids,display_name,works_count,institution,status\n"
        "Ana López,A7|A9,Ana López,12,Instituto Politécnico Nacional,corregido\n"
    )
    monkeypatch.setattr(openalex, "DATA", tmp_path)
    monkeypatch.setattr(openalex, "MAPPING", mapping)
    api.replies = [{"results": [CONFIDENT_LUIS]}]

    openalex.match()

    rows = {row["person"]: row for row in openalex.read_mapping()}
    assert rows["Luis Pérez"]["status"] == "revisar"
    assert rows["Ana López"]["status"] == "corregido"
