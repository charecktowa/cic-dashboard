"""Los topes de búsquedas semánticas acotan lo que se puede gastar en la llave."""

import logging
import threading

import pytest

from search_limits import DAY, HOUR, SearchLimiter, client_key


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def limiter(clock: Clock, **limits: int) -> SearchLimiter:
    defaults = {"per_client_per_hour": 3, "global_per_hour": 5, "global_per_day": 8}
    return SearchLimiter(**(defaults | limits), clock=clock)


def test_a_visitor_gets_their_own_hourly_limit() -> None:
    clock = Clock()
    searches = limiter(clock)
    assert [searches.allow("ana") for _ in range(4)] == [None, None, None, "visitante"]
    # Otra persona no paga el límite de Ana.
    assert searches.allow("luis") is None
    clock.now += HOUR
    assert searches.allow("ana") is None


def test_the_whole_site_has_hourly_and_daily_limits() -> None:
    clock = Clock()
    searches = limiter(clock)
    assert [searches.allow(f"ip{i}") for i in range(6)][-1] == "hora"
    clock.now += HOUR
    assert [searches.allow(f"ip{i}") for i in range(6, 9)] == [None, None, None]
    assert searches.allow("otra") == "día"
    clock.now += DAY
    assert searches.allow("otra") is None


def test_each_limit_reached_is_logged_without_the_ip(caplog: pytest.LogCaptureFixture) -> None:
    searches = limiter(Clock(), per_client_per_hour=1)
    with caplog.at_level(logging.INFO, logger="cic.topes"):
        searches.allow("203.0.113.7")
        searches.allow("203.0.113.7")
    assert caplog.messages == ["tope alcanzado: visitante"]
    assert "203.0.113.7" not in caplog.text


def test_a_refused_search_is_not_counted() -> None:
    clock = Clock()
    searches = limiter(clock, per_client_per_hour=1)
    searches.allow("ana")
    for _ in range(10):
        assert searches.allow("ana") == "visitante"
    # Las negadas no consumieron el tope global.
    assert [searches.allow(f"ip{i}") for i in range(4)] == [None] * 4


def test_concurrent_searches_never_pass_the_limit() -> None:
    searches = SearchLimiter(per_client_per_hour=1000, global_per_hour=50, global_per_day=1000)
    allowed = []

    def search() -> None:
        allowed.append(searches.allow("bot"))

    threads = [threading.Thread(target=search) for _ in range(200)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert allowed.count(None) == 50


@pytest.mark.parametrize(
    ("headers", "ip", "expected"),
    [
        ({"CF-Connecting-IP": "203.0.113.7"}, "172.18.0.2", "203.0.113.7"),
        ({}, "172.18.0.2", "172.18.0.2"),
        ({"CF-Connecting-IP": "  "}, None, "sin-ip"),
    ],
)
def test_client_key_prefers_the_ip_cloudflare_saw(
    headers: dict[str, str], ip: str | None, expected: str
) -> None:
    assert client_key(headers, ip) == expected
