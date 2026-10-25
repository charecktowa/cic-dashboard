"""Topes de búsquedas semánticas: cada una son llamadas que se cobran a la llave de DashScope.

La app corre en un solo proceso, así que un contador en memoria lo comparten todas las
sesiones; se reinicia con el contenedor, lo que solo puede dar unas búsquedas de más.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping

HOUR, DAY = 3600.0, 86400.0
# Con la idea más larga, una búsqueda cuesta ~$0.0017 (el reranker cobra la idea por cada
# uno de los 100 títulos): estos topes dejan el gasto en ~$3.40 al día como máximo.
PER_CLIENT_PER_HOUR = 20
GLOBAL_PER_HOUR = 300
GLOBAL_PER_DAY = 2000
# Con el índice ya preparado, cada búsqueda en curso suma ~5 MB y ~0.1 s de CPU; el tope
# acota memoria y CPU si llegan muchas a la vez. Las demás esperan su turno.
SEARCHES_AT_ONCE = 4


class SearchLimiter:
    """Ventanas deslizantes por visitante, por hora y por día, seguras entre hilos."""

    def __init__(
        self,
        *,
        per_client_per_hour: int = PER_CLIENT_PER_HOUR,
        global_per_hour: int = GLOBAL_PER_HOUR,
        global_per_day: int = GLOBAL_PER_DAY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_client_per_hour = per_client_per_hour
        self.global_per_hour = global_per_hour
        self.global_per_day = global_per_day
        self.clock = clock
        self._lock = threading.Lock()
        self._searches: deque[float] = deque()
        self._clients: dict[str, deque[float]] = {}

    def allow(self, client: str) -> str | None:
        """Registra una búsqueda si cabe en los topes; si no, devuelve cuál se alcanzó."""
        now = self.clock()
        with self._lock:
            drop_before(self._searches, now - DAY)
            for key in list(self._clients):
                drop_before(self._clients[key], now - HOUR)
                if not self._clients[key]:
                    del self._clients[key]
            mine = self._clients.get(client, deque())
            if len(self._searches) >= self.global_per_day:
                return "día"
            if sum(1 for t in self._searches if t > now - HOUR) >= self.global_per_hour:
                return "hora"
            if len(mine) >= self.per_client_per_hour:
                return "visitante"
            self._searches.append(now)
            mine.append(now)
            self._clients[client] = mine
            return None


def drop_before(times: deque[float], cutoff: float) -> None:
    """Quita de la izquierda los momentos anteriores a `cutoff` (la ventana va ordenada)."""
    while times and times[0] <= cutoff:
        times.popleft()


def client_key(headers: Mapping[str, str], ip_address: str | None) -> str:
    """Quién busca: la IP real que pone Cloudflare o, sin Cloudflare, la de la conexión.

    Detrás de Nginx Proxy Manager, la IP de la conexión es la del proxy. CF-Connecting-IP
    solo es confiable si al servidor únicamente llega Cloudflare (ver Seguridad en el README).
    """
    forwarded = headers.get("CF-Connecting-IP", "").strip()
    return forwarded or ip_address or "sin-ip"
