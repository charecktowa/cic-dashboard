"""Vincula al personal investigador del CIC con OpenAlex y trae sus métricas y trabajos.

Dos pasos, ambos en tu máquina (el VPS nunca llama a OpenAlex):

    uv run --env-file .env python openalex.py match   # propone un autor por persona
    # revisa openalex_authors.csv a mano: corrige ids, borra los que no son
    uv run --env-file .env python openalex.py fetch   # métricas y trabajos de los revisados

Los datos de OpenAlex son CC0 (dominio público). Con OPENALEX_API_KEY en .env el
presupuesto diario es mayor; sin ella también alcanza para unas decenas de personas.
"""

import csv
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from text import words

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("CIC_DATA_DIR", ROOT / "data"))
MAPPING = ROOT / "openalex_authors.csv"
API = "https://api.openalex.org"
TOP_WORKS = 25
# Las tesis que indexa OpenAlex no se publican: ahí el autor suele ser el alumno. La app
# no incluye este script, así que advisor.HIDDEN_WORK_TYPES repite la lista.
HIDDEN_TYPES = ("dissertation",)
# Fuentes que son lugares donde se publica. Quedan fuera los repositorios (arXiv, SSRN,
# LA Referencia) y las plataformas de libros electrónicos (Springer eBooks).
VENUE_TYPES = ("journal", "conference", "book series")
# OpenAlex acepta hasta 100 valores unidos con "|" en un mismo filtro.
MAX_FILTER_VALUES = 100
IPN_NAMES = ("instituto politecnico nacional", "national polytechnic institute")
FIELDS = ["person", "openalex_ids", "display_name", "works_count", "institution", "status"]
# Estado de una fila revisada a mano: match la respeta.
REVIEWED = "corregido"
# Una llamada cada 0.2 s: lejos del límite de OpenAlex (10 por segundo sin llave, 30 con ella).
MIN_INTERVAL = 0.2
# Esperas antes de cada reintento; si el último también falla, el error se propaga.
BACKOFF = (1, 2, 4, 8)
# Demasiado rápido (429) o una falla pasajera del servidor: vale la pena reintentar.
RETRY_CODES = {429, 500, 502, 503, 504}
# Sin red un momento, o la conexión se corta a media respuesta: también se reintenta.
TRANSIENT_ERRORS = (
    urllib.error.URLError,
    http.client.IncompleteRead,
    TimeoutError,
    ConnectionError,
)
last_request = 0.0


def get(path: str, **params: str | int) -> dict[str, Any]:
    """Una llamada a la API de OpenAlex, con la llave si está en el entorno.

    La llave va en la cabecera y no en la URL: un error o un log muestran la URL. Las
    fallas pasajeras se reintentan con esperas crecientes (BACKOFF).
    """
    key = os.environ.get("OPENALEX_API_KEY")
    headers = {"User-Agent": "cic-dashboard", **({"Authorization": f"Bearer {key}"} if key else {})}
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=headers)
    for delay in BACKOFF:
        try:
            return fetch_json(request)
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_CODES:
                raise
            if error.code == 429 and error.headers.get("X-RateLimit-Remaining") == "0":
                raise SystemExit(
                    "Se acabó el presupuesto diario de OpenAlex; se renueva a medianoche UTC. "
                    "Una llave gratuita en OPENALEX_API_KEY lo multiplica."
                ) from error
        except TRANSIENT_ERRORS:
            pass
        time.sleep(delay)
    return fetch_json(request)


def fetch_json(request: urllib.request.Request) -> dict[str, Any]:
    """Manda la petición respetando MIN_INTERVAL desde la anterior."""
    global last_request
    pause = last_request + MIN_INTERVAL - time.monotonic()
    if pause > 0:
        time.sleep(pause)
    last_request = time.monotonic()
    # La URL siempre empieza con la constante API (https), nunca con file:// u otro.
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
        return json.load(response)


def is_ipn(author: dict[str, Any]) -> bool:
    institutions = [
        i.get("display_name") or "" for i in author.get("last_known_institutions") or []
    ]
    institutions += [
        (a.get("institution") or {}).get("display_name") or ""
        for a in author.get("affiliations") or []
    ]
    return any(" ".join(words(name)) in IPN_NAMES for name in institutions)


def name_score(person: str, author: dict[str, Any]) -> float:
    """Qué fracción de las palabras del nombre del CIC aparece en el nombre de OpenAlex."""
    wanted = set(words(person))
    names = [author.get("display_name") or ""] + (author.get("display_name_alternatives") or [])
    # Cada nombre por separado: unir todas las variantes inflaría el puntaje.
    return max((len(wanted & set(words(name))) / max(len(wanted), 1) for name in names), default=0)


def match() -> None:
    """Propone, para cada persona investigadora del CIC, su autor más probable en OpenAlex.

    Las filas marcadas a mano como `corregido` se conservan tal cual y no se vuelven a
    buscar: volver a correr match no deshace una corrección.
    """
    people = pl.read_parquet(DATA / "people.parquet").drop_nulls("lab")["person"].to_list()
    rows = [row for row in read_mapping() if row["status"] == REVIEWED]
    reviewed = {row["person"] for row in rows}
    for person in sorted(set(people) - reviewed):
        rows.append(propose(person))
        print(f"{rows[-1]['status']:>16}  {person}  →  {rows[-1].get('display_name', '')}")
    # Dos personas del CIC con un mismo autor de OpenAlex: al menos una está mal. Una fila
    # puede llevar varios ids, así que se cuentan uno por uno.
    claims = Counter(author_id for row in rows for author_id in set(split_ids(row)))
    for row in rows:
        if row["status"] != REVIEWED and any(claims[i] > 1 for i in split_ids(row)):
            row["status"] = "revisar"
    with MAPPING.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["person"]))
    print(f"\nRevisa {MAPPING.name}: corrige los ids marcados como 'revisar' y borra los erróneos.")


def propose(person: str) -> dict[str, Any]:
    """El autor de OpenAlex más probable para una persona, como fila del CSV."""
    candidates = get("/authors", search=person, per_page=10)["results"]
    scored = sorted(
        candidates,
        key=lambda a: (is_ipn(a), name_score(person, a), a.get("works_count", 0)),
        reverse=True,
    )
    if not scored:
        return {"person": person, "status": "sin coincidencia"}
    best = scored[0]
    institution = (best.get("last_known_institutions") or [{}])[0].get("display_name", "")
    confident = is_ipn(best) and name_score(person, best) >= 0.75
    return {
        "person": person,
        "openalex_ids": best["id"].rsplit("/", 1)[-1],
        "display_name": best.get("display_name", ""),
        "works_count": best.get("works_count", 0),
        "institution": institution,
        "status": "auto" if confident else "revisar",
    }


def split_ids(row: dict[str, Any]) -> list[str]:
    """Los ids de OpenAlex de una fila: una persona partida en varios perfiles lleva "A1|A2"."""
    return [i.strip() for i in (row.get("openalex_ids") or "").split("|") if i.strip()]


def read_mapping() -> list[dict[str, str]]:
    """Las filas de openalex_authors.csv; ninguna si todavía no existe."""
    if not MAPPING.exists():
        return []
    with MAPPING.open(newline="") as file:
        return list(csv.DictReader(file))


def works_filter(ids: list[str]) -> str:
    """Los trabajos de estos autores de OpenAlex, sin los tipos ocultos (tesis)."""
    return f"author.id:{'|'.join(ids)}" + "".join(f",type:!{t}" for t in HIDDEN_TYPES)


def venue_groups(ids: list[str]) -> list[tuple[str, int]]:
    """Los trabajos de estos autores por revista o congreso: (nombre, trabajos).

    Una sola consulta agrupada trae hasta 200 revistas, de la que más trabajos tiene a la
    que menos, y cuenta una vez un trabajo firmado por varios de estos autores. Solo cuentan
    los tipos de VENUE_TYPES; los trabajos sin revista no forman grupo.
    """
    venues_only = f",primary_location.source.type:{'|'.join(VENUE_TYPES)}"
    groups = get(
        "/works", filter=works_filter(ids) + venues_only, group_by="primary_location.source.id"
    )
    return [
        (group["key_display_name"], group["count"])
        for group in groups["group_by"]
        if group["key_display_name"]
    ]


def venue_counts(person: str, ids: list[str]) -> list[dict[str, Any]]:
    """Todos sus trabajos por revista o congreso, no solo los más citados."""
    return [{"person": person, "venue": v, "works": n} for v, n in venue_groups(ids)]


def cic_venue_counts(mapping: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Trabajos por revista de todo el CIC (lab nulo) y de cada laboratorio.

    Sumar las revistas de cada persona contaría dos veces un artículo con dos autores del
    CIC; una consulta con los ids de todo el grupo lo cuenta una vez.
    """
    people = pl.read_parquet(DATA / "people.parquet").drop_nulls("lab")
    labs = dict(people.select("person", "lab").iter_rows())
    groups: dict[str | None, list[str]] = {None: []}
    for row in mapping:
        ids = split_ids(row)
        groups[None] += ids
        if row["person"] in labs:
            groups.setdefault(labs[row["person"]], []).extend(ids)
    rows = []
    for lab, ids in groups.items():
        unique = list(dict.fromkeys(ids))
        if len(unique) > MAX_FILTER_VALUES:
            raise SystemExit(
                f"{len(unique)} ids de OpenAlex en {lab or 'el CIC'}: el filtro acepta hasta "
                f"{MAX_FILTER_VALUES}. Hay que partir la consulta."
            )
        if unique:
            rows += [{"lab": lab, "venue": v, "works": n} for v, n in venue_groups(unique)]
    return rows


def fetch() -> None:
    """Métricas, citas por año, revistas y trabajos más citados de los autores revisados."""
    mapping = [row for row in read_mapping() if row["openalex_ids"].strip()]
    authors, years, works, venues = [], [], [], []
    for row in mapping:
        person = row["person"]
        # Una persona puede estar partida en varios perfiles de OpenAlex: se suman.
        ids = split_ids(row)
        profiles = [get(f"/authors/{author_id}") for author_id in ids]
        stats = [p.get("summary_stats") or {} for p in profiles]
        topics = [t["display_name"] for p in profiles for t in (p.get("topics") or [])[:5]]
        authors.append(
            {
                "person": person,
                "openalex_ids": "|".join(ids),
                "works_count": sum(p.get("works_count", 0) for p in profiles),
                "cited_by_count": sum(p.get("cited_by_count", 0) for p in profiles),
                # El índice h no se suma; el del perfil principal es la mejor aproximación.
                "h_index": max((s.get("h_index", 0) for s in stats), default=0),
                "i10_index": max((s.get("i10_index", 0) for s in stats), default=0),
                "topics": " · ".join(dict.fromkeys(topics)),
            }
        )
        for p in profiles:
            for year in p.get("counts_by_year") or []:
                years.append(
                    {
                        "person": person,
                        "year": year["year"],
                        "works": year.get("works_count", 0),
                        "citations": year.get("cited_by_count", 0),
                    }
                )
        venues += venue_counts(person, ids)
        listed = get(
            "/works",
            filter=works_filter(ids),
            sort="cited_by_count:desc",
            per_page=TOP_WORKS,
            select="title,publication_year,type,doi,cited_by_count,open_access,primary_location",
        )["results"]
        for work in listed:
            source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            works.append(
                {
                    "person": person,
                    "title": work.get("title") or "",
                    "year": work.get("publication_year"),
                    "type": work.get("type") or "",
                    "venue": source or "",
                    "doi": work.get("doi") or "",
                    "cited_by_count": work.get("cited_by_count", 0),
                    "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
                }
            )
        print(f"{person}: {authors[-1]['cited_by_count']:,} citas, {len(listed)} trabajos")

    pl.DataFrame(authors).write_parquet(DATA / "openalex_authors.parquet")
    pl.DataFrame(years).group_by("person", "year").agg(pl.sum("works", "citations")).write_parquet(
        DATA / "openalex_years.parquet"
    )
    pl.DataFrame(works).write_parquet(DATA / "openalex_works.parquet")
    pl.DataFrame(
        venues, schema={"person": pl.String, "venue": pl.String, "works": pl.Int64}
    ).write_parquet(DATA / "openalex_venues.parquet")
    pl.DataFrame(
        cic_venue_counts(mapping),
        schema={"lab": pl.String, "venue": pl.String, "works": pl.Int64},
    ).write_parquet(DATA / "openalex_cic_venues.parquet")
    print(f"\n{len(authors)} perfiles de OpenAlex escritos en {DATA}")


if __name__ == "__main__":
    commands = {"match": match, "fetch": fetch}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Uso: python openalex.py match | fetch")
    commands[sys.argv[1]]()
