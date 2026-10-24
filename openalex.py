"""Vincula al personal investigador del CIC con OpenAlex y trae sus métricas y trabajos.

Dos pasos, ambos en tu máquina (el VPS nunca llama a OpenAlex):

    uv run python openalex.py match   # propone un autor de OpenAlex por persona
    # revisa openalex_authors.csv a mano: corrige ids, borra los que no son
    uv run python openalex.py fetch   # trae métricas y trabajos de los ids revisados

Los datos de OpenAlex son CC0 (dominio público). Con OPENALEX_API_KEY en el entorno el
presupuesto diario es mayor; sin ella también alcanza para unas decenas de personas.
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

from text import words

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("CIC_DATA_DIR", ROOT / "data"))
MAPPING = ROOT / "openalex_authors.csv"
API = "https://api.openalex.org"
TOP_WORKS = 25
IPN_NAMES = ("instituto politecnico nacional", "national polytechnic institute")
FIELDS = ["person", "openalex_ids", "display_name", "works_count", "institution", "status"]


def get(path: str, **params: str | int) -> dict[str, Any]:
    """Una llamada a la API de OpenAlex, con la llave si está en el entorno."""
    key = os.environ.get("OPENALEX_API_KEY")
    query = {**params, **({"api_key": key} if key else {})}
    url = f"{API}{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "cic-dashboard"})
    for attempt in range(3):
        try:
            # La URL siempre empieza con la constante API (https), nunca con file:// u otro.
            with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


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
    """Propone, para cada persona investigadora del CIC, su autor más probable en OpenAlex."""
    people = pl.read_parquet(DATA / "people.parquet").drop_nulls("lab")["person"].to_list()
    rows = []
    for person in sorted(people):
        candidates = get("/authors", search=person, per_page=10)["results"]
        scored = sorted(
            candidates,
            key=lambda a: (is_ipn(a), name_score(person, a), a.get("works_count", 0)),
            reverse=True,
        )
        best = scored[0] if scored else None
        if best is None:
            rows.append({"person": person, "status": "sin coincidencia"})
            continue
        institution = (best.get("last_known_institutions") or [{}])[0].get("display_name", "")
        confident = is_ipn(best) and name_score(person, best) >= 0.75
        rows.append(
            {
                "person": person,
                "openalex_ids": best["id"].rsplit("/", 1)[-1],
                "display_name": best.get("display_name", ""),
                "works_count": best.get("works_count", 0),
                "institution": institution,
                "status": "auto" if confident else "revisar",
            }
        )
        print(f"{rows[-1]['status']:>16}  {person}  →  {rows[-1].get('display_name', '')}")
    # Dos personas del CIC con el mismo autor de OpenAlex: al menos una está mal.
    ids = [row.get("openalex_ids") for row in rows]
    for row in rows:
        if row.get("openalex_ids") and ids.count(row["openalex_ids"]) > 1:
            row["status"] = "revisar"
    with MAPPING.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRevisa {MAPPING.name}: corrige los ids marcados como 'revisar' y borra los erróneos.")


def fetch() -> None:
    """Métricas, citas por año y trabajos más citados de los autores revisados."""
    with MAPPING.open() as file:
        mapping = [row for row in csv.DictReader(file) if row["openalex_ids"].strip()]
    authors, years, works = [], [], []
    for row in mapping:
        person = row["person"]
        # Una persona puede estar partida en varios perfiles de OpenAlex: se suman.
        ids = [i.strip() for i in row["openalex_ids"].split("|") if i.strip()]
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
        listed = get(
            "/works",
            filter=f"author.id:{'|'.join(ids)}",
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
    print(f"\n{len(authors)} perfiles de OpenAlex escritos en {DATA}")


if __name__ == "__main__":
    commands = {"match": match, "fetch": fetch}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Uso: python openalex.py match | fetch")
    commands[sys.argv[1]]()
