"""Prepara títulos públicos y sus vínculos para búsqueda semántica."""

import hashlib
import unicodedata
from pathlib import Path

import polars as pl

DATA = Path(__file__).parent / "data"
TITLE_DOCUMENT_SCHEMA = {
    "title_id": pl.String,
    "title": pl.String,
    "source": pl.String,
    "year": pl.Int32,
    "level": pl.String,
    "person": pl.String,
    "lab": pl.String,
    "role": pl.String,
}


def normalize_title(title: str) -> str:
    """Normaliza caja y espacios, conservando acentos y puntuación distintiva."""
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def title_id(title: str) -> str:
    """Identificador reproducible derivado del título, sin publicar ids del catálogo."""
    return hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()


def collect_title_documents(data_dir: Path = DATA) -> pl.DataFrame:
    """Una fila por vínculo público título-persona-fuente, sin nombres de estudiantes.

    Las tesis se relacionan solo con sus directores; los comités no cuentan como
    experiencia de dirección. OpenAlex se reserva para una fase posterior porque
    `openalex_works` solo contiene una selección de trabajos por citas.
    """
    required = ("thesis_titles", "output_titles", "project_titles", "people")
    tables = {name: pl.read_parquet(data_dir / f"{name}.parquet") for name in required}
    labs = dict(tables["people"].select("person", "lab").iter_rows())
    rows: list[dict] = []

    def add(
        title: str | None,
        source: str,
        *,
        year: int | None = None,
        level: str | None = None,
        person: str | None = None,
        lab: str | None = None,
        role: str | None = None,
    ) -> None:
        if not title or not title.strip():
            return
        rows.append(
            {
                "title_id": title_id(title),
                "title": title.strip(),
                "source": source,
                "year": year,
                "level": level,
                "person": person,
                "lab": lab or labs.get(person),
                "role": role,
            }
        )

    for thesis in tables["thesis_titles"].iter_rows(named=True):
        directors = [
            (role, thesis.get(column))
            for role, column in (("Director 1", "director_1"), ("Director 2", "director_2"))
            if thesis.get(column)
        ]
        if directors:
            for role, person in directors:
                add(
                    thesis.get("title"),
                    "Tesis",
                    level=thesis.get("level"),
                    person=person,
                    lab=thesis.get("lab"),
                    role=role,
                )
        else:
            add(thesis.get("title"), "Tesis", level=thesis.get("level"), lab=thesis.get("lab"))

    for output in tables["output_titles"].iter_rows(named=True):
        add(
            output.get("title"),
            output.get("kind") or "Producción CIC",
            year=output.get("year"),
            person=output.get("person"),
            role="Autor",
        )

    for project in tables["project_titles"].iter_rows(named=True):
        years = project.get("years") or []
        add(
            project.get("title"),
            "Proyecto",
            year=max(years) if years else None,
            person=project.get("person"),
            role=project.get("role"),
        )

    return (
        pl.DataFrame(rows, schema=TITLE_DOCUMENT_SCHEMA)
        .unique()
        .sort(["title_id", "source", "year", "level", "person", "role"], nulls_last=True)
    )
