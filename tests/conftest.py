"""Una instantánea falsa para las pruebas.

CI no tiene la base de datos ni data/, así que aquí se inventa un catálogo pequeño
con la misma forma que devuelve `export.load()` y se pasa por `export.build()`, el
mismo camino que siguen los datos reales.
"""

import datetime
import os
import random
import tempfile
from pathlib import Path

import polars as pl
import pytest

# Debe fijarse antes de que algo importe `common`, que lee la ruta al importarse.
FAKE_DATA = Path(tempfile.mkdtemp(prefix="cic-data-"))
os.environ["CIC_DATA_DIR"] = str(FAKE_DATA)

import export  # noqa: E402

YEAR = datetime.date.today().year
TOPICS = {
    "Procesamiento de Lenguaje Natural": [
        "deep learning",
        "sentiment analysis",
        "redes neuronales",
        "hate speech",
    ],
    "Robótica y Mecatrónica": ["robots moviles", "control difuso", "navegacion autonoma"],
    "Ciberseguridad": ["criptografia ligera", "blockchain", "deteccion de intrusiones"],
    "Ciencias de la Información Cuántica": ["computacion cuantica", "algoritmos cuanticos"],
}
VENUES = ["Computación y Sistemas", "IEEE Access", "Research in Computing Science", "MICAI"]


def fake_catalog(seed: int = 7) -> dict[str, pl.DataFrame]:
    """Un catálogo inventado: 4 laboratorios, 15 personas y unos cientos de títulos."""
    rng = random.Random(seed)
    labs = list(TOPICS)
    lab_ids = {lab: i for i, lab in enumerate(labs, start=1)}

    people: list[tuple] = []
    for lab in labs:
        for _ in range(3):
            person_id = len(people) + 1
            people.append((person_id, f"Persona {person_id}", lab_ids[lab], lab))
    for i in range(3):
        people.append((len(people) + 1, f"Externa {i}", None, None))
    by_lab = {lab: [p[0] for p in people if p[3] == lab] for lab in labs}
    everyone = [p[0] for p in people]

    def title(lab: str) -> str:
        first, second = rng.sample(TOPICS[lab], 2)
        return f"Estudio de {first} con {second}"

    theses, participations = [], []
    for thesis_id in range(1, 81):
        lab = rng.choice(labs)
        director = rng.choice(by_lab[lab])
        theses.append(
            (thesis_id, title(lab), rng.choice(["Maestría", "Doctorado"]), lab_ids[lab], lab)
        )
        co_director = rng.choice([p for p in everyone if p != director])
        committee = rng.sample([p for p in everyone if p not in (director, co_director)], 2)
        participations += [
            (thesis_id, director, 1, "Director 1"),
            (thesis_id, co_director, 2, "Director 2"),
        ]
        participations += [(thesis_id, p, 3, "Comité tutorial") for p in committee]

    outputs, output_labs, output_authors = [], [], []
    for product_id in range(1, 241):
        own = rng.sample(labs, 2 if rng.random() < 0.25 else 1)
        kind = rng.choice(["Artículo", "Congreso"])
        outputs.append(
            (product_id, title(own[0]), kind, rng.randint(YEAR - 12, YEAR), rng.choice(VENUES))
        )
        output_labs += [(product_id, lab_ids[lab], lab) for lab in own]
        authors = {rng.choice(by_lab[lab]) for lab in own} | {rng.choice(by_lab[own[0]])}
        output_authors += [(product_id, person) for person in authors]

    projects, project_labs, project_members = [], [], []
    for project_id in range(1, 41):
        own = rng.sample(labs, 2 if rng.random() < 0.3 else 1)
        year = None if rng.random() < 0.1 else rng.randint(YEAR - 12, YEAR)
        projects.append((project_id, title(own[0]), year))
        project_labs += [(project_id, lab_ids[lab], lab) for lab in own]
        team = rng.sample(by_lab[own[0]], 2)
        project_members += [
            (project_id, team[0], "Responsable técnico"),
            (project_id, team[1], "Colaborador"),
        ]

    def frame(rows: list, columns: list[str]) -> pl.DataFrame:
        return pl.DataFrame(rows, schema=columns, orient="row")

    return {
        "labs": frame([(i, lab) for lab, i in lab_ids.items()], ["lab_id", "lab"]),
        "people": frame(people, ["person_id", "person", "lab_id", "lab"]),
        "theses": frame(theses, ["thesis_id", "title", "level", "lab_id", "lab"]),
        "participations": frame(participations, ["thesis_id", "person_id", "strength", "role"]),
        "outputs": frame(outputs, ["product_id", "title", "kind", "year", "venue"]),
        "output_labs": frame(output_labs, ["product_id", "lab_id", "lab"]),
        "output_authors": frame(output_authors, ["product_id", "person_id"]),
        "projects": frame(projects, ["project_id", "title", "year"]),
        "project_labs": frame(project_labs, ["project_id", "lab_id", "lab"]),
        "project_members": frame(project_members, ["project_id", "person_id", "role"]),
    }


@pytest.fixture(scope="session")
def catalog() -> dict[str, pl.DataFrame]:
    return fake_catalog()


@pytest.fixture(scope="session")
def tables(catalog: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    built = export.build(catalog)
    export.write(built, FAKE_DATA)
    write_fake_openalex()
    return built


def write_fake_openalex() -> None:
    """Datos de OpenAlex para una sola persona: las demás no tienen perfil vinculado."""
    pl.DataFrame(
        {
            "person": ["Persona 1"],
            "openalex_ids": ["A1"],
            "works_count": [40],
            "cited_by_count": [900],
            "h_index": [12],
            "i10_index": [15],
            "topics": ["Natural Language Processing"],
        }
    ).write_parquet(FAKE_DATA / "openalex_authors.parquet")
    pl.DataFrame(
        {"person": ["Persona 1"] * 3, "year": [YEAR - 2, YEAR - 1, YEAR], "works": [3, 4, 2]}
    ).with_columns(citations=pl.lit(50)).write_parquet(FAKE_DATA / "openalex_years.parquet")
    pl.DataFrame(
        {
            "person": ["Persona 1", "Persona 1"],
            "title": ["Un título con *asteriscos* y [corchetes]", "Otro trabajo"],
            "year": [YEAR - 1, None],
            "type": ["article", "conference-paper"],
            "venue": ["IEEE Access", ""],
            "doi": ["https://doi.org/10.1234/abc", ""],
            "cited_by_count": [120, 3],
            "is_oa": [True, False],
        }
    ).write_parquet(FAKE_DATA / "openalex_works.parquet")
