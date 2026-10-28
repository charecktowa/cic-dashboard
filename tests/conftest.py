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
# Una tesis indexada en OpenAlex: nunca debe mostrarse.
OPENALEX_THESIS = "Mi tesis doctoral"
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

    # Cada fuente redacta distinto: si un título de tesis o de proyecto coincidiera con uno
    # de producto, la prueba de privacidad no podría distinguir cuál se filtró.
    def title(lab: str, opening: str = "Estudio de") -> str:
        first, second = rng.sample(TOPICS[lab], 2)
        return f"{opening} {first} con {second}"

    theses, participations = [], []
    for thesis_id in range(1, 81):
        lab = rng.choice(labs)
        director = rng.choice(by_lab[lab])
        theses.append(
            (
                thesis_id,
                title(lab, "Tesis sobre"),
                rng.choice(["Maestría", "Doctorado"]),
                lab_ids[lab],
                lab,
            )
        )
        co_director = rng.choice([p for p in everyone if p != director])
        committee = rng.sample([p for p in everyone if p not in (director, co_director)], 2)
        participations += [
            (thesis_id, director, 1, "Director 1"),
            (thesis_id, co_director, 2, "Director 2"),
        ]
        participations += [(thesis_id, p, 3, "Comité tutorial") for p in committee]

    # Una profesora que ya no está en el CIC y estuvo en un solo comité: no entra al
    # tablero, así que su tesis la cuenta sin nombrarla.
    retired = len(people) + 1
    people.append((retired, "Profesora Antigua", None, None))
    participations.append((1, retired, 3, "Comité tutorial"))

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
        projects.append((project_id, title(own[0], "Proyecto de"), year))
        project_labs += [(project_id, lab_ids[lab], lab) for lab in own]
        team = rng.sample(by_lab[own[0]], 2)
        project_members += [
            (project_id, team[0], "Responsable técnico"),
            (project_id, team[1], "Colaborador"),
        ]

    # Un asesor externo que firma un producto no es personal investigador: sus títulos no
    # se publican.
    externals = [p[0] for p in people if p[3] is None]
    output_authors.append((1, externals[0]))

    # Dos productos con el mismo título, año y tipo (por ejemplo, un artículo y su
    # reedición) son dos publicaciones, y el perfil debe listar ambas.
    reprint_id = 242
    _, same_title, same_kind, same_year, _ = outputs[0]
    outputs.append((reprint_id, same_title, same_kind, same_year, VENUES[1]))
    output_labs += [(reprint_id, lab_id, lab) for pid, lab_id, lab in output_labs if pid == 1]
    output_authors += [(reprint_id, person) for pid, person in output_authors if pid == 1]

    # Un año futuro es un error de captura: la instantánea debe descartarlo.
    future_id = 241
    outputs.append((future_id, title(labs[0]), "Artículo", YEAR + 1, VENUES[0]))
    # Compartido por dos laboratorios, para que también se note en lab_pairs.
    output_labs += [(future_id, lab_ids[lab], lab) for lab in labs[:2]]
    output_authors += [(future_id, by_lab[lab][0]) for lab in labs[:2]]
    # Lo mismo para un proyecto: no debe contar en totales, tendencias ni pares.
    future_project = 41
    projects.append((future_project, title(labs[0], "Proyecto de"), YEAR + 1))
    project_labs += [(future_project, lab_ids[lab], lab) for lab in labs[:2]]
    project_members += [(future_project, by_lab[lab][0], "Colaborador") for lab in labs[:2]]

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
    write_fake_openalex(built["people"])
    write_fake_snii()
    return built


def write_fake_snii() -> None:
    """Tres personas en el SNII: una vigente, una con la vigencia ya pasada y una emérita."""
    pl.DataFrame(
        {
            "person": ["Persona 1", "Persona 2", "Persona 4"],
            "level": ["Nivel II", "Nivel I", "Emérito"],
            "start": [YEAR - 1, YEAR - 5, YEAR - 10],
            "end": [YEAR + 3, YEAR - 1, None],
            "cut": ["2.º semestre de 2025"] * 3,
        },
        schema_overrides={"start": pl.Int32, "end": pl.Int32},
    ).write_parquet(FAKE_DATA / "snii.parquet")


def write_fake_openalex(people: pl.DataFrame) -> None:
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
            "person": ["Persona 1"] * 3,
            "title": ["Un título con *asteriscos* y [corchetes]", "Otro trabajo", OPENALEX_THESIS],
            "year": [YEAR - 1, None, YEAR - 9],
            "type": ["article", "conference-paper", "dissertation"],
            "venue": ["IEEE Access", "", ""],
            "doi": ["https://doi.org/10.1234/abc", "", ""],
            "cited_by_count": [120, 3, 500],
            "is_oa": [True, False, False],
        }
    ).write_parquet(FAKE_DATA / "openalex_works.parquet")
    pl.DataFrame(
        {"person": ["Persona 1"] * 2, "venue": ["IEEE Access", "arXiv"], "works": [9, 4]}
    ).write_parquet(FAKE_DATA / "openalex_venues.parquet")
    # Todo el CIC (lab nulo) y el laboratorio de Persona 1, la única con perfil.
    lab = people.filter(pl.col("person") == "Persona 1")["lab"][0]
    pl.DataFrame(
        {"lab": [None, lab], "venue": ["IEEE Access"] * 2, "works": [9, 9]},
        schema={"lab": pl.String, "venue": pl.String, "works": pl.Int64},
    ).write_parquet(FAKE_DATA / "openalex_cic_venues.parquet")
