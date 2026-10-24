"""Construye la instantánea del tablero: archivos Parquet del catálogo CIC.

Todo lo que escribe este script está agregado: conteos por año, laboratorio,
rol, revista o término. Las únicas excepciones son los títulos de artículos,
congresos y proyectos del personal investigador, y los de tesis con sus
directores y comité. No sale ningún id ni nombre de estudiante (el catálogo no
guarda al alumno), así que la carpeta data/ se puede copiar a un servidor
público.

Se ejecuta donde la base de datos es accesible:

    uv run --env-file .env --group export python export.py
"""

import datetime
import fcntl
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
import psycopg
from lingua import Language, LanguageDetectorBuilder

from text import is_content, words_expr

OUT = Path(__file__).parent / "data"
CURRENT_YEAR = datetime.date.today().year
SIGNATURE_TERMS = 15
# Los términos raros dicen poco y son lo más parecido a la huella de un solo título.
MIN_TERM_TITLES = 3
# Quien no tiene laboratorio en el CIC entra si participó en al menos estas tesis.
MIN_EXTERNAL_THESES = 3
TOP_COLLABORATORS = 10

PG: dict[str, Any] = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "theiss"),
    # Sin valor por defecto: la contraseña vive en .env, nunca en el código.
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "dbname": os.environ.get("POSTGRES_DB", "cic_scrape"),
}
KINDS = {"publication": "Artículo", "conference": "Congreso"}
LEVELS = {"masters": "Maestría", "doctorate": "Doctorado"}
COMMITTEE = "Comité tutorial"
# Los títulos del CIC están en español o en inglés: con solo esos dos candidatos, Lingua
# no confunde un título corto con otro idioma parecido.
LANGUAGES = {Language.SPANISH: "es", Language.ENGLISH: "en"}
# Una traducción al final, entre paréntesis o corchetes y de cuatro palabras o más, como en
# "Geocodificación de microblogs (Geocoding of microblogs based on ontologies)". El idioma es
# el de la parte original; un paréntesis corto o a media frase, como "(fuzzy controller)",
# es del título.
WORD_IN_BRACKETS = r"[^\s()\[\]]+"
TRANSLATION = re.compile(
    rf"\s*[(\[]\s*{WORD_IN_BRACKETS}(?:\s+{WORD_IN_BRACKETS}){{3,}}\s*[)\]]\s*$"
)


def query(sql: str) -> pl.DataFrame:
    """Consulta de solo lectura al catálogo CIC, como DataFrame de Polars."""
    with psycopg.connect(
        **PG,
        autocommit=True,
        # read_only se ignora en autocommit, así que se impone del lado del servidor.
        options="-c statement_timeout=20000 -c default_transaction_read_only=on",
    ) as conn:
        cursor = conn.execute(sql)
        columns = [column.name for column in cursor.description or []]
        return pl.DataFrame(cursor.fetchall(), schema=columns, orient="row")


def label_homonyms(people: pl.DataFrame) -> pl.DataFrame:
    """Un nombre distinto por persona: los homónimos llevan un sufijo " (n)".

    El nombre es la llave de cada persona en la instantánea, que no publica ids. El
    sufijo sigue el orden de `person_id`, así que la misma persona conserva su número
    entre exportaciones, y salta los nombres que ya existen: otra persona podría
    llamarse literalmente "Ana López (2)".
    """
    people = people.sort("person_id")
    names = people["person"].to_list()
    counts = Counter(names)
    taken = {name for name, n in counts.items() if n == 1}
    labels = []
    for name in names:
        if name is None or counts[name] == 1:
            labels.append(name)
            continue
        n = 1
        while f"{name} ({n})" in taken:
            n += 1
        taken.add(f"{name} ({n})")
        labels.append(f"{name} ({n})")
    return people.with_columns(person=pl.Series(labels, dtype=pl.String))


def load() -> dict[str, pl.DataFrame]:
    """Tablas a nivel de registro. Se quedan en memoria y nunca se escriben."""
    # Algunos nombres repiten el prefijo ("Laboratorio Laboratorio de ...").
    short_lab = pl.col("lab").str.replace(r"^(Laboratorio\s+)+(de\s+|del\s+)?", "")
    labs = query("SELECT id AS lab_id, name AS lab FROM catalog.laboratory").with_columns(
        lab=short_lab
    )
    lab_name = labs.select("lab_id", "lab")

    people = query(
        """
        SELECT p.id AS person_id, p.display_name AS person,
               (array_agg(pa.laboratory_id ORDER BY pa.is_primary DESC))[1] AS lab_id
        FROM catalog.person p
        LEFT JOIN catalog.person_affiliation pa ON pa.person_id = p.id
        GROUP BY p.id, p.display_name
        """
    ).join(lab_name, on="lab_id", how="left")
    people = label_homonyms(people)

    theses = query(
        """
        SELECT t.id AS thesis_id, t.title, pr.level::text AS level,
               tp.assigned_laboratory_id AS lab_id
        FROM catalog.thesis t
        LEFT JOIN catalog.program pr ON pr.code = t.program_code
        LEFT JOIN catalog.thesis_participant tp
               ON tp.thesis_id = t.id AND tp.role = 'primary_advisor'
        """
    ).with_columns(pl.col("level").replace_strict(LEVELS, default=None))
    theses = theses.join(lab_name, on="lab_id", how="left")

    # Una fila por (tesis, persona) con su rol más fuerte: la lista del comité
    # en la página de la tesis también nombra a los directores.
    participations = query(
        """
        SELECT tp.thesis_id, tp.person_id,
               min(CASE tp.role WHEN 'primary_advisor' THEN 1
                                WHEN 'co_advisor' THEN 2
                                ELSE 3 END) AS strength
        FROM catalog.thesis_participant tp
        WHERE tp.role <> 'unknown'
        GROUP BY tp.thesis_id, tp.person_id
        """
    ).with_columns(
        role=pl.col("strength").replace_strict(
            {1: "Director 1", 2: "Director 2", 3: COMMITTEE}, return_dtype=pl.String
        )
    )

    outputs = query(
        """
        SELECT rp.id AS product_id, rp.title, rp.kind::text AS kind,
               rp.year::int AS year, v.name AS venue
        FROM catalog.research_product rp
        LEFT JOIN catalog.venue v ON v.id = rp.venue_id
        """
    ).with_columns(pl.col("kind").replace_strict(KINDS, default="Otro"))
    output_labs = query(
        "SELECT product_id, laboratory_id AS lab_id FROM catalog.product_laboratory"
    ).join(lab_name, on="lab_id")
    output_authors = query("SELECT DISTINCT product_id, person_id FROM catalog.product_author")

    projects = query(
        "SELECT id AS project_id, title, year::int AS year FROM catalog.research_project"
    )
    project_labs = query(
        "SELECT project_id, laboratory_id AS lab_id FROM catalog.project_laboratory"
    ).join(lab_name, on="lab_id")
    project_members = query("SELECT project_id, person_id, role FROM catalog.project_member")

    return {
        "labs": labs,
        "people": people,
        "theses": theses,
        "participations": participations,
        "outputs": outputs,
        "output_labs": output_labs,
        "output_authors": output_authors,
        "projects": projects,
        "project_labs": project_labs,
        "project_members": project_members,
    }


def totals(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "metrica": [
                "Personas investigadoras",
                "Laboratorios",
                "Tesis",
                "Proyectos",
                "Artículos y congresos",
            ],
            "valor": [
                d["people"].drop_nulls("lab").height,
                d["labs"].height,
                d["theses"].height,
                d["projects"].height,
                d["outputs"].height,
            ],
        }
    )


def outputs_by_year(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Productos por año y tipo, una vez en total (lab nulo) y otra por laboratorio."""
    dated = d["outputs"].drop_nulls("year")
    overall = dated.group_by("year", "kind").agg(outputs=pl.len())
    per_lab = (
        dated.join(d["output_labs"], on="product_id")
        .group_by("year", "kind", "lab")
        .agg(outputs=pl.len())
    )
    return pl.concat([overall, per_lab], how="diagonal_relaxed").sort("year")


def lab_stats(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return (
        d["labs"]
        .select("lab")
        .join(
            d["people"].drop_nulls("lab").group_by("lab").agg(researchers=pl.len()),
            on="lab",
            how="left",
        )
        .join(
            d["theses"]
            .drop_nulls("lab")
            .group_by("lab")
            .agg(theses=pl.len(), doctorate=(pl.col("level") == "Doctorado").sum()),
            on="lab",
            how="left",
        )
        .join(
            d["output_labs"]
            .join(d["outputs"], on="product_id")
            .group_by("lab")
            .agg(
                outputs=pl.len(),
                recent_outputs=(pl.col("year") > CURRENT_YEAR - 5).sum(),
            ),
            on="lab",
            how="left",
        )
        .join(d["project_labs"].group_by("lab").agg(projects=pl.len()), on="lab", how="left")
        .fill_null(0)
        .with_columns(
            outputs_per_researcher=pl.when(pl.col("researchers") > 0).then(
                (pl.col("outputs") / pl.col("researchers")).round(1)
            ),
            doctorate_share=pl.when(pl.col("theses") > 0).then(
                pl.col("doctorate") / pl.col("theses")
            ),
        )
    )


def people_profiles(d: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Perfil agregado de cada persona investigadora o asesora frecuente."""
    roles = d["participations"].join(d["theses"].select("thesis_id", "level"), on="thesis_id")
    by_role = roles.group_by("person_id").agg(
        director_1=(pl.col("role") == "Director 1").sum(),
        director_2=(pl.col("role") == "Director 2").sum(),
        committee=(pl.col("role") == COMMITTEE).sum(),
        masters=(pl.col("level") == "Maestría").sum(),
        doctorate=(pl.col("level") == "Doctorado").sum(),
        theses=pl.len(),
    )
    authored = (
        d["output_authors"]
        .join(d["outputs"].select("product_id", "year", "kind"), on="product_id")
        .group_by("person_id")
        .agg(
            outputs=pl.len(),
            articles=(pl.col("kind") == "Artículo").sum(),
            conferences=(pl.col("kind") == "Congreso").sum(),
            first_year=pl.min("year"),
            last_year=pl.max("year"),
        )
    )
    led = (
        d["project_members"]
        .group_by("person_id")
        .agg(projects=pl.len(), projects_led=(pl.col("role") == "Responsable técnico").sum())
    )
    profiles = (
        d["people"]
        .select("person_id", "person", "lab")
        .join(by_role, on="person_id", how="left")
        .join(authored, on="person_id", how="left")
        .join(led, on="person_id", how="left")
        .with_columns(
            pl.exclude("person_id", "person", "lab", "first_year", "last_year").fill_null(0)
        )
        .filter(pl.col("lab").is_not_null() | (pl.col("theses") >= MIN_EXTERNAL_THESES))
    )
    kept = profiles.select("person_id", "person")

    person_year = (
        d["output_authors"]
        .join(kept, on="person_id")
        .join(d["outputs"].select("product_id", "year", "kind"), on="product_id")
        .drop_nulls("year")
        .group_by("person", "year", "kind")
        .agg(outputs=pl.len())
    )
    advisor_roles = (
        roles.join(kept, on="person_id")
        .drop_nulls("level")
        .group_by("person", "level", "role")
        .agg(theses=pl.len())
    )

    # Quién coincide con quién en las mismas tesis (como director o en el comité).
    pairs = (
        d["participations"]
        .join(kept, on="person_id")
        .select("thesis_id", "person")
        .join(
            d["participations"].join(kept, on="person_id").select("thesis_id", other="person"),
            on="thesis_id",
        )
        .filter(pl.col("person") != pl.col("other"))
        .group_by("person", "other")
        .agg(theses=pl.len())
        # El nombre desempata: sin él, el top de cada persona cambiaría entre exportaciones.
        .sort(["theses", "other"], descending=[True, False])
        .group_by("person", maintain_order=True)
        .head(TOP_COLLABORATORS)
    )
    return {
        "people": profiles.drop("person_id"),
        "person_year": person_year,
        "advisor_roles": advisor_roles,
        "person_pairs": pairs,
    }


def lab_pairs(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Elementos compartidos por cada par de laboratorios, por tipo de vínculo."""

    def shared(item_labs: pl.DataFrame, key: str) -> pl.DataFrame:
        return (
            item_labs.select(key, lab_a="lab")
            .join(item_labs.select(key, lab_b="lab"), on=key)
            .filter(pl.col("lab_a") != pl.col("lab_b"))
            .group_by("lab_a", "lab_b")
            .agg(items=pl.col(key).n_unique())
        )

    advisor_pairs = (
        d["participations"]
        .filter(pl.col("role") == "Director 2")
        .join(d["people"].select("person_id", lab_b="lab"), on="person_id")
        .join(d["theses"].select("thesis_id", lab_a="lab"), on="thesis_id")
        .drop_nulls(["lab_a", "lab_b"])
        .group_by("lab_a", "lab_b")
        .agg(items=pl.col("thesis_id").n_unique())
    )
    return pl.concat(
        [
            shared(d["output_labs"], "product_id").with_columns(
                link=pl.lit("Artículos y congresos")
            ),
            shared(d["project_labs"], "project_id").with_columns(link=pl.lit("Proyectos")),
            advisor_pairs.with_columns(link=pl.lit("Codirección de tesis")),
        ]
    )


def project_stats(d: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Proyectos por laboratorio, año y alcance, y por laboratorio, año y equipo.

    `lab` es nulo en las filas de todos los laboratorios, así que un proyecto de
    dos laboratorios cuenta una sola vez ahí.
    """
    lab_count = d["project_labs"].group_by("project_id").agg(labs=pl.len())
    team = d["project_members"].group_by("project_id").agg(members=pl.len())
    items = (
        d["projects"]
        .select("project_id", "year")
        .join(lab_count, on="project_id", how="left")
        .join(team, on="project_id", how="left")
        .with_columns(
            scope=pl.when(pl.col("labs") > 1)
            .then(pl.lit("Varios laboratorios"))
            .otherwise(pl.lit("Un laboratorio")),
            members=pl.col("members").fill_null(0),
        )
    )
    with_lab = pl.concat(
        [
            items.with_columns(lab=pl.lit(None, dtype=pl.String)),
            items.join(d["project_labs"].select("project_id", "lab"), on="project_id"),
        ]
    )
    by_scope = with_lab.group_by("lab", "year", "scope").agg(projects=pl.len())
    by_team = with_lab.group_by("lab", "year", "members").agg(projects=pl.len())
    return by_scope, by_team


def keyed_venues(outputs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Productos con revista o congreso, y un nombre por cada forma de escribirlo.

    Las variantes de escritura de una misma revista (con o sin acentos, mayúsculas, "&"
    o "and") comparten `venue_key` y se muestran con su nombre más frecuente.
    """
    scoped = (
        outputs.drop_nulls("venue")
        # "Intelligent & Fuzzy" y "Intelligent and Fuzzy" son la misma revista. Solo el "&"
        # entre palabras: dentro de una sigla, como ROC&C, no es un "and".
        .with_columns(spelled=pl.col("venue").str.replace_all(r"\s&\s", " and "))
        .with_columns(venue_key=words_expr("spelled").list.join(" "))
        .drop("spelled")
    )
    names = scoped.group_by("venue_key").agg(venue=pl.col("venue").mode().first())
    return scoped, names


def count_venues(scoped: pl.DataFrame, names: pl.DataFrame, by: str) -> pl.DataFrame:
    """Productos por revista o congreso, tipo y `by`, con su primer y último año."""
    return (
        scoped.group_by("venue_key", "kind", by)
        .agg(outputs=pl.col("product_id").n_unique(), first=pl.min("year"), last=pl.max("year"))
        .join(names, on="venue_key")
        .drop("venue_key")
    )


def venues(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Productos por revista o congreso y tipo, en total (lab nulo) y por laboratorio."""
    scoped, names = keyed_venues(d["outputs"])
    overall = scoped.with_columns(lab=pl.lit(None, dtype=pl.String))
    per_lab = scoped.join(d["output_labs"].select("product_id", "lab"), on="product_id")
    return count_venues(pl.concat([overall, per_lab]), names, "lab")


def person_venues(d: dict[str, pl.DataFrame], kept: pl.DataFrame) -> pl.DataFrame:
    """Productos de cada persona del tablero por revista o congreso y tipo."""
    scoped, names = keyed_venues(d["outputs"])
    authors = (
        d["output_authors"]
        .join(d["people"].select("person_id", "person"), on="person_id")
        .join(kept, on="person")
        .select("product_id", "person")
    )
    return count_venues(scoped.join(authors, on="product_id"), names, "person")


def documents(d: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Cada título como documento, con sus laboratorios y sus personas."""
    docs = pl.concat(
        [
            d["theses"].select(
                doc_id=pl.format("thesis-{}", "thesis_id"),
                source=pl.lit("Tesis"),
                title="title",
                year=pl.lit(None, dtype=pl.Int32),
                level="level",
            ),
            d["outputs"].select(
                doc_id=pl.format("output-{}", "product_id"),
                source="kind",
                title="title",
                year=pl.col("year").cast(pl.Int32),
                level=pl.lit(None, dtype=pl.String),
            ),
            d["projects"].select(
                doc_id=pl.format("project-{}", "project_id"),
                source=pl.lit("Proyecto"),
                title="title",
                year=pl.col("year").cast(pl.Int32),
                level=pl.lit(None, dtype=pl.String),
            ),
        ]
    )
    doc_labs = pl.concat(
        [
            d["theses"].select(doc_id=pl.format("thesis-{}", "thesis_id"), lab="lab").drop_nulls(),
            d["output_labs"].select(doc_id=pl.format("output-{}", "product_id"), lab="lab"),
            d["project_labs"].select(doc_id=pl.format("project-{}", "project_id"), lab="lab"),
        ]
    ).unique()
    # Los temas de una persona salen de las tesis que dirige (no del comité),
    # de sus artículos y congresos, y de sus proyectos.
    doc_people = pl.concat(
        [
            d["participations"]
            .filter(pl.col("role") != COMMITTEE)
            .select(doc_id=pl.format("thesis-{}", "thesis_id"), person_id="person_id"),
            d["output_authors"].select(
                doc_id=pl.format("output-{}", "product_id"), person_id="person_id"
            ),
            d["project_members"].select(
                doc_id=pl.format("project-{}", "project_id"), person_id="person_id"
            ),
        ]
    ).unique()
    return {"docs": docs, "doc_labs": doc_labs, "doc_people": doc_people}


def original_title(title: str) -> str:
    """El título sin la traducción que a veces lleva al final."""
    return TRANSLATION.sub("", title) or title


def title_languages(titles: list[str]) -> list[str | None]:
    """ "es" o "en" para cada título; None si no tiene con qué decidir, como "2023"."""
    detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
    found = detector.detect_languages_in_parallel_of([original_title(t) for t in titles])
    return [LANGUAGES.get(language) if language else None for language in found]


def title_language(docs: pl.DataFrame) -> pl.DataFrame:
    """Títulos en inglés y títulos con idioma, por fuente, año y nivel (solo las tesis)."""
    titled = docs.drop_nulls("title")
    return (
        titled.with_columns(
            language=pl.Series(title_languages(titled["title"].to_list()), dtype=pl.String)
        )
        .drop_nulls("language")
        .group_by("source", "year", "level")
        .agg(english=(pl.col("language") == "en").sum(), titles=pl.len())
    )


def term_tables(
    docs: pl.DataFrame,
    doc_labs: pl.DataFrame,
    doc_people: pl.DataFrame,
    people: pl.DataFrame,
    kept: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Conteos de títulos por término, nunca los títulos."""
    tokens = docs.select("doc_id", word=words_expr("title")).explode("word").drop_nulls("word")

    # Las palabras conservan su orden, así que las vecinas forman pares.
    content = tokens.filter(pl.col("word").map_elements(is_content, return_dtype=pl.Boolean))
    pairs = (
        content.with_columns(following=pl.col("word").shift(-1).over("doc_id"))
        .drop_nulls("following")
        .select("doc_id", term=pl.col("word") + " " + pl.col("following"))
    )
    doc_terms = pl.concat([content.select("doc_id", term="word"), pairs]).unique()
    term_df = (
        doc_terms.group_by("term").agg(titles=pl.len()).filter(pl.col("titles") >= MIN_TERM_TITLES)
    )
    doc_terms = doc_terms.join(term_df.select("term"), on="term")

    term_year = (
        doc_terms.join(docs.select("doc_id", "source", "year"), on="doc_id")
        .group_by("term", "source", "year")
        .agg(titles=pl.len())
    )
    term_lab = (
        doc_terms.join(docs.select("doc_id", "source"), on="doc_id")
        .join(doc_labs, on="doc_id")
        .group_by("term", "source", "lab")
        .agg(titles=pl.len())
    )
    person_docs = doc_people.join(people.select("person_id", "person"), on="person_id").join(
        kept, on="person"
    )
    term_person = (
        doc_terms.join(person_docs, on="doc_id").group_by("term", "person").agg(titles=pl.len())
    )
    lab_titles = doc_labs.group_by("lab").agg(titles=pl.len())
    person_titles = person_docs.group_by("person").agg(titles=pl.n_unique("doc_id"))

    # Términos distintivos: frecuentes en un laboratorio y raros en el resto,
    # puntuados como títulos × log(lift) para que un término exclusivo no sature.
    signature = (
        doc_terms.join(doc_labs, on="doc_id")
        .group_by("lab", "term")
        .agg(here=pl.len())
        .filter(pl.col("here") >= 4)
        .join(term_df.rename({"titles": "everywhere"}), on="term")
        .join(lab_titles.rename({"titles": "lab_titles"}), on="lab")
        .with_columns(
            lift=((pl.col("here") + 1) / (pl.col("lab_titles") + 1))
            / ((pl.col("everywhere") + 1) / (docs.height + 1))
        )
        .with_columns(score=pl.col("here") * pl.col("lift").log())
        .sort(["score", "term"], descending=[True, False])
        .group_by("lab", maintain_order=True)
        .head(SIGNATURE_TERMS)
        .drop("lab_titles")
    )
    return {
        "term_df": term_df,
        "term_year": term_year,
        "term_lab": term_lab,
        "term_person": term_person,
        "lab_titles": lab_titles,
        "person_titles": person_titles,
        "doc_year": docs.group_by("source", "year").agg(titles=pl.len()),
        "language": title_language(docs),
        "signature": signature,
    }


def output_titles(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Títulos de artículos y congresos del personal investigador.

    Solo publicaciones del personal investigador, que ya son públicas y llevan su nombre en
    el sitio del CIC; los proyectos y las tesis van aparte (project_titles, thesis_titles).
    Quien no tiene laboratorio (un asesor externo) no aparece, aunque firme un producto.
    """
    staff = d["people"].drop_nulls("lab").select("person_id", "person")
    return (
        d["output_authors"]
        .join(d["outputs"].select("product_id", "title", "year", "kind"), on="product_id")
        .join(staff, on="person_id")
        .filter(
            pl.col("kind").is_in(list(KINDS.values()))
            & pl.col("title").is_not_null()
            & (pl.col("title").str.len_chars() > 0)
        )
        # Dos productos con el mismo título, año y tipo siguen siendo dos: el perfil
        # los cuenta por separado y la lista debe coincidir con ese conteo.
        .unique(["person_id", "product_id"])
        .sort(["person", "year", "title"], descending=[False, True, False], nulls_last=True)
        .select("person", "title", "year", "kind")
    )


def project_titles(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Títulos de los proyectos del personal investigador, con sus años y el rol de cada uno.

    Un proyecto registrado en varios años (el mismo título, sin importar mayúsculas)
    aparece una vez con todos sus años, y con el título y el rol de su registro más
    reciente: alguien puede pasar de colaborador a responsable técnico. Como en
    output_titles, quien no tiene laboratorio en el CIC no aparece.
    """
    staff = d["people"].drop_nulls("lab").select("person_id", "person")
    title = pl.col("title").str.strip_chars()
    return (
        d["project_members"]
        .join(d["projects"].select("project_id", "title", "year"), on="project_id")
        .join(staff, on="person_id")
        .filter(title.str.len_chars() > 0)
        .with_columns(key=title.str.to_lowercase(), role=pl.col("role").fill_null("Indefinido"))
        .group_by("person", "key")
        # Título y rol del registro más reciente; el propio valor desempata para que el
        # elegido no cambie entre exportaciones.
        .agg(
            title=title.sort_by(["year", title], descending=[True, False], nulls_last=True).first(),
            role=pl.col("role")
            .sort_by(["year", "role"], descending=[True, False], nulls_last=True)
            .first(),
            years=pl.col("year").drop_nulls().unique().sort(),
        )
        .with_columns(last=pl.col("years").list.max())
        .sort(["person", "last", "title"], descending=[False, True, False], nulls_last=True)
        .select("person", "title", "years", "role")
    )


def thesis_titles(d: dict[str, pl.DataFrame], kept: pl.DataFrame) -> pl.DataFrame:
    """Títulos de tesis con su nivel, laboratorio, directores y comité; nunca el alumno.

    El catálogo no guarda al alumno, así que no hay forma de que salga. Solo se nombra a
    quien ya aparece en el tablero (`kept`); el resto de quienes participaron, como
    profesores que ya no están en el CIC, solo se cuenta en `others`. Cada persona cuenta
    una vez por tesis, con su rol más fuerte.
    """
    named = d["people"].select("person_id", "person").join(kept, on="person")
    person = pl.col("person")
    roles = (
        d["participations"]
        .join(named, on="person_id", how="left")
        .group_by("thesis_id")
        .agg(
            director_1=person.filter(pl.col("role") == "Director 1").first(),
            director_2=person.filter(pl.col("role") == "Director 2").first(),
            committee=person.filter(pl.col("role") == COMMITTEE).drop_nulls().sort(),
            others=person.is_null().sum(),
        )
    )
    return (
        d["theses"]
        .select("thesis_id", "title", "level", "lab")
        .filter(pl.col("title").str.strip_chars().str.len_chars() > 0)
        .join(roles, on="thesis_id", how="left")
        .with_columns(pl.col("others").fill_null(0))
        .sort("title")
        .drop("thesis_id")
    )


def build(d: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Todas las tablas agregadas a partir de las tablas a nivel de registro."""
    # Un año futuro es un error de captura: no debe mover totales ni tendencias. Sus
    # laboratorios, autores y equipos también se van, o seguirían contando en pares y temas.
    known = pl.col("year").is_null() | (pl.col("year") <= CURRENT_YEAR)
    outputs = d["outputs"].filter(known)
    projects = d["projects"].filter(known)
    kept, kept_projects = outputs.select("product_id"), projects.select("project_id")
    d = {
        **d,
        "outputs": outputs,
        "output_labs": d["output_labs"].join(kept, on="product_id", how="semi"),
        "output_authors": d["output_authors"].join(kept, on="product_id", how="semi"),
        "projects": projects,
        "project_labs": d["project_labs"].join(kept_projects, on="project_id", how="semi"),
        "project_members": d["project_members"].join(kept_projects, on="project_id", how="semi"),
    }
    profiles = people_profiles(d)
    docs = documents(d)
    by_scope, by_team = project_stats(d)
    tables = {
        "meta": pl.DataFrame(
            {"snapshot": [datetime.date.today()], "titles": [docs["docs"].height]}
        ),
        "totals": totals(d),
        "outputs_by_year": outputs_by_year(d),
        "lab_stats": lab_stats(d),
        **profiles,
        "lab_pairs": lab_pairs(d),
        "projects_by_scope": by_scope,
        "projects_by_team": by_team,
        "venues": venues(d),
        "person_venues": person_venues(d, profiles["people"].select("person")),
        "output_titles": output_titles(d),
        "project_titles": project_titles(d),
        "thesis_titles": thesis_titles(d, profiles["people"].select("person")),
        **term_tables(
            docs["docs"],
            docs["doc_labs"],
            docs["doc_people"],
            d["people"],
            profiles["people"].select("person"),
        ),
    }
    return tables


@contextmanager
def export_lock(out: Path) -> Iterator[None]:
    """Una sola exportación a la vez sobre `out`, de la lectura a la publicación.

    Si el candado cubriera solo la escritura, una exportación que leyó la base antes
    podría publicar después y deshacer una más reciente.
    """
    out.mkdir(exist_ok=True)
    with (out / ".export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write(tables: dict[str, pl.DataFrame], out: Path) -> None:
    """Reemplaza la instantánea en `out` por estas tablas. Se llama dentro de export_lock().

    Todas se escriben primero en una carpeta oculta dentro de `out` (el mismo disco,
    para que mover un archivo sea un rename atómico) y solo al final reemplazan a las
    anteriores: una exportación que falla a medias deja intacta la instantánea previa.
    """
    out.mkdir(exist_ok=True)
    # Una exportación que murió de golpe deja su carpeta temporal; con el candado puesto,
    # ninguna otra la está usando.
    for leftover in out.glob(".staging-*"):
        shutil.rmtree(leftover)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=out) as staging:
        staged = [Path(staging) / f"{name}.parquet" for name in tables]
        for path, frame in zip(staged, tables.values(), strict=True):
            frame.write_parquet(path)
        # Los archivos openalex_* los escribe openalex.py y no dependen de esta
        # exportación. Solo se borra lo que la nueva instantánea ya no trae.
        fresh = {path.name for path in staged}
        for old in out.glob("*.parquet"):
            if not old.name.startswith("openalex_") and old.name not in fresh:
                old.unlink()
        for path in staged:
            path.replace(out / path.name)


def main() -> None:
    if not PG["password"]:
        raise SystemExit(
            "Falta POSTGRES_PASSWORD. Copia .env.example a .env, escribe ahí la contraseña de "
            f"tu base {PG['dbname']} y corre con `uv run --env-file .env --group export python "
            "export.py` (ver README)."
        )
    with export_lock(OUT):
        tables = build(load())
        write(tables, OUT)
    for name, frame in tables.items():
        print(f"{name:>18}: {frame.height:>7,} filas")
    print(f"Instantánea escrita en {OUT} ({datetime.date.today()})")


if __name__ == "__main__":
    main()
