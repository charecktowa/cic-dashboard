"""Construye la instantánea del tablero: archivos Parquet del catálogo CIC.

Todo lo que escribe este script está agregado: conteos por año, laboratorio,
rol, revista o término. La única excepción son los títulos de artículos y
congresos del personal investigador (nunca de tesis ni de proyectos). No sale
ningún id ni nombre de estudiante, así que la carpeta data/ se puede copiar a
un servidor público.

Se ejecuta donde la base de datos es accesible:

    uv run --group export python export.py
"""

import datetime
import os
from pathlib import Path
from typing import Any

import polars as pl
import psycopg

from text import ENGLISH_MARKERS, SPANISH_MARKERS, is_content, words_expr

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
    "password": os.environ.get("POSTGRES_PASSWORD", "theiss"),
    "dbname": os.environ.get("POSTGRES_DB", "cic_scrape"),
}
KINDS = {"publication": "Artículo", "conference": "Congreso"}
LEVELS = {"masters": "Maestría", "doctorate": "Doctorado"}
COMMITTEE = "Comité tutorial"


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
    # Dos personas pueden compartir nombre; un sufijo las mantiene separadas.
    people = people.with_columns(
        person=pl.when(pl.col("person").is_duplicated())
        .then(
            pl.col("person")
            + " ("
            + (pl.int_range(pl.len()).over("person") + 1).cast(pl.String)
            + ")"
        )
        .otherwise(pl.col("person"))
    )

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


def venues(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Productos por revista o congreso y tipo, en total (lab nulo) y por laboratorio."""
    scoped = (
        d["outputs"].drop_nulls("venue").with_columns(venue_key=words_expr("venue").list.join(" "))
    )
    names = scoped.group_by("venue_key").agg(venue=pl.col("venue").mode().first())
    overall = scoped.with_columns(lab=pl.lit(None, dtype=pl.String))
    per_lab = scoped.join(d["output_labs"].select("product_id", "lab"), on="product_id")
    return (
        pl.concat([overall, per_lab])
        .group_by("venue_key", "kind", "lab")
        .agg(outputs=pl.col("product_id").n_unique(), first=pl.min("year"), last=pl.max("year"))
        .join(names, on="venue_key")
        .drop("venue_key")
    )


def documents(d: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Cada título como documento, con sus laboratorios y sus personas."""
    docs = pl.concat(
        [
            d["theses"].select(
                doc_id=pl.format("thesis-{}", "thesis_id"),
                source=pl.lit("Tesis"),
                title="title",
                year=pl.lit(None, dtype=pl.Int32),
            ),
            d["outputs"].select(
                doc_id=pl.format("output-{}", "product_id"),
                source="kind",
                title="title",
                year=pl.col("year").cast(pl.Int32),
            ),
            d["projects"].select(
                doc_id=pl.format("project-{}", "project_id"),
                source=pl.lit("Proyecto"),
                title="title",
                year=pl.col("year").cast(pl.Int32),
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


def term_tables(
    docs: pl.DataFrame,
    doc_labs: pl.DataFrame,
    doc_people: pl.DataFrame,
    people: pl.DataFrame,
    kept: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Conteos de títulos por término, nunca los títulos."""
    tokens = docs.select("doc_id", word=words_expr("title")).explode("word").drop_nulls("word")

    language = (
        tokens.group_by("doc_id")
        .agg(
            es=pl.col("word").is_in(list(SPANISH_MARKERS)).sum(),
            en=pl.col("word").is_in(list(ENGLISH_MARKERS)).sum(),
        )
        .filter(pl.col("es") != pl.col("en"))
        .join(docs, on="doc_id")
        .group_by("source", "year")
        .agg(english=(pl.col("en") > pl.col("es")).sum(), titles=pl.len())
    )

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
        "language": language,
        "signature": signature,
    }


def output_titles(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Títulos de artículos y congresos del personal investigador.

    Nunca se incluyen títulos de tesis ni de proyectos: solo publicaciones del personal
    investigador, que ya son públicas y llevan su nombre en el sitio del CIC. Quien no
    tiene laboratorio (un asesor externo) no aparece, aunque firme un producto.
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
        .select("person", "title", "year", "kind")
        .unique()
        .sort("person", "year", descending=[False, True], nulls_last=True)
    )


def build(d: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Todas las tablas agregadas a partir de las tablas a nivel de registro."""
    # Un año futuro es un error de captura: no debe mover totales ni tendencias. Sus
    # laboratorios y autores también se van, o seguirían contando en pares y temas.
    outputs = d["outputs"].filter(pl.col("year").is_null() | (pl.col("year") <= CURRENT_YEAR))
    kept = outputs.select("product_id")
    d = {
        **d,
        "outputs": outputs,
        "output_labs": d["output_labs"].join(kept, on="product_id", how="semi"),
        "output_authors": d["output_authors"].join(kept, on="product_id", how="semi"),
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
        "output_titles": output_titles(d),
        **term_tables(
            docs["docs"],
            docs["doc_labs"],
            docs["doc_people"],
            d["people"],
            profiles["people"].select("person"),
        ),
    }
    return tables


def write(tables: dict[str, pl.DataFrame], out: Path) -> None:
    """Reemplaza la instantánea en `out` por estas tablas."""
    out.mkdir(exist_ok=True)
    # Los archivos openalex_* los escribe openalex.py y no dependen de esta exportación.
    for old in out.glob("*.parquet"):
        if not old.name.startswith("openalex_"):
            old.unlink()
    for name, frame in tables.items():
        frame.write_parquet(out / f"{name}.parquet")


def main() -> None:
    tables = build(load())
    write(tables, OUT)
    for name, frame in tables.items():
        print(f"{name:>18}: {frame.height:>7,} filas")
    print(f"Instantánea escrita en {OUT} ({datetime.date.today()})")


if __name__ == "__main__":
    main()
