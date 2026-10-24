"""La instantánea es lo único que se publica, así que se revisa qué contiene."""

import datetime
from collections import Counter
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import export


def test_every_table_is_built(tables: dict[str, pl.DataFrame]) -> None:
    empty = [name for name, frame in tables.items() if frame.is_empty()]
    assert not empty, f"tablas vacías: {empty}"


def test_only_staff_titles_are_published(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    products = set(catalog["outputs"]["title"])
    projects = set(catalog["projects"]["title"])
    theses = set(catalog["theses"]["title"])
    # Si se repitieran, un título filtrado pasaría por uno permitido y no se notaría.
    assert theses and projects
    assert not theses & (products | projects)
    assert not products & projects
    allowed = {"output_titles": products, "project_titles": projects, "thesis_titles": theses}
    for name, frame in tables.items():
        assert "title" not in frame.columns or name in allowed, name
        # Los títulos de tesis y de proyectos solo salen en su propia tabla.
        hidden = (theses | projects) - allowed.get(name, set())
        for column in frame.select(pl.col(pl.String)).columns:
            leaked = hidden & set(frame[column].drop_nulls())
            assert not leaked, f"{name}.{column} publica un título oculto"
    for name, titles in allowed.items():
        assert set(tables[name]["title"]) <= titles, name


def test_titles_belong_to_research_staff(tables: dict[str, pl.DataFrame]) -> None:
    people = tables["people"]
    externals = people.filter(pl.col("lab").is_null() & (pl.col("outputs") > 0))
    assert not externals.is_empty(), "el catálogo falso debe tener un externo con productos"
    staff = set(people.drop_nulls("lab")["person"])
    assert set(tables["output_titles"]["person"]) <= staff
    assert set(tables["project_titles"]["person"]) <= staff


def test_profiles_list_every_publication(tables: dict[str, pl.DataFrame]) -> None:
    listed = tables["output_titles"].group_by("person").agg(listed=pl.len())
    # Solo artículos y congresos llevan título publicado; un producto "Otro" no cuenta.
    counted = (
        tables["people"]
        .drop_nulls("lab")
        .with_columns(published=pl.col("articles") + pl.col("conferences"))
        .filter(pl.col("published") > 0)
    )
    both = counted.join(listed, on="person", how="left")
    assert both["listed"].to_list() == both["published"].to_list()
    repeated = tables["output_titles"].filter(pl.struct("person", "title", "year").is_duplicated())
    assert not repeated.is_empty(), "el catálogo falso debe tener un título repetido"


def test_no_future_years_reach_the_snapshot(tables: dict[str, pl.DataFrame]) -> None:
    this_year = datetime.date.today().year
    for name, frame in tables.items():
        if "year" in frame.columns:
            assert frame.filter(pl.col("year") > this_year).is_empty(), name


@pytest.mark.parametrize(("source", "key"), [("outputs", "product_id"), ("projects", "project_id")])
def test_future_items_leave_no_trace(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame], source: str, key: str
) -> None:
    this_year = datetime.date.today().year
    future = catalog[source].filter(pl.col("year") > this_year).select(key)
    assert not future.is_empty()
    # Quitarlos del catálogo antes de exportar debe dar exactamente la misma instantánea.
    without = {
        name: frame.join(future, on=key, how="anti") if key in frame.columns else frame
        for name, frame in catalog.items()
    }
    changed = []
    for name, frame in export.build(without).items():
        try:
            assert_frame_equal(frame, tables[name], check_row_order=False)
        except AssertionError:
            changed.append(name)
    assert not changed, f"un elemento futuro todavía cambia: {changed}"


def test_no_ids_reach_the_snapshot(tables: dict[str, pl.DataFrame]) -> None:
    for name, frame in tables.items():
        ids = [c for c in frame.columns if c.endswith("_id")]
        assert not ids, f"{name} contiene ids: {ids}"


def test_a_shared_output_counts_once_overall(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    overall = tables["outputs_by_year"].filter(pl.col("lab").is_null())["outputs"].sum()
    dated = catalog["outputs"].filter(pl.col("year") <= datetime.date.today().year)
    assert overall == dated.height


def homonyms(rows: list[tuple[int, str]]) -> dict[int, str]:
    people = pl.DataFrame(rows, schema=["person_id", "person"], orient="row")
    return dict(export.label_homonyms(people).iter_rows())


def test_homonyms_keep_their_number_between_exports() -> None:
    rows = [(7, "Ana López"), (3, "Ana López"), (5, "Luis Pérez"), (1, "Ana López")]
    expected = {1: "Ana López (1)", 3: "Ana López (2)", 7: "Ana López (3)", 5: "Luis Pérez"}
    # La base de datos no promete un orden: cualquiera debe dar los mismos nombres.
    assert homonyms(rows) == expected
    assert homonyms(rows[::-1]) == expected


def test_homonym_labels_never_take_an_existing_name() -> None:
    labels = homonyms([(1, "Ana López"), (2, "Ana López (2)"), (3, "Ana López")])
    assert labels == {1: "Ana López (1)", 2: "Ana López (2)", 3: "Ana López (3)"}
    # Ni siquiera cuando el nombre literal también se repite.
    labels = homonyms([(1, "Ana"), (2, "Ana"), (3, "Ana (1)"), (4, "Ana (1)")])
    assert len(set(labels.values())) == 4


def snapshot_files(out: Path) -> dict[str, pl.DataFrame]:
    return {path.stem: pl.read_parquet(path) for path in sorted(out.glob("*.parquet"))}


def test_write_replaces_the_snapshot(tmp_path: Path) -> None:
    openalex = pl.DataFrame({"person": ["Persona 1"]})
    openalex.write_parquet(tmp_path / "openalex_authors.parquet")
    export.write({"meta": pl.DataFrame({"v": [1]}), "stale": pl.DataFrame({"v": [1]})}, tmp_path)
    export.write({"meta": pl.DataFrame({"v": [2]})}, tmp_path)

    files = snapshot_files(tmp_path)
    # La tabla que ya no se exporta se va; la de OpenAlex no es de esta exportación.
    assert set(files) == {"meta", "openalex_authors"}
    assert files["meta"]["v"].to_list() == [2]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".staging-")]


def test_write_cleans_abandoned_staging(tmp_path: Path) -> None:
    abandoned = tmp_path / ".staging-muerta"
    abandoned.mkdir()
    pl.DataFrame({"v": [1]}).write_parquet(abandoned / "meta.parquet")
    with export.export_lock(tmp_path):
        export.write({"meta": pl.DataFrame({"v": [2]})}, tmp_path)
    assert not abandoned.exists()


def test_failed_write_keeps_the_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {"meta": pl.DataFrame({"v": [1]}), "totals": pl.DataFrame({"v": [1]})}
    export.write(before, tmp_path)

    original = pl.DataFrame.write_parquet

    def fail_on_totals(frame: pl.DataFrame, path: Path, **kwargs: object) -> None:
        if Path(path).stem == "totals":
            raise OSError("disco lleno")
        original(frame, path)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_on_totals)
    with pytest.raises(OSError, match="disco lleno"):
        export.write(
            {"meta": pl.DataFrame({"v": [2]}), "totals": pl.DataFrame({"v": [2]})}, tmp_path
        )

    files = snapshot_files(tmp_path)
    assert set(files) == set(before)
    assert all(frame["v"].to_list() == [1] for frame in files.values())
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".staging-")]


def test_person_venues_add_up_to_each_profile(tables: dict[str, pl.DataFrame]) -> None:
    by_person = tables["person_venues"].group_by("person").agg(pl.sum("outputs"))
    # En el catálogo falso todos los productos tienen revista o congreso.
    expected = tables["people"].filter(pl.col("outputs") > 0).select("person", "outputs")
    assert_frame_equal(by_person, expected, check_row_order=False, check_dtypes=False)
    # Una revista se llama igual en el perfil de una persona que en el de su laboratorio.
    assert set(tables["person_venues"]["venue"]) <= set(tables["venues"]["venue"])


def test_person_venues_match_the_catalog(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    # Calculado directo del catálogo: los totales podrían cuadrar aunque un producto
    # cayera en la revista equivocada. El catálogo falso escribe cada revista de una sola
    # forma, así que su nombre es el que se publica.
    dated = catalog["outputs"].filter(pl.col("year") <= datetime.date.today().year)
    expected = (
        catalog["output_authors"]
        .join(catalog["people"].select("person_id", "person"), on="person_id")
        .join(tables["people"].select("person"), on="person")
        .join(dated, on="product_id")
        .group_by("person", "venue", "kind")
        .agg(outputs=pl.col("product_id").n_unique(), first=pl.min("year"), last=pl.max("year"))
    )
    assert_frame_equal(
        tables["person_venues"],
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


def test_project_titles_group_each_project_once() -> None:
    def frame(rows: list, columns: list[str]) -> pl.DataFrame:
        return pl.DataFrame(rows, schema=columns, orient="row")

    d = {
        "people": frame(
            [(1, "Ana López", "IA"), (2, "Luis Pérez", None)], ["person_id", "person", "lab"]
        ),
        # El mismo proyecto registrado en dos años, una vez en mayúsculas y con otro rol;
        # otro sin año.
        "projects": frame(
            [
                (10, "Redes neuronales", 2023),
                (11, "REDES NEURONALES", 2021),
                (12, "Robots móviles", None),
            ],
            ["project_id", "title", "year"],
        ),
        "project_members": frame(
            [
                (10, 1, "Responsable técnico"),
                (11, 1, "Colaborador"),
                (12, 1, None),
                (10, 2, "Colaborador"),
            ],
            ["project_id", "person_id", "role"],
        ),
    }
    titles = export.project_titles(d).to_dicts()
    # Luis no tiene laboratorio en el CIC: no aparece.
    assert titles == [
        {
            "person": "Ana López",
            # Una sola fila, con el título y el rol de su registro más reciente.
            "title": "Redes neuronales",
            "years": [2021, 2023],
            "role": "Responsable técnico",
        },
        {"person": "Ana López", "title": "Robots móviles", "years": [], "role": "Indefinido"},
    ]


def test_thesis_titles_name_only_people_on_the_dashboard(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    theses = tables["thesis_titles"]
    # Ni alumno ni ids: si alguien agrega una columna, esta prueba obliga a revisarla.
    assert set(theses.columns) == {
        "title",
        "level",
        "lab",
        "director_1",
        "director_2",
        "committee",
        "others",
    }
    assert theses.height == catalog["theses"].height
    shown = set(tables["people"]["person"])
    named = (
        set(theses["director_1"].drop_nulls())
        | set(theses["director_2"].drop_nulls())
        | set(theses["committee"].explode().drop_nulls())
    )
    assert named <= shown
    # Quien participó sin estar en el tablero se cuenta, no se nombra.
    hidden = set(catalog["people"]["person"]) - shown
    assert hidden, "el catálogo falso debe tener a alguien fuera del tablero"
    assert theses["others"].sum() > 0


def test_thesis_titles_match_the_catalog(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    # Calculado directo del catálogo, tesis por tesis. Los títulos falsos se repiten, así
    # que se comparan tesis completas y no se busca por título.
    shown = set(tables["people"]["person"])
    names = dict(catalog["people"].select("person_id", "person").iter_rows())
    roles: dict[int, dict[str, list[str]]] = {}
    others: Counter[int] = Counter()
    for thesis_id, person_id, _, role in catalog["participations"].iter_rows():
        if names[person_id] in shown:
            roles.setdefault(thesis_id, {}).setdefault(role, []).append(names[person_id])
        else:
            others[thesis_id] += 1
    expected = Counter(
        (
            title,
            level,
            lab,
            roles.get(thesis_id, {}).get("Director 1", [None])[0],
            roles.get(thesis_id, {}).get("Director 2", [None])[0],
            tuple(sorted(roles.get(thesis_id, {}).get(export.COMMITTEE, []))),
            others[thesis_id],
        )
        for thesis_id, title, level, _, lab in catalog["theses"].iter_rows()
    )
    published = Counter(
        (
            row["title"],
            row["level"],
            row["lab"],
            row["director_1"],
            row["director_2"],
            tuple(row["committee"]),
            row["others"],
        )
        for row in tables["thesis_titles"].iter_rows(named=True)
    )
    assert published == expected


def test_venue_spellings_share_a_key() -> None:
    outputs = pl.DataFrame(
        {
            "product_id": [1, 2, 3, 4],
            # La forma minoritaria va primero: así no basta con mostrar la primera.
            "venue": [
                "Journal of Intelligent and Fuzzy Systems",
                "Journal of Intelligent & Fuzzy Systems",
                "Journal of Intelligent & Fuzzy Systems",
                "IEEE ROC&C 2014",
            ],
        }
    )
    scoped, names = export.keyed_venues(outputs)
    keys = scoped["venue_key"].to_list()
    assert keys[0] == keys[1] == keys[2] == "journal of intelligent and fuzzy systems"
    # Dentro de una sigla, "&" no se vuelve "and".
    assert keys[3] == "ieee roc c 2014"
    # Se muestra con la forma más frecuente.
    shown = names.filter(pl.col("venue_key") == keys[0])["venue"][0]
    assert shown == "Journal of Intelligent & Fuzzy Systems"


def test_export_without_password_says_where_to_put_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(export.PG, "password", None)
    monkeypatch.setitem(export.PG, "dbname", "otra_base")
    with pytest.raises(SystemExit) as stop:
        export.main()
    message = str(stop.value)
    # Dónde va la contraseña, cómo cargarla y de qué base: la que está configurada.
    assert "escribe ahí la contraseña" in message
    assert "a .env" in message
    assert "--env-file .env" in message
    assert "tu base otra_base" in message


@pytest.mark.parametrize(
    ("title", "original"),
    [
        (
            "Geocodificación de microblogs basada en ontologías "
            "(Geocoding of microblogs based on ontologies)",
            "Geocodificación de microblogs basada en ontologías",
        ),
        (
            "Metodología sistémica para la enseñanza [Systemic methodology for teaching]",
            "Metodología sistémica para la enseñanza",
        ),
        # Un paréntesis corto o a media frase es parte del título.
        (
            "Implementación de un controlador difuso (fuzzy controller) en un FPGA",
            "Implementación de un controlador difuso (fuzzy controller) en un FPGA",
        ),
        (
            "One More Geometric Interpretation of Pearson's Correlation (Aceptado)",
            "One More Geometric Interpretation of Pearson's Correlation (Aceptado)",
        ),
        # Un título que es todo paréntesis no se queda vacío.
        ("(Estudio de caso sobre redes neuronales)", "(Estudio de caso sobre redes neuronales)"),
    ],
)
def test_original_title_drops_a_trailing_translation(title: str, original: str) -> None:
    assert export.original_title(title) == original


def test_title_languages_read_grammar_not_borrowed_terms() -> None:
    titles = {
        # Términos técnicos en inglés dentro de un título en español.
        "Aplicaciones de seguridad con Man-in-the-Middle para casos de malware": "es",
        "Deep learning para detección de malware en Android": "es",
        # Títulos cortos, sin palabras como "de" o "the", que la heurística de antes empataba.
        "Automatic Sarcasm Detection": "en",
        "Parametric Type-2 Fuzzy Logic Systems": "en",
        "Apoyo encuentro Red Cómputo": "es",
        # Con su traducción al final, cuenta la parte original.
        "Dynamic Stability in an Endoreversible Chemical Reactor "
        "(Estabilidad Dinámica de un Reactor Químico Endorreversible)": "en",
        # Sin letras no hay con qué decidir.
        "2023": None,
    }
    assert export.title_languages(list(titles)) == list(titles.values())


def test_language_counts_every_title_once() -> None:
    docs = pl.DataFrame(
        {
            "doc_id": ["a", "b", "c", "d", "e"],
            "source": ["Artículo", "Artículo", "Tesis", "Tesis", "Tesis"],
            "title": [
                "Automatic Sarcasm Detection",
                "Detección automática de sarcasmo",
                "Sistema de recomendación geoespacial",
                "2023",
                None,
            ],
            "year": [2020, 2020, None, None, None],
            "level": [None, None, "Maestría", "Doctorado", "Doctorado"],
        },
        schema_overrides={"year": pl.Int32},
    )
    language = export.title_language(docs).sort("source")
    # Los títulos sin idioma (sin letras o vacíos) no se cuentan.
    assert language.select("source", "level", "english", "titles").rows() == [
        ("Artículo", None, 1, 2),
        ("Tesis", "Maestría", 0, 1),
    ]


def test_language_splits_theses_by_level(
    tables: dict[str, pl.DataFrame], catalog: dict[str, pl.DataFrame]
) -> None:
    language = tables["language"]
    # Solo las tesis tienen nivel, y cada nivel suma sus tesis del catálogo.
    assert language.filter(pl.col("level").is_not_null())["source"].unique().to_list() == ["Tesis"]
    theses = language.filter(pl.col("source") == "Tesis")
    counted = dict(theses.group_by("level").agg(pl.sum("titles")).iter_rows())
    expected = dict(catalog["theses"].group_by("level").agg(pl.len()).iter_rows())
    assert counted == expected
