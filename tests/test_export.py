"""La instantánea es lo único que se publica, así que se revisa qué contiene."""

import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import export


def test_every_table_is_built(tables: dict[str, pl.DataFrame]) -> None:
    empty = [name for name, frame in tables.items() if frame.is_empty()]
    assert not empty, f"tablas vacías: {empty}"


def test_only_product_titles_are_published(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    products = set(catalog["outputs"]["title"])
    hidden = set(catalog["theses"]["title"]) | set(catalog["projects"]["title"])
    # Si se repitieran, un título filtrado pasaría por uno de producto y no se notaría.
    assert hidden
    assert not hidden & products
    for name, frame in tables.items():
        assert "title" not in frame.columns or name == "output_titles", name
        for column in frame.select(pl.col(pl.String)).columns:
            leaked = hidden & set(frame[column].drop_nulls())
            assert not leaked, f"{name}.{column} publica un título oculto"
    assert set(tables["output_titles"]["title"]) <= products


def test_titles_belong_to_research_staff(tables: dict[str, pl.DataFrame]) -> None:
    people = tables["people"]
    externals = people.filter(pl.col("lab").is_null() & (pl.col("outputs") > 0))
    assert not externals.is_empty(), "el catálogo falso debe tener un externo con productos"
    staff = set(people.drop_nulls("lab")["person"])
    assert set(tables["output_titles"]["person"]) <= staff


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
