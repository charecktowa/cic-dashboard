"""La instantánea es lo único que se publica, así que se revisa qué contiene."""

import datetime

import polars as pl
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


def test_no_future_years_reach_the_snapshot(tables: dict[str, pl.DataFrame]) -> None:
    this_year = datetime.date.today().year
    for name, frame in tables.items():
        if "year" in frame.columns:
            assert frame.filter(pl.col("year") > this_year).is_empty(), name


def test_future_products_leave_no_trace(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    this_year = datetime.date.today().year
    future = catalog["outputs"].filter(pl.col("year") > this_year).select("product_id")
    assert not future.is_empty()
    # Quitarlos del catálogo antes de exportar debe dar exactamente la misma instantánea.
    without = {
        name: frame.join(future, on="product_id", how="anti")
        if "product_id" in frame.columns
        else frame
        for name, frame in catalog.items()
    }
    changed = []
    for name, frame in export.build(without).items():
        try:
            assert_frame_equal(frame, tables[name], check_row_order=False)
        except AssertionError:
            changed.append(name)
    assert not changed, f"un producto futuro todavía cambia: {changed}"


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
