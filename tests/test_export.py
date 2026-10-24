"""La instantánea es lo único que se publica, así que se revisa qué contiene."""

import datetime

import polars as pl


def test_every_table_is_built(tables: dict[str, pl.DataFrame]) -> None:
    empty = [name for name, frame in tables.items() if frame.is_empty()]
    assert not empty, f"tablas vacías: {empty}"


def test_only_product_titles_are_published(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    products = set(catalog["outputs"]["title"])
    # El catálogo falso reutiliza la misma frase para tesis y productos; en los datos
    # reales no coinciden, así que solo se vigilan los títulos que no son de producto.
    hidden = (set(catalog["theses"]["title"]) | set(catalog["projects"]["title"])) - products
    for name, frame in tables.items():
        assert "title" not in frame.columns or name == "output_titles", name
        for column in frame.select(pl.col(pl.String)).columns:
            leaked = hidden & set(frame[column].drop_nulls())
            assert not leaked, f"{name}.{column} publica un título oculto"
    assert set(tables["output_titles"]["title"]) <= products


def test_no_future_years_reach_the_snapshot(tables: dict[str, pl.DataFrame]) -> None:
    this_year = datetime.date.today().year
    for name, frame in tables.items():
        if "year" in frame.columns:
            assert frame.filter(pl.col("year") > this_year).is_empty(), name


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
