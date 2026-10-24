"""La instantánea es lo único que se publica, así que se revisa qué contiene."""

import polars as pl


def test_every_table_is_built(tables: dict[str, pl.DataFrame]) -> None:
    empty = [name for name, frame in tables.items() if frame.is_empty()]
    assert not empty, f"tablas vacías: {empty}"


def test_no_title_reaches_the_snapshot(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    titles = set()
    for name in ["theses", "outputs", "projects"]:
        titles |= set(catalog[name]["title"])
    for name, frame in tables.items():
        assert "title" not in frame.columns, name
        for column in frame.select(pl.col(pl.String)).columns:
            leaked = titles & set(frame[column].drop_nulls())
            assert not leaked, f"{name}.{column} contiene títulos"


def test_no_ids_reach_the_snapshot(tables: dict[str, pl.DataFrame]) -> None:
    for name, frame in tables.items():
        ids = [c for c in frame.columns if c.endswith("_id")]
        assert not ids, f"{name} contiene ids: {ids}"


def test_a_shared_output_counts_once_overall(
    catalog: dict[str, pl.DataFrame], tables: dict[str, pl.DataFrame]
) -> None:
    overall = tables["outputs_by_year"].filter(pl.col("lab").is_null())["outputs"].sum()
    assert overall == catalog["outputs"].height
