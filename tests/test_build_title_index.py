"""La generación del índice reutiliza vectores y no publica estados parciales."""

from pathlib import Path

import polars as pl
import pytest

from build_title_index import OUTPUT_NAME, build_title_index
from dashscope_client import DIMENSION, MODEL


def test_build_title_index_embeds_unique_titles_and_keeps_relationships(
    tables: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    data_dir = _copy_tables(tmp_path)
    calls: list[list[str]] = []

    def embed_batch(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(index + 1)] * DIMENSION for index, _ in enumerate(texts)]

    index = build_title_index(data_dir, embed_batch=embed_batch)
    assert sum(map(len, calls)) == index["title_id"].n_unique()
    assert max(map(len, calls)) <= 20
    assert index["embedding"].dtype == pl.List(pl.Float32)
    assert index["model"].unique().to_list() == [MODEL]
    assert index["dimension"].unique().to_list() == [DIMENSION]
    meta = pl.read_parquet(data_dir / "meta.parquet")
    assert index["snapshot"].unique().to_list() == meta["snapshot"].to_list()
    assert (
        index.filter(pl.col("source") == "Tesis")["role"]
        .drop_nulls()
        .is_in(["Director 1", "Director 2"])
        .all()
    )
    assert pl.read_parquet(data_dir / OUTPUT_NAME).equals(index)


def test_build_title_index_reuses_compatible_vectors(
    tables: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    data_dir = _copy_tables(tmp_path)
    old = build_title_index(data_dir, embed_batch=lambda texts: [[0.5] * DIMENSION for _ in texts])

    def unexpected_call(texts: list[str]) -> list[list[float]]:
        pytest.fail(f"el índice volvió a vectorizar {len(texts)} títulos")

    reused = build_title_index(data_dir, embed_batch=unexpected_call)
    assert reused.equals(old)


def test_failed_build_preserves_previous_index(
    tables: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    data_dir = _copy_tables(tmp_path)
    previous = pl.DataFrame({"marker": ["índice anterior"]})
    previous.write_parquet(data_dir / OUTPUT_NAME)

    def fail(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("fallo simulado")

    with pytest.raises(RuntimeError, match="simulado"):
        build_title_index(data_dir, embed_batch=fail)
    assert pl.read_parquet(data_dir / OUTPUT_NAME).equals(previous)


def _copy_tables(tmp_path: Path) -> Path:
    import os

    source = Path(os.environ["CIC_DATA_DIR"])
    for name in ("thesis_titles", "output_titles", "project_titles", "people", "meta"):
        (tmp_path / f"{name}.parquet").write_bytes((source / f"{name}.parquet").read_bytes())
    return tmp_path
