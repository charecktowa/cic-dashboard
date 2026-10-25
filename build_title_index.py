"""Genera data/title_embeddings.parquet desde la instantánea de títulos públicos.

Ejecutar en la máquina que tiene la llave de DashScope:

    uv run --env-file .env python build_title_index.py
"""

import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from dashscope_client import DIMENSION, MAX_BATCH_SIZE, MODEL, embed_texts
from title_documents import DATA, collect_title_documents

OUTPUT_NAME = "title_embeddings.parquet"
MAX_CONCURRENT_REQUESTS = 4
EmbeddingBatch = Callable[[list[str]], list[list[float]]]


def build_title_index(
    data_dir: Path = DATA,
    *,
    embed_batch: EmbeddingBatch = embed_texts,
) -> pl.DataFrame:
    """Vectoriza títulos únicos y publica la tabla sin dejar archivos parciales."""
    documents = collect_title_documents(data_dir)
    if documents.is_empty():
        raise ValueError("No hay títulos públicos para indexar")

    output_path = data_dir / OUTPUT_NAME
    vectors: dict[str, list[float]] = {}
    if output_path.exists():
        cached = pl.read_parquet(output_path)
        if {"title_id", "embedding", "model", "dimension"} <= set(cached.columns):
            valid = (
                cached.filter((pl.col("model") == MODEL) & (pl.col("dimension") == DIMENSION))
                .select("title_id", "embedding")
                .unique("title_id")
            )
            vectors = dict(valid.iter_rows())

    unique_titles = documents.select("title_id", "title").unique(
        subset="title_id", keep="first", maintain_order=True
    )
    missing = [
        (row["title_id"], row["title"])
        for row in unique_titles.iter_rows(named=True)
        if row["title_id"] not in vectors
    ]
    batches = [
        missing[start : start + MAX_BATCH_SIZE] for start in range(0, len(missing), MAX_BATCH_SIZE)
    ]

    def run_batch(batch: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[list[float]]]:
        batch_vectors = embed_batch([title for _, title in batch])
        if len(batch_vectors) != len(batch):
            raise ValueError("El proveedor devolvió una cantidad inesperada de vectores")
        return batch, batch_vectors

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        for batch_number, (batch, batch_vectors) in enumerate(
            pool.map(run_batch, batches), start=1
        ):
            vectors.update(
                (title_id, vector)
                for (title_id, _), vector in zip(batch, batch_vectors, strict=True)
            )
            if batch_number % 25 == 0 or batch_number == len(batches):
                done = min(batch_number * MAX_BATCH_SIZE, len(missing))
                print(f"Títulos vectorizados: {done:,}/{len(missing):,}")

    embedded = documents.with_columns(
        embedding=pl.Series(
            [vectors[title_id] for title_id in documents["title_id"].to_list()],
            dtype=pl.List(pl.Float32),
        ),
        model=pl.lit(MODEL),
        dimension=pl.lit(DIMENSION, dtype=pl.Int16),
        # La instantánea de la que salieron los vínculos: la app no usa un índice de otra.
        snapshot=pl.lit(pl.read_parquet(data_dir / "meta.parquet")["snapshot"][0]),
    ).sort(["title_id", "source", "year", "level", "person", "role"], nulls_last=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    file_descriptor, staging_name = tempfile.mkstemp(
        prefix=".title-embeddings-", suffix=".parquet", dir=data_dir
    )
    os.close(file_descriptor)
    staging_path = Path(staging_name)
    try:
        embedded.write_parquet(staging_path, compression="zstd")
        os.replace(staging_path, output_path)
    finally:
        staging_path.unlink(missing_ok=True)
    return embedded


if __name__ == "__main__":
    index = build_title_index()
    print(
        f"Índice listo: {index.height:,} asociaciones, "
        f"{index['title_id'].n_unique():,} títulos únicos, "
        f"{OUTPUT_NAME} ({MODEL}, {DIMENSION} dimensiones)."
    )
