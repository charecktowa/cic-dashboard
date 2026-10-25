"""Recupera títulos por similitud semántica y agrupa evidencia por asesor o laboratorio."""

from typing import cast

import numpy as np
import polars as pl

from dashscope_client import DIMENSION, MODEL

EVIDENCE_LIMIT = 100
INDEX_COLUMNS = {"title_id", "title", "embedding", "model", "dimension", "source", "person", "lab"}


def validate_index(index: pl.DataFrame) -> None:
    """Falla antes de llamar a la API si el índice no corresponde al modelo actual."""
    missing = INDEX_COLUMNS - set(index.columns)
    if missing:
        raise ValueError(f"Al índice semántico le faltan columnas: {', '.join(sorted(missing))}")
    models = set(index["model"].drop_nulls().unique().to_list())
    dimensions = set(index["dimension"].drop_nulls().unique().to_list())
    if models != {MODEL} or dimensions != {DIMENSION}:
        raise ValueError("El modelo o la dimensión del índice semántico no son compatibles")


def rank_titles(index: pl.DataFrame, query_vector: list[float]) -> pl.DataFrame:
    """Ordena títulos únicos por similitud coseno con la consulta."""
    validate_index(index)
    if index.is_empty():
        return pl.DataFrame(schema={"title_id": pl.String, "score": pl.Float64, "rank": pl.UInt32})

    titles = index.select("title_id", "title").unique(
        subset="title_id", keep="first", maintain_order=True
    )
    vectors = (
        index.select("title_id", "embedding")
        .unique(subset="title_id", keep="first", maintain_order=True)["embedding"]
        .to_list()
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.size:
        raise ValueError("La dimensión del vector de consulta no coincide con el índice")

    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0 or np.any(matrix_norms == 0):
        raise ValueError("Los embeddings deben tener norma distinta de cero")
    scores = matrix @ query / (matrix_norms * query_norm)
    order = np.lexsort((np.asarray(titles["title_id"]), -scores))
    ranked = titles[order.tolist()].with_columns(
        score=pl.Series(scores[order], dtype=pl.Float64),
        rank=pl.Series(range(1, len(order) + 1), dtype=pl.UInt32),
    )
    return ranked


def recommend(
    index: pl.DataFrame,
    ranked: pl.DataFrame,
    *,
    group_by: str,
    limit: int = 5,
) -> pl.DataFrame:
    """Agrupa los títulos más relevantes por persona o laboratorio.

    Cada grupo se ordena por su mejor título, para que quienes tienen más registros no
    acumulen una ventaja por volumen. La obra que lo justificó se conserva como evidencia.
    """
    if group_by not in {"person", "lab"}:
        raise ValueError("group_by debe ser 'person' o 'lab'")
    if limit < 1:
        raise ValueError("limit debe ser positivo")
    if ranked.is_empty() or index.is_empty():
        return pl.DataFrame(
            schema={
                group_by: pl.String,
                "score": pl.Float64,
                "affinity": pl.Float64,
                "matched": pl.String,
                "evidence_title": pl.String,
                "evidence_source": pl.String,
                "evidence_count": pl.UInt32,
            }
        )

    links = (
        index.select("title_id", "title", "source", "person", "lab")
        .join(ranked.head(EVIDENCE_LIMIT), on="title_id", how="inner")
        .filter(pl.col(group_by).is_not_null())
        .sort("rank")
        .unique(subset=["title_id", "source", group_by], keep="first", maintain_order=True)
    )
    if links.is_empty():
        return pl.DataFrame(
            schema={
                group_by: pl.String,
                "score": pl.Float64,
                "affinity": pl.Float64,
                "matched": pl.String,
                "evidence_title": pl.String,
                "evidence_source": pl.String,
                "evidence_count": pl.UInt32,
            }
        )

    grouped = (
        links.group_by(group_by, maintain_order=True)
        .agg(
            score=pl.col("score").max(),
            evidence_title=pl.col("title").first(),
            evidence_source=pl.col("source").first(),
            evidence_count=pl.col("title_id").n_unique().cast(pl.UInt32),
        )
        .with_columns(matched=pl.col("evidence_source") + pl.lit(" · ") + pl.col("evidence_title"))
        .sort(["score", group_by], descending=[True, False])
        .head(limit)
    )
    top = cast(float, grouped["score"].max() or 0.0)
    return grouped.with_columns(
        affinity=(pl.col("score").clip(0, top) / top * 100).round(0) if top > 0 else 0.0
    )
