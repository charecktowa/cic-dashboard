"""Recupera títulos por búsqueda híbrida y agrupa evidencia por asesor o laboratorio."""

import math
from collections import Counter
from difflib import SequenceMatcher
from typing import cast

import numpy as np
import polars as pl

import text
from dashscope_client import DIMENSION, MODEL

EVIDENCE_LIMIT = 100
INDEX_COLUMNS = {"title_id", "title", "embedding", "model", "dimension", "source", "person", "lab"}
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
VARIANT_SIMILARITY = 0.82
VARIANT_WEIGHT = 0.65
MAX_TERM_VARIANTS = 4


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


def lexical_variants(query: str, titles: list[str]) -> dict[str, list[str]]:
    """Busca variantes léxicas cercanas presentes en el catálogo, sin modelo externo."""
    query_terms = text.terms(query)
    if not query_terms or not titles:
        return {}

    documents = [text.terms(title) for title in titles]
    vocabulary = {term for terms in documents for term in terms if " " not in term}
    variants: dict[str, list[str]] = {}
    for term in query_terms:
        if " " in term or term in vocabulary or len(term) < 6:
            continue
        candidates = [
            (SequenceMatcher(None, term, candidate).ratio(), candidate)
            for candidate in vocabulary
            if abs(len(term) - len(candidate)) <= 3
        ]
        matches = sorted(
            (
                (similarity, candidate)
                for similarity, candidate in candidates
                if similarity >= VARIANT_SIMILARITY
            ),
            key=lambda match: (-match[0], match[1]),
        )[:MAX_TERM_VARIANTS]
        if matches:
            variants[term] = [candidate for _, candidate in matches]
    return variants


def bm25_scores(query: str, titles: list[str]) -> np.ndarray:
    """Calcula BM25 y añade coincidencias aproximadas de términos poco comunes."""
    query_terms = text.terms(query)
    if not query_terms or not titles:
        return np.zeros(len(titles), dtype=np.float64)

    documents = [text.terms(title) for title in titles]
    variants = lexical_variants(query, titles)
    term_weights = dict.fromkeys(query_terms, 1.0)
    term_weights.update(
        {
            variant: max(term_weights.get(variant, 0.0), VARIANT_WEIGHT)
            for matches in variants.values()
            for variant in matches
        }
    )
    lengths = np.asarray([len(terms) for terms in documents], dtype=np.float64)
    average_length = float(lengths.mean()) if len(lengths) else 0.0
    if average_length == 0:
        return np.zeros(len(titles), dtype=np.float64)

    document_frequency = Counter(term for terms in documents for term in set(terms))
    total_documents = len(documents)
    scores = np.zeros(total_documents, dtype=np.float64)
    for term, weight in term_weights.items():
        frequency = document_frequency.get(term, 0)
        if not frequency:
            continue
        inverse_frequency = math.log1p((total_documents - frequency + 0.5) / (frequency + 0.5))
        for index, terms in enumerate(documents):
            term_frequency = terms.count(term)
            if term_frequency:
                denominator = term_frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * lengths[index] / average_length
                )
                scores[index] += (
                    weight * inverse_frequency * term_frequency * (BM25_K1 + 1) / denominator
                )
    return scores


def hybrid_rank_titles(index: pl.DataFrame, query: str, query_vector: list[float]) -> pl.DataFrame:
    """Fusiona los rankings semántico y BM25 sin mezclar sus escalas de puntuación."""
    semantic = rank_titles(index, query_vector)
    if semantic.is_empty():
        return semantic.with_columns(
            semantic_score=pl.Series([], dtype=pl.Float64),
            bm25_score=pl.Series([], dtype=pl.Float64),
        )

    title_ids = semantic["title_id"].to_list()
    lexical_scores = bm25_scores(query, semantic["title"].to_list())
    lexical_order = np.lexsort((np.asarray(title_ids), -lexical_scores))
    lexical_ranks = {
        title_ids[index]: rank
        for rank, index in enumerate(lexical_order, start=1)
        if lexical_scores[index] > 0
    }
    semantic_ranks = dict(zip(title_ids, semantic["rank"].to_list(), strict=True))
    fused_scores = [
        1 / (RRF_K + semantic_ranks[title_id])
        + (1 / (RRF_K + lexical_ranks[title_id]) if title_id in lexical_ranks else 0)
        for title_id in title_ids
    ]
    order = np.lexsort((np.asarray(title_ids), -np.asarray(fused_scores)))
    return semantic[order.tolist()].with_columns(
        semantic_score=pl.col("score"),
        bm25_score=pl.Series(lexical_scores[order], dtype=pl.Float64),
        score=pl.Series(np.asarray(fused_scores)[order], dtype=pl.Float64),
        rank=pl.Series(range(1, len(order) + 1), dtype=pl.UInt32),
    )


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
    schema = {
        group_by: pl.String,
        "score": pl.Float64,
        "affinity": pl.Float64,
        "matched": pl.String,
        "evidence_title": pl.String,
        "evidence_source": pl.String,
        "evidence_count": pl.UInt32,
    }
    if group_by == "person":
        schema["lab"] = pl.String
    if ranked.is_empty() or index.is_empty():
        return pl.DataFrame(schema=schema)

    links = (
        index.select("title_id", "title", "source", "person", "lab")
        .join(ranked.head(EVIDENCE_LIMIT), on="title_id", how="inner")
        .filter(pl.col(group_by).is_not_null())
        .sort("rank")
        .unique(subset=["title_id", "source", group_by], keep="first", maintain_order=True)
    )
    if links.is_empty():
        return pl.DataFrame(schema=schema)

    aggregations = {
        "score": pl.col("score").max(),
        "evidence_title": pl.col("title").first(),
        "evidence_source": pl.col("source").first(),
        "evidence_count": pl.col("title_id").n_unique().cast(pl.UInt32),
    }
    if group_by == "person":
        aggregations["lab"] = pl.col("lab").drop_nulls().first()

    grouped = (
        links.group_by(group_by, maintain_order=True)
        .agg(**aggregations)
        .with_columns(matched=pl.col("evidence_source") + pl.lit(" · ") + pl.col("evidence_title"))
        .sort(["score", group_by], descending=[True, False])
        .head(limit)
    )
    top = cast(float, grouped["score"].max() or 0.0)
    return grouped.with_columns(
        affinity=(pl.col("score").clip(0, top) / top * 100).round(0) if top > 0 else 0.0
    )
