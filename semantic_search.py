"""Recupera títulos por búsqueda híbrida y agrupa evidencia por asesor o laboratorio."""

import datetime
import math
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import cast

import numpy as np
import polars as pl

import text
from dashscope_client import DIMENSION, MODEL

EVIDENCE_LIMIT = 100
# La profundidad de un asesor suma la relevancia de sus DEPTH obras más afines, la mejor
# completa, la segunda a la mitad y la tercera a un cuarto: premia a quien trabaja el tema de
# fondo, sin que gane por tener cientos de títulos ni que dos obras flojas superen a una muy
# afín (0.50 + 0.49 / 2 < 0.95).
DEPTH = 3
DEPTH_DECAY = 0.5
# Cuántas obras afines se muestran por asesor o laboratorio.
RELATED_SHOWN = 5
INDEX_COLUMNS = {"title_id", "title", "embedding", "model", "dimension", "source", "person", "lab"}
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
VARIANT_SIMILARITY = 0.82
VARIANT_WEIGHT = 0.65
MAX_TERM_VARIANTS = 4
# Letras iniciales que una variante debe compartir con la palabra de la idea.
PREFIX = 3
# Palabras más cortas suelen ser de relleno ("hacer", "usar"): no merecen un aviso.
MIN_UNCOVERED = 6
# Quién representa una obra con varios integrantes: el papel más directo primero.
ROLE_ORDER = {
    "Director 1": 0,
    "Responsable técnico": 0,
    "Autor": 1,
    "Director 2": 1,
    "Colaborador": 2,
}


def validate_index(index: pl.DataFrame) -> None:
    """Falla antes de llamar a la API si el índice no corresponde al modelo actual."""
    missing = INDEX_COLUMNS - set(index.columns)
    if missing:
        raise ValueError(f"Al índice semántico le faltan columnas: {', '.join(sorted(missing))}")
    models = set(index["model"].drop_nulls().unique().to_list())
    dimensions = set(index["dimension"].drop_nulls().unique().to_list())
    if models != {MODEL} or dimensions != {DIMENSION}:
        raise ValueError("El modelo o la dimensión del índice semántico no son compatibles")


@dataclass(frozen=True)
class Corpus:
    """Los títulos partidos en términos, con lo que BM25 necesita; se calcula una sola vez."""

    documents: list[list[str]]
    lengths: np.ndarray
    document_frequency: Counter[str]
    vocabulary: frozenset[str]

    @classmethod
    def of(cls, titles: list[str]) -> "Corpus":
        documents = [text.terms(title) for title in titles]
        return cls(
            documents=documents,
            lengths=np.asarray([len(terms) for terms in documents], dtype=np.float64),
            document_frequency=Counter(term for terms in documents for term in set(terms)),
            vocabulary=frozenset(t for terms in documents for t in terms if " " not in t),
        )


@dataclass(frozen=True)
class TitleIndex:
    """El índice listo para buscar. Lo que no depende de la idea se prepara una vez.

    Cada búsqueda solo multiplica la matriz por el vector de la idea y recorre los términos
    ya calculados: así varias búsquedas simultáneas caben en un contenedor de 512 MB.
    """

    links: pl.DataFrame
    titles: pl.DataFrame
    unit_vectors: np.ndarray
    corpus: Corpus
    # La instantánea con la que se construyó; None en un índice anterior a esta columna.
    snapshot: datetime.date | None = None


def prepare_index(index: pl.DataFrame) -> TitleIndex:
    """Valida el índice y precalcula los vectores unitarios y los términos de cada título."""
    validate_index(index)
    unique = index.unique(subset="title_id", keep="first", maintain_order=True)
    embeddings = unique["embedding"]
    widths = embeddings.list.len().unique()
    if unique.height and (widths.len() != 1 or widths[0] is None):
        raise ValueError("Los vectores del índice no tienen todos la misma dimensión")
    width = widths[0] if unique.height else 0
    # De Polars a NumPy sin pasar por listas de Python: .to_list() crea un float de Python por
    # valor, más de 250 MB con el índice real, y el contenedor tiene un límite de 512 MB.
    matrix = embeddings.list.to_array(width).to_numpy().astype(np.float32).reshape(-1, width)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Los embeddings deben tener norma distinta de cero")
    titles = unique.select("title_id", "title")
    return TitleIndex(
        links=index.select(
            "title_id",
            "title",
            "source",
            "person",
            "lab",
            role=pl.col("role") if "role" in index.columns else pl.lit(None, dtype=pl.String),
            year=pl.col("year") if "year" in index.columns else pl.lit(None, dtype=pl.Int32),
        ),
        titles=titles,
        unit_vectors=matrix / norms,
        corpus=Corpus.of(titles["title"].to_list()),
        snapshot=index["snapshot"][0] if "snapshot" in index.columns and index.height else None,
    )


def rank_titles(index: TitleIndex, query_vector: list[float]) -> pl.DataFrame:
    """Ordena títulos únicos por similitud coseno con la consulta."""
    if index.titles.is_empty():
        return pl.DataFrame(schema={"title_id": pl.String, "score": pl.Float64, "rank": pl.UInt32})
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or index.unit_vectors.shape[1] != query.size:
        raise ValueError("La dimensión del vector de consulta no coincide con el índice")
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        raise ValueError("Los embeddings deben tener norma distinta de cero")
    scores = index.unit_vectors @ (query / query_norm)
    order = np.lexsort((np.asarray(index.titles["title_id"]), -scores))
    return index.titles[order.tolist()].with_columns(
        score=pl.Series(scores[order], dtype=pl.Float64),
        rank=pl.Series(range(1, len(order) + 1), dtype=pl.UInt32),
    )


def lexical_variants(query: str, corpus: Corpus) -> dict[str, list[str]]:
    """Busca variantes léxicas cercanas presentes en el catálogo, sin modelo externo."""
    query_terms = text.terms(query)
    if not query_terms or not corpus.documents:
        return {}

    vocabulary = corpus.vocabulary
    variants: dict[str, list[str]] = {}
    for term in query_terms:
        if " " in term or term in vocabulary or len(term) < 6:
            continue
        candidates = [
            (SequenceMatcher(None, term, candidate).ratio(), candidate)
            for candidate in vocabulary
            # Mismo comienzo: "deteccion" y "detection" sí; "odontologia" y "ontologia", que
            # se parecen en 82% de sus letras, no.
            if abs(len(term) - len(candidate)) <= 3 and candidate[:PREFIX] == term[:PREFIX]
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


def uncovered_words(query: str, corpus: Corpus, variants: dict[str, list[str]]) -> list[str]:
    """Palabras de la idea que ningún título del catálogo menciona, ni con una variante.

    Es literal a propósito: con el embedding de una palabra suelta, las ausentes y las raras
    se parecen igual a los títulos (0.45–0.58 contra 0.51–0.68 de similitud), y ningún umbral
    las separa sin equivocarse con varias.
    """
    return [
        term
        for term in text.terms(query)
        if " " not in term
        and len(term) >= MIN_UNCOVERED
        and not corpus.document_frequency.get(term)
        and term not in variants
    ]


def bm25_scores(query: str, corpus: Corpus) -> np.ndarray:
    """Calcula BM25 y añade coincidencias aproximadas de términos poco comunes."""
    documents = corpus.documents
    query_terms = text.terms(query)
    if not query_terms or not documents:
        return np.zeros(len(documents), dtype=np.float64)

    variants = lexical_variants(query, corpus)
    term_weights = dict.fromkeys(query_terms, 1.0)
    term_weights.update(
        {
            variant: max(term_weights.get(variant, 0.0), VARIANT_WEIGHT)
            for matches in variants.values()
            for variant in matches
        }
    )
    lengths = corpus.lengths
    average_length = float(lengths.mean()) if len(lengths) else 0.0
    if average_length == 0:
        return np.zeros(len(documents), dtype=np.float64)

    document_frequency = corpus.document_frequency
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


def hybrid_rank_titles(index: TitleIndex, query: str, query_vector: list[float]) -> pl.DataFrame:
    """Fusiona los rankings semántico y BM25 sin mezclar sus escalas de puntuación."""
    semantic = rank_titles(index, query_vector)
    if semantic.is_empty():
        return semantic.with_columns(
            semantic_score=pl.Series([], dtype=pl.Float64),
            bm25_score=pl.Series([], dtype=pl.Float64),
        )

    # BM25 sale en el orden del índice; se reordena al del ranking semántico.
    positions = {title_id: i for i, title_id in enumerate(index.titles["title_id"].to_list())}
    title_ids = semantic["title_id"].to_list()
    lexical_scores = bm25_scores(query, index.corpus)[[positions[t] for t in title_ids]]
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


def apply_rerank(ranked: pl.DataFrame, rerank_results: list[tuple[int, float]]) -> pl.DataFrame:
    """Reordena la ventana de candidatos conservando el ranking híbrido fuera de ella.

    Dentro de la ventana, `score` pasa a ser la relevancia del reranker, para que la afinidad
    siga el nuevo orden; fuera de ella, recommend() no lee nada (EVIDENCE_LIMIT es la ventana).
    """
    if not rerank_results:
        return ranked
    # El orden lo da la relevancia, aunque la API no la devolviera ordenada.
    rerank_results = sorted(rerank_results, key=lambda result: -result[1])
    candidate_count = len(rerank_results)
    if candidate_count > ranked.height:
        raise ValueError("El reranker devolvió más candidatos que el ranking original")
    indices = [index for index, _ in rerank_results]
    if len(set(indices)) != candidate_count or any(
        index < 0 or index >= candidate_count for index in indices
    ):
        raise ValueError("El reranker devolvió índices inválidos")

    order = indices + list(range(candidate_count, ranked.height))
    scores_by_index = dict(rerank_results)
    reranked = ranked[order].with_columns(
        rerank_score=pl.Series([scores_by_index.get(index) for index in order], dtype=pl.Float64),
        rank=pl.Series(range(1, ranked.height + 1), dtype=pl.UInt32),
    )
    return reranked.with_columns(score=pl.coalesce("rerank_score", "score"))


def recommend(
    index: TitleIndex,
    ranked: pl.DataFrame,
    *,
    group_by: str,
    limit: int = 5,
) -> pl.DataFrame:
    """Agrupa los títulos más relevantes por persona o laboratorio.

    Primero se eligen candidatos: se recorren las obras de la mejor posicionada (el menor
    `rank`) hacia abajo, y cada obra respalda a un solo grupo nuevo, el de papel más directo
    (ROLE_ORDER). Así una obra con varios integrantes no llena la lista y los demás pueden
    entrar por otra obra suya; si sobran lugares, se llenan reutilizando obras. Luego los
    candidatos se ordenan por profundidad: la relevancia de sus DEPTH obras más afines, con
    descuento (DEPTH_DECAY) para cada obra después de la mejor, para que quien trabaja el
    tema de fondo quede arriba de quien tiene una sola obra parecida. La obra que lo hizo
    candidato queda como evidencia, y sus RELATED_SHOWN obras más afines, en `related`.
    """
    if group_by not in {"person", "lab"}:
        raise ValueError("group_by debe ser 'person' o 'lab'")
    if limit < 1:
        raise ValueError("limit debe ser positivo")
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        group_by: pl.String,
        "score": pl.Float64,
        "affinity": pl.Float64,
        "matched": pl.String,
        "evidence_title": pl.String,
        "evidence_source": pl.String,
        "evidence_count": pl.UInt32,
        "related": pl.List(pl.String),
    }
    if group_by == "person":
        schema["lab"] = pl.String
    if ranked.is_empty() or index.links.is_empty():
        return pl.DataFrame(schema=schema)

    # Se ordena por papel antes de quitar repetidos: si dos integrantes de un laboratorio
    # firman la misma obra, ese laboratorio la representa con el papel más directo.
    links = (
        index.links.join(ranked.head(EVIDENCE_LIMIT), on="title_id", how="inner")
        .filter(pl.col(group_by).is_not_null())
        .with_columns(role_order=pl.col("role").replace_strict(ROLE_ORDER, default=len(ROLE_ORDER)))
        .sort("rank", "role_order", group_by)
        .unique(subset=["title_id", "source", group_by], keep="first", maintain_order=True)
    )
    if links.is_empty():
        return pl.DataFrame(schema=schema)

    # El doble de candidatos de los que se muestran: la profundidad decide quiénes quedan.
    pool = 2 * limit
    chosen: dict[str, dict] = {}
    for _, work in links.group_by("title_id", maintain_order=True):
        newcomer = next((r for r in work.iter_rows(named=True) if r[group_by] not in chosen), None)
        if newcomer is not None:
            chosen[newcomer[group_by]] = newcomer
        if len(chosen) == pool:
            break
    # Lugares libres: quienes solo aparecen en obras ya usadas, en el orden de esas obras.
    for row in links.iter_rows(named=True):
        if len(chosen) == pool:
            break
        chosen.setdefault(row[group_by], row)
    order = {key: position for position, key in enumerate(chosen)}

    extra = {"lab": pl.col("lab").drop_nulls().first()} if group_by == "person" else {}
    # Una obra cuenta una vez aunque el catálogo la tenga dos veces con otras mayúsculas,
    # acentos, puntuación o guiones (52 obras lo están): si no, sumaría doble en la
    # profundidad. "Support-Decisions" y "Support Decisions" son la misma.
    by_relevance = (
        links.with_columns(work=text.words_expr("title").list.join(" ").str.replace_all("-", " "))
        .sort(["score", "rank"], descending=[True, False])
        .unique(subset=["work", group_by], keep="first", maintain_order=True)
    )
    position = pl.int_range(pl.len()).over(group_by)
    stats = (
        by_relevance.with_columns(
            weight=pl.when(position < DEPTH).then(DEPTH_DECAY**position).otherwise(0.0)
        )
        .group_by(group_by)
        .agg(
            depth=(pl.col("score") * pl.col("weight")).sum(),
            evidence_count=pl.col("work").n_unique().cast(pl.UInt32),
            # "Artículo · título (2024)": el año, cuando el catálogo lo tiene (las tesis no),
            # deja ver si la persona sigue trabajando el tema.
            related=(
                pl.col("source")
                + pl.lit(" · ")
                + pl.col("title")
                + pl.when(pl.col("year").is_not_null())
                .then(pl.lit(" (") + pl.col("year").cast(pl.String) + pl.lit(")"))
                .otherwise(pl.lit(""))
            ).head(RELATED_SHOWN),
            **extra,
        )
    )
    evidence = pl.DataFrame(
        [(key, row["title"], row["source"]) for key, row in chosen.items()],
        schema={group_by: pl.String, "evidence_title": pl.String, "evidence_source": pl.String},
        orient="row",
    )
    grouped = (
        evidence.join(stats, on=group_by, how="left")
        .with_columns(order=pl.col(group_by).replace_strict(order, return_dtype=pl.UInt32))
        .sort(["depth", "order"], descending=[True, False])
        .head(limit)
        .drop("order")
        .rename({"depth": "score"})
        .with_columns(matched=pl.col("evidence_source") + pl.lit(" · ") + pl.col("evidence_title"))
    )
    top = cast(float, grouped["score"].max() or 0.0)
    return grouped.with_columns(
        affinity=(pl.col("score").clip(0, top) / top * 100).round(0) if top > 0 else 0.0
    )
