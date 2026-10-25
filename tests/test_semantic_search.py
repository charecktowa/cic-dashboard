"""La recuperación y recomendación mantienen las obras como evidencia explícita."""

import polars as pl
import pytest

from semantic_search import (
    apply_rerank,
    bm25_scores,
    hybrid_rank_titles,
    lexical_variants,
    rank_titles,
    recommend,
    validate_index,
)


def index_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "title_id": ["a", "a", "b", "c", "d"],
            "title": [
                "Seguridad con malware",
                "Seguridad con malware",
                "Detección de software malicioso",
                "Robótica móvil",
                "Título sin asesor",
            ],
            "source": ["Tesis", "Tesis", "Artículo", "Proyecto", "Tesis"],
            "model": ["qwen3.7-text-embedding"] * 5,
            "dimension": [1024] * 5,
            "person": ["Asesora A", "Asesor B", "Asesora A", "Asesor C", None],
            "lab": ["Ciberseguridad", "Ciberseguridad", "Ciberseguridad", "Robótica", "Ciencias"],
            "embedding": [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
        }
    )


def test_rank_titles_deduplicates_shared_titles_and_uses_cosine() -> None:
    ranked = rank_titles(index_frame(), [1.0, 0.0])
    assert ranked["title_id"].to_list() == ["a", "b", "c", "d"]
    assert ranked["score"][0] == pytest.approx(1.0)
    assert ranked["rank"].to_list() == [1, 2, 3, 4]


def test_recommend_uses_best_evidence_and_preserves_co_advisors() -> None:
    index = index_frame()
    ranked = rank_titles(index, [1.0, 0.0])
    people = recommend(index, ranked, group_by="person", limit=5)
    rows = {row["person"]: row for row in people.iter_rows(named=True)}
    assert set(rows) == {"Asesora A", "Asesor B", "Asesor C"}
    assert rows["Asesora A"]["evidence_title"] == "Seguridad con malware"
    assert rows["Asesor B"]["evidence_title"] == "Seguridad con malware"
    assert rows["Asesora A"]["evidence_count"] == 2
    assert rows["Asesor B"]["evidence_count"] == 1
    assert rows["Asesora A"]["lab"] == "Ciberseguridad"
    assert "lab" in recommend(index, ranked.head(0), group_by="person").columns


def test_recommend_returns_empty_for_unknown_group_or_no_links() -> None:
    index = index_frame()
    ranked = rank_titles(index, [1.0, 0.0])
    assert recommend(index, ranked.head(0), group_by="person").is_empty()
    with pytest.raises(ValueError, match="group_by"):
        recommend(index, ranked, group_by="committee")


def test_rank_titles_checks_vector_dimensions() -> None:
    index = index_frame()
    with pytest.raises(ValueError, match="dimensión"):
        rank_titles(index, [1.0, 0.0, 0.0])


def test_validate_index_rejects_a_stale_model() -> None:
    index = index_frame().with_columns(model=pl.lit("otro-modelo"))
    with pytest.raises(ValueError, match="no son compatibles"):
        validate_index(index)


def test_bm25_matches_acronyms_and_hyphenated_terms() -> None:
    scores = bm25_scores(
        "Aplicaciones de seguridad con Man-in-the-Middle para casos de malware",
        [
            "Detección de software malicioso",
            "Análisis de malware en aplicaciones IoT",
            "Man-in-the-Middle malware detection",
        ],
    )
    assert scores[2] > scores[1] > scores[0]


def test_lexical_variants_match_spanish_and_english_forms() -> None:
    titles = [
        "Clasificación automática de hipertrofia adenoidea en imágenes médicas",
        "A Hybrid Approach for Adenoid Hypertrophy Detection in X-Ray Images",
        *(f"Detección de aplicaciones con redes neuronales {index}" for index in range(40)),
    ]
    variants = lexical_variants("detección de adenoides", titles)
    assert variants["adenoides"] == ["adenoidea", "adenoid"]
    scores = bm25_scores("detección de adenoides", titles)
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]


def test_hybrid_ranking_promotes_literal_match_without_losing_semantic_rank() -> None:
    index = pl.DataFrame(
        {
            "title_id": ["semantic", "exact", "unrelated"],
            "title": [
                "Detección de software malicioso",
                "Man-in-the-Middle malware detection",
                "Robótica móvil",
            ],
            "source": ["Tesis", "Artículo", "Proyecto"],
            "person": ["Asesora A", "Asesor B", "Asesor C"],
            "lab": ["Ciberseguridad", "Ciberseguridad", "Robótica"],
            "embedding": [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
            "model": ["qwen3.7-text-embedding"] * 3,
            "dimension": [1024] * 3,
        }
    )
    ranked = hybrid_rank_titles(
        index,
        "Aplicaciones de seguridad con Man-in-the-Middle para casos de malware",
        [1.0, 0.0],
    )
    assert ranked["title_id"].to_list()[0] == "exact"
    assert ranked["bm25_score"][0] > 0
    assert ranked["semantic_score"][0] < 1


def test_hybrid_ranking_uses_semantic_results_when_bm25_has_no_hits() -> None:
    index = index_frame()
    ranked = hybrid_rank_titles(index, "privacy healthcare", [0.0, 1.0])
    assert ranked["title_id"][0] == "c"
    assert ranked["bm25_score"].sum() == 0


def test_apply_rerank_reorders_candidates_and_keeps_hybrid_tail() -> None:
    ranked = pl.DataFrame(
        {
            "title_id": ["a", "b", "c"],
            "title": ["Título A", "Título B", "Título C"],
            "rank": [1, 2, 3],
            "score": [0.03, 0.02, 0.01],
        }
    )

    result = apply_rerank(ranked, [(1, 0.9), (0, 0.4)])

    assert result["title_id"].to_list() == ["b", "a", "c"]
    assert result["rank"].to_list() == [1, 2, 3]
    assert result["rerank_score"].to_list() == [0.9, 0.4, None]
