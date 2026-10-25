"""La recuperación y recomendación mantienen las obras como evidencia explícita."""

import polars as pl
import pytest

from semantic_search import (
    Corpus,
    apply_rerank,
    bm25_scores,
    hybrid_rank_titles,
    lexical_variants,
    prepare_index,
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
    ranked = rank_titles(prepare_index(index_frame()), [1.0, 0.0])
    assert ranked["title_id"].to_list() == ["a", "b", "c", "d"]
    assert ranked["score"][0] == pytest.approx(1.0)
    assert ranked["rank"].to_list() == [1, 2, 3, 4]


def test_recommend_keeps_co_advisors_when_there_is_room() -> None:
    index = prepare_index(index_frame())
    ranked = rank_titles(index, [1.0, 0.0])
    people = recommend(index, ranked, group_by="person", limit=5)
    rows = {row["person"]: row for row in people.iter_rows(named=True)}
    assert set(rows) == {"Asesora A", "Asesor B", "Asesor C"}
    # "Seguridad con malware" es de las dos personas: respalda a una, y Asesora A entra por
    # su otra obra. Asesor B solo tiene esa, y entra porque sobra lugar.
    assert rows["Asesor B"]["evidence_title"] == "Seguridad con malware"
    assert rows["Asesora A"]["evidence_title"] == "Detección de software malicioso"
    assert rows["Asesora A"]["evidence_count"] == 2
    assert rows["Asesor B"]["evidence_count"] == 1
    assert rows["Asesora A"]["lab"] == "Ciberseguridad"
    assert "lab" in recommend(index, ranked.head(0), group_by="person").columns


SHARED = "Aplicaciones de aprendizaje profundo"


def shared_work_frame() -> pl.DataFrame:
    """Un proyecto genérico de tres personas arriba; cada una tiene otra obra más abajo."""
    rows = [
        ("p", SHARED, "Proyecto", "Colaboradora", "Colaborador", [1.0, 0.0]),
        ("p", SHARED, "Proyecto", "Responsable", "Responsable técnico", [1.0, 0.0]),
        ("p", SHARED, "Proyecto", "Otro colaborador", "Colaborador", [1.0, 0.0]),
        ("t1", "Segmentación de radiografías", "Tesis", "Colaboradora", "Director 1", [0.9, 0.1]),
        ("t2", "Clasificación de imágenes", "Tesis", "Otro colaborador", "Director 1", [0.8, 0.2]),
        ("t3", "Diagnóstico por visión", "Artículo", "Tercera persona", "Autor", [0.7, 0.3]),
    ]
    columns = ["title_id", "title", "source", "person", "role", "embedding"]
    return pl.DataFrame(rows, schema=columns, orient="row").with_columns(
        lab=pl.col("person"),
        model=pl.lit("qwen3.7-text-embedding"),
        dimension=pl.lit(1024),
    )


def test_one_work_backs_one_person_the_most_direct_role() -> None:
    index = prepare_index(shared_work_frame())
    people = recommend(index, rank_titles(index, [1.0, 0.0]), group_by="person", limit=3)
    # El proyecto lo representa su responsable técnico; las otras dos personas entran por su
    # tesis, y el proyecto no se lleva los tres lugares.
    assert people.select("person", "evidence_title").rows() == [
        ("Responsable", SHARED),
        ("Colaboradora", "Segmentación de radiografías"),
        ("Otro colaborador", "Clasificación de imágenes"),
    ]
    assert people["affinity"].n_unique() == 3


def test_one_work_backs_one_lab() -> None:
    index = prepare_index(shared_work_frame())
    labs = recommend(index, rank_titles(index, [1.0, 0.0]), group_by="lab", limit=2)
    assert labs["lab"].to_list() == ["Responsable", "Colaboradora"]
    assert labs["affinity"].to_list()[1] < 100


def test_recommend_returns_empty_for_unknown_group_or_no_links() -> None:
    index = prepare_index(index_frame())
    ranked = rank_titles(index, [1.0, 0.0])
    assert recommend(index, ranked.head(0), group_by="person").is_empty()
    with pytest.raises(ValueError, match="group_by"):
        recommend(index, ranked, group_by="committee")


def test_rank_titles_checks_vector_dimensions() -> None:
    index = prepare_index(index_frame())
    with pytest.raises(ValueError, match="dimensión"):
        rank_titles(index, [1.0, 0.0, 0.0])


def test_validate_index_rejects_a_stale_model() -> None:
    index = index_frame().with_columns(model=pl.lit("otro-modelo"))
    with pytest.raises(ValueError, match="no son compatibles"):
        validate_index(index)


def test_bm25_matches_acronyms_and_hyphenated_terms() -> None:
    scores = bm25_scores(
        "Aplicaciones de seguridad con Man-in-the-Middle para casos de malware",
        Corpus.of(
            [
                "Detección de software malicioso",
                "Análisis de malware en aplicaciones IoT",
                "Man-in-the-Middle malware detection",
            ]
        ),
    )
    assert scores[2] > scores[1] > scores[0]


def test_lexical_variants_match_spanish_and_english_forms() -> None:
    titles = [
        "Clasificación automática de hipertrofia adenoidea en imágenes médicas",
        "A Hybrid Approach for Adenoid Hypertrophy Detection in X-Ray Images",
        *(f"Detección de aplicaciones con redes neuronales {index}" for index in range(40)),
    ]
    corpus = Corpus.of(titles)
    variants = lexical_variants("detección de adenoides", corpus)
    assert variants["adenoides"] == ["adenoidea", "adenoid"]
    scores = bm25_scores("detección de adenoides", corpus)
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
        prepare_index(index),
        "Aplicaciones de seguridad con Man-in-the-Middle para casos de malware",
        [1.0, 0.0],
    )
    assert ranked["title_id"].to_list()[0] == "exact"
    assert ranked["bm25_score"][0] > 0
    assert ranked["semantic_score"][0] < 1


def test_hybrid_ranking_uses_semantic_results_when_bm25_has_no_hits() -> None:
    index = prepare_index(index_frame())
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
    # La afinidad sigue al reranker dentro de la ventana y al ranking híbrido fuera.
    assert result["score"].to_list() == [0.9, 0.4, 0.01]


def test_apply_rerank_orders_by_relevance_even_if_unsorted() -> None:
    ranked = pl.DataFrame(
        {"title_id": ["a", "b"], "title": ["A", "B"], "rank": [1, 2], "score": [0.03, 0.02]}
    )
    assert apply_rerank(ranked, [(0, 0.2), (1, 0.7)])["title_id"].to_list() == ["b", "a"]


def test_recommend_follows_the_reranked_order() -> None:
    index = prepare_index(index_frame())
    ranked = rank_titles(index, [1.0, 0.0])
    # Por similitud, "Seguridad con malware" (Ciberseguridad) va primero; el reranker
    # prefiere "Robótica móvil" y su laboratorio debe pasar arriba.
    robotics = ranked["title_id"].to_list().index("c")
    window = [(robotics, 0.95)] + [
        (i, 0.5 - i / 100) for i in range(ranked.height) if i != robotics
    ]
    reranked = apply_rerank(ranked, window)
    labs = recommend(index, reranked, group_by="lab")
    assert labs["lab"].to_list()[0] == "Robótica"
    assert labs["affinity"].to_list()[0] == 100
    people = recommend(index, reranked, group_by="person")
    assert people["person"].to_list()[0] == "Asesor C"


def test_prepare_index_rejects_vectors_of_different_lengths() -> None:
    index = index_frame().with_columns(
        embedding=pl.Series([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    )
    with pytest.raises(ValueError, match="misma dimensión"):
        prepare_index(index)


def test_prepare_index_keeps_links_without_vectors() -> None:
    index = prepare_index(index_frame())
    # Los vínculos no cargan los vectores: estos viven una sola vez, normalizados.
    assert "embedding" not in index.links.columns
    assert index.links.height == 5
    assert index.unit_vectors.shape == (4, 2)
    assert index.unit_vectors.dtype.name == "float32"
    assert (abs((index.unit_vectors**2).sum(axis=1) - 1) < 1e-6).all()


def test_lexical_variants_need_the_same_beginning() -> None:
    corpus = Corpus.of(
        ["Alineación de ontologías", "Una ontologia de dominio", "Ontological engineering"]
        + ["Detection of fake news", "Deteccion de rostros"] * 3
    )
    variants = lexical_variants("odontologia y deteccions", corpus)
    # Parecida en letras, pero otra palabra: no es variante.
    assert "odontologia" not in variants
    assert set(variants["deteccions"]) >= {"detection", "deteccion"}
