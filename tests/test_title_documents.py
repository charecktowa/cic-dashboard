"""El corpus semántico conserva evidencia pública y excluye personas estudiantes."""

from pathlib import Path

import polars as pl

from title_documents import collect_title_documents, normalize_title, title_id


def test_title_ids_normalize_case_and_whitespace_but_keep_accents() -> None:
    assert title_id("  Redes   Neuronales ") == title_id("redes neuronales")
    assert title_id("cómputo") != title_id("computo")
    assert normalize_title("  Qwen 3.7 ") == "qwen 3.7"


def test_theses_are_linked_to_directors_but_not_committee(
    tables: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    documents = collect_title_documents(_copy_tables(tmp_path))
    thesis = tables["thesis_titles"].row(0, named=True)
    rows = documents.filter(pl.col("title") == thesis["title"])
    source_theses = tables["thesis_titles"].filter(pl.col("title") == thesis["title"])
    # Cada director conserva su papel: el comité y los papeles cruzados no entran.
    expected = {
        (person, role)
        for row in source_theses.iter_rows(named=True)
        for role, person in (("Director 1", row["director_1"]), ("Director 2", row["director_2"]))
        if person
    }
    assert set(rows.drop_nulls("person").select("person", "role").iter_rows()) == expected
    assert rows["lab"].to_list() == [thesis["lab"]] * rows.height


def test_corpus_includes_projects_and_catalogue_outputs_but_not_openalex_theses(
    tables: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    documents = collect_title_documents(_copy_tables(tmp_path))
    assert {"Tesis", "Artículo", "Congreso", "Proyecto"} <= set(documents["source"])
    assert "Mi tesis doctoral" not in documents["title"].to_list()
    assert "openalex_works" not in documents.columns


def test_shared_titles_have_one_stable_id_and_keep_all_relationships(tmp_path: Path) -> None:
    _write_tables(
        tmp_path,
        thesis_titles=pl.DataFrame(
            {
                "title": ["Shared research title"],
                "level": ["Maestría"],
                "lab": ["Lab A"],
                "director_1": ["Advisor A"],
                "director_2": ["Advisor B"],
                "committee": [["Committee member"]],
                "others": [0],
            }
        ),
        output_titles=pl.DataFrame(
            {
                "person": ["Advisor A"],
                "title": ["Shared research title"],
                "year": [2024],
                "kind": ["Artículo"],
            }
        ),
        project_titles=pl.DataFrame(
            {
                "person": ["Advisor B"],
                "title": ["Shared research title"],
                "years": [[2023]],
                "role": ["Responsable"],
            }
        ),
        people=pl.DataFrame({"person": ["Advisor A", "Advisor B"], "lab": ["Lab A", "Lab B"]}),
    )

    documents = collect_title_documents(tmp_path)
    assert documents["title_id"].n_unique() == 1
    assert set(documents["person"]) == {"Advisor A", "Advisor B"}
    assert set(documents["source"]) == {"Tesis", "Artículo", "Proyecto"}
    assert "Committee member" not in documents["person"].to_list()


def _copy_tables(tmp_path: Path) -> Path:
    import os

    source = Path(os.environ["CIC_DATA_DIR"])
    for name in ("thesis_titles", "output_titles", "project_titles", "people"):
        (tmp_path / f"{name}.parquet").write_bytes((source / f"{name}.parquet").read_bytes())
    return tmp_path


def _write_tables(
    data_dir: Path,
    *,
    thesis_titles: pl.DataFrame,
    output_titles: pl.DataFrame,
    project_titles: pl.DataFrame,
    people: pl.DataFrame,
) -> None:
    for name, frame in (
        ("thesis_titles", thesis_titles),
        ("output_titles", output_titles),
        ("project_titles", project_titles),
        ("people", people),
    ):
        frame.write_parquet(data_dir / f"{name}.parquet")
