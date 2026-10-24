"""Cada página corre de principio a fin con la instantánea falsa."""

from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
PAGES = [
    "inicio",
    "encuentra",
    "laboratorios",
    "asesores",
    "temas",
    "colaboracion",
    "produccion",
    "acerca",
]


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    if page != "inicio":
        app.switch_page(f"views/{page}.py").run()
    return app


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page: str) -> None:
    app = open_page(page)
    assert not app.exception, app.exception


@pytest.mark.usefixtures("tables")
def test_finder_ranks_the_lab_that_uses_the_terms() -> None:
    app = open_page("encuentra")
    app.text_area(key="idea").set_value("análisis de sentimiento con deep learning").run()
    assert not app.exception, app.exception
    subheaders = [s.value for s in app.subheader]
    assert "Laboratorios afines" in subheaders


@pytest.mark.usefixtures("tables")
def test_finder_warns_when_nothing_matches() -> None:
    app = open_page("encuentra")
    app.text_area(key="idea").set_value("zzzz qqqq").run()
    assert not app.exception, app.exception
    assert app.warning


def test_match_puts_the_right_lab_first(tables: dict[str, pl.DataFrame]) -> None:
    import common

    recognized, labs, _ = common.match("deep learning y hate speech")
    assert "deep learning" in recognized
    assert labs["lab"][0] == "Procesamiento de Lenguaje Natural"


@pytest.mark.usefixtures("tables")
def test_finder_opens_a_profile_without_leaving_the_page() -> None:
    app = open_page("encuentra")
    app.text_area(key="idea").set_value("deep learning").run()
    opener = next(b for b in app.button if str(b.key).startswith("open-"))
    opener.click().run()
    assert not app.exception, app.exception
    assert "Laboratorios afines" in [s.value for s in app.subheader]


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize("person", ["Persona 1", "Persona 2"])
def test_profile_renders_with_and_without_openalex(person: str) -> None:
    script = f"""
import streamlit as st
from advisor import render_profile
render_profile({person!r})
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception, app.exception
    captions = " ".join(c.value for c in app.caption)
    linked = "no tiene un perfil de OpenAlex" not in captions
    assert linked == (person == "Persona 1")


def test_missing_optional_table_is_none() -> None:
    import common

    assert common.load_optional("no_existe") is None
