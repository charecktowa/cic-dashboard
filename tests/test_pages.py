"""Cada página corre de principio a fin con la instantánea falsa."""

import datetime
import json
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
PAGES_WITH_LAB_CHARTS = {"laboratorios", "asesores", "produccion"}


def open_page(page: str, **query: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.query_params.update(query)
    app.run()
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
    # La tesis de OpenAlex del catálogo falso (la más citada) nunca se muestra.
    assert "Mi tesis doctoral" not in " ".join(m.value for m in app.markdown)


def test_missing_optional_table_is_none() -> None:
    import common

    assert common.load_optional("no_existe") is None


@pytest.mark.usefixtures("tables")
def test_profile_lists_catalog_titles(catalog: dict[str, pl.DataFrame]) -> None:
    from advisor import MARKDOWN_SPECIAL

    newest = (
        catalog["output_authors"]
        .filter(pl.col("person_id") == 1)
        .join(catalog["outputs"], on="product_id")
        .filter(pl.col("year") <= datetime.date.today().year)
        .sort("year", descending=True)["title"][0]
    )
    script = """
import streamlit as st
from advisor import render_profile
render_profile('Persona 1')
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception, app.exception
    markdown = " ".join(m.value for m in app.markdown)
    assert "catálogo del CIC" in markdown
    assert newest.translate(MARKDOWN_SPECIAL) in markdown


@pytest.mark.usefixtures("tables")
def test_lab_colors_are_distinct() -> None:
    import common

    colors = common.lab_colors()
    assert set(colors) == set(common.load("lab_stats")["lab"])
    assert len(set(colors.values())) == len(colors)


def test_lab_colors_survive_a_new_lab() -> None:
    import common

    known = list(common.LAB_COLORS)
    before = common.assign_lab_colors(known[1:])
    after = common.assign_lab_colors([*known[1:], "Laboratorio Nuevo"])
    # Los demás conservan su color y el nuevo toma el que quedó libre.
    assert {lab: after[lab] for lab in before} == before
    assert after["Laboratorio Nuevo"] == common.LAB_COLORS[known[0]]


def test_year_axis_marks_every_year() -> None:
    import plotly.graph_objects as go

    import common

    fig = go.Figure(go.Bar(x=[2020, 2021], y=[1, 2]))
    common.year_axis(fig, [2020, 2021])
    assert fig.layout.xaxis.tickmode == "array"
    assert list(fig.layout.xaxis.tickvals) == [2020, 2021]
    assert fig.layout.xaxis.tickangle == -60

    # Nunca más de 20 marcas, el último año siempre marcado y sin dos marcas pegadas.
    for n in range(1, 200):
        years = list(range(1900, 1900 + n))
        fig = go.Figure()
        common.year_axis(fig, years)
        ticks = list(fig.layout.xaxis.tickvals)
        assert len(ticks) <= 20, n
        assert ticks[-1] == years[-1], n
        gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
        assert n <= 20 or min(gaps) > 1, n

    empty = go.Figure()
    common.year_axis(empty, [])
    assert empty.layout.xaxis.tickmode is None


def chart_specs(app: AppTest) -> list[dict]:
    return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]


@pytest.mark.usefixtures("tables")
def test_ranking_puts_the_leader_on_top() -> None:
    import common

    leader = common.load("lab_stats").sort("theses", descending=True)["lab"][0]
    ranking = chart_specs(open_page("laboratorios"))[0]
    # Plotly dibuja categoryarray de abajo hacia arriba; Plotly Express ya lo invierte
    # para que la primera categoría de category_orders quede arriba.
    assert ranking["layout"]["yaxis"]["categoryarray"][-1] == leader


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize("page", PAGES)
def test_every_lab_keeps_its_color(page: str) -> None:
    import common

    full = common.lab_colors()
    # Una gráfica estrecha nombra a los laboratorios por sus siglas, con el mismo color.
    colors = full | {common.lab_code(lab): color for lab, color in full.items()}
    traces = [trace for spec in chart_specs(open_page(page)) for trace in spec["data"]]
    by_lab = [trace for trace in traces if trace.get("name") in colors]
    # Sin esto, una página que dejara de dibujar sus gráficas por laboratorio pasaría.
    assert bool(by_lab) == (page in PAGES_WITH_LAB_CHARTS), page
    for trace in by_lab:
        drawn = trace.get("marker", {}).get("color") or trace.get("line", {}).get("color")
        assert drawn == colors[trace["name"]], (page, trace["name"])


def test_every_year_fills_the_gaps_with_zero() -> None:
    import common

    totals = pl.DataFrame({"year": [2020, 2023, 2021], "outputs": [4, 1, 2]})
    filled = common.every_year(totals, "outputs")
    assert filled["year"].to_list() == [2020, 2021, 2022, 2023]
    assert filled["outputs"].to_list() == [4, 2, 0, 1]
    assert common.every_year(totals.clear(), "outputs").is_empty()


@pytest.mark.usefixtures("tables")
def test_profile_sparkline_has_one_bar_per_year() -> None:
    import common

    years = common.load("person_year").filter(pl.col("person") == "Persona 9")["year"]
    first, last = int(common.number(years.min())), int(common.number(years.max()))
    span = last - first + 1
    assert years.n_unique() < span, "el catálogo falso debe tener un año sin productos"
    script = """
import streamlit as st
from advisor import render_profile
render_profile('Persona 9')
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    card = next(m for m in app.metric if m.label == "Productos")
    assert len(card.proto.chart_data) == span
    assert f"de {first} a {last}" in card.proto.help


def test_translucent_keeps_the_color() -> None:
    import common

    assert common.translucent("#2a9d8f", 0.45) == "rgba(42, 157, 143, 0.45)"


@pytest.mark.usefixtures("tables")
def test_codirection_flow_uses_lab_colors() -> None:
    import common

    colors = common.lab_colors()
    sankey = next(
        trace
        for spec in chart_specs(open_page("colaboracion"))
        for trace in spec["data"]
        if trace["type"] == "sankey"
    )
    nodes = sankey["node"]
    assert nodes["customdata"], "el catálogo falso debe tener codirecciones"
    assert nodes["color"] == [colors[lab] for lab in nodes["customdata"]]


def profile_headers(app: AppTest) -> list[str]:
    return [m.value for m in app.markdown if m.value.startswith("### ")]


@pytest.mark.usefixtures("tables")
def test_advisor_search_starts_empty() -> None:
    app = open_page("asesores")
    assert not app.exception, app.exception
    assert app.selectbox(key="persona").value is None
    assert not profile_headers(app)


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize(("name", "shown"), [("Persona 3", True), ("Nadie", False)])
def test_advisor_profile_opens_from_the_url(name: str, shown: bool) -> None:
    app = open_page("asesores", persona=name)
    assert not app.exception, app.exception
    assert profile_headers(app) == ([f"### {name}"] if shown else [])


@pytest.mark.usefixtures("tables")
def test_publishing_vs_supervising_fits_a_phone() -> None:
    import common

    scatter = next(
        spec
        for spec in chart_specs(open_page("asesores"))
        if spec["data"] and spec["data"][0]["type"] == "scatter"
    )
    codes = {common.lab_code(lab) for lab in common.lab_colors()}
    assert {trace["name"] for trace in scatter["data"]} <= codes
    assert scatter["layout"]["legend"]["orientation"] == "h"
