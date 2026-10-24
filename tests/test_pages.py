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
    "perfil",
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


def numbers(values: list | dict) -> list[float]:
    """Los valores de un eje: Plotly guarda los numéricos como arreglo binario en base64."""
    if isinstance(values, list):
        return values
    import base64

    import numpy as np

    return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]).tolist()


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
    app = open_page("perfil")
    assert not app.exception, app.exception
    assert app.selectbox(key="persona").value is None
    assert not profile_headers(app)


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize(("name", "shown"), [("Persona 3", True), ("Nadie", False)])
def test_advisor_profile_opens_from_the_url(name: str, shown: bool) -> None:
    app = open_page("perfil", persona=name)
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
    labs = set(common.load("people").drop_nulls("lab")["lab"])
    # Una serie por laboratorio, con su sigla en la leyenda: dos siglas iguales no se juntan.
    assert {trace["legendgroup"] for trace in scatter["data"]} == labs
    for trace in scatter["data"]:
        assert trace["name"] == common.lab_code(trace["legendgroup"])
    assert scatter["layout"]["legend"]["orientation"] == "h"


def render_profile_app(person: str) -> AppTest:
    script = f"""
import streamlit as st
from advisor import render_profile
render_profile({person!r})
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception, app.exception
    return app


@pytest.mark.usefixtures("tables")
def test_standing_places_and_highlights_the_person() -> None:
    import common

    researchers = common.load("people").drop_nulls("lab")
    me = researchers.filter(pl.col("person") == "Persona 1").row(0, named=True)
    place = researchers.filter(pl.col("director_1") > me["director_1"]).height + 1
    app = render_profile_app("Persona 1")
    card = next(m for m in app.metric if m.label == "Tesis como Director 1" and "#" in m.value)
    assert card.value == f"#{place} de {researchers.height}"
    stars = [
        trace
        for spec in chart_specs(app)
        for trace in spec["data"]
        if trace.get("marker", {}).get("symbol") == "star"
    ]
    assert [trace["name"] for trace in stars] == ["Persona 1"]


@pytest.mark.usefixtures("tables")
def test_standing_explains_why_externals_are_not_ranked() -> None:
    import common

    external = common.load("people").filter(pl.col("lab").is_null())["person"]
    assert not external.is_empty(), "el catálogo falso debe tener una persona externa"
    app = render_profile_app(external[0])
    assert any("participa como externa" in c.value for c in app.caption)
    # Solo las tarjetas de siempre: ni lugares en el CIC ni la gráfica de publicar vs. dirigir.
    assert len(app.metric) == 5
    assert not [spec for spec in chart_specs(app) if spec["data"][0]["type"] == "scatter"]


def profile_links(app: AppTest) -> list[str]:
    from urllib.parse import parse_qs

    queries = [parse_qs(link.proto.query_string) for link in app.get("page_link")]
    return [q["persona"][0] for q in queries if "persona" in q]


@pytest.mark.usefixtures("tables")
def test_lab_members_link_to_their_profile() -> None:
    import common

    app = open_page("laboratorios")
    lab = app.selectbox[0].value
    members = common.load("people").filter(pl.col("lab") == lab)["person"]
    assert sorted(profile_links(app)) == sorted(members)


@pytest.mark.usefixtures("tables")
def test_finder_dialog_links_to_the_profile_page() -> None:
    app = open_page("encuentra")
    app.text_area(key="idea").set_value("deep learning").run()
    opener = next(b for b in app.button if str(b.key).startswith("open-"))
    opener.click().run()
    assert profile_links(app) == [str(opener.key).removeprefix("open-")]


@pytest.mark.usefixtures("tables")
def test_profile_lists_where_the_person_publishes() -> None:
    import common

    outputs = common.load("people").filter(pl.col("person") == "Persona 1")["outputs"][0]
    venues = set(common.load("venues")["venue"])
    bars = [
        trace
        for spec in chart_specs(render_profile_app("Persona 1"))
        for trace in spec["data"]
        if trace["type"] == "bar" and set(trace["y"]) <= venues
    ]
    assert bars, "el perfil debe dibujar dónde publica"
    # Pocas revistas en el catálogo falso: caben todas, así que suman todos sus productos.
    assert sum(sum(numbers(trace["x"])) for trace in bars) == outputs
    # Una revista con artículos y congresos muestra los dos tipos, no solo el mayoritario.
    by_kind = (
        common.load("person_venues")
        .filter(pl.col("person") == "Persona 1")
        .group_by("kind")
        .agg(pl.sum("outputs"))
    )
    assert {trace["name"]: sum(numbers(trace["x"])) for trace in bars} == dict(by_kind.iter_rows())


def test_long_axis_labels_fit_in_two_lines() -> None:
    import common

    short = "Computación y Sistemas"
    assert common.wrap(short) == short
    long = "4th Workshop on Speech, Vision, and Language Technologies for Dravidian Languages"
    lines = common.wrap(long).split("<br>")
    assert len(lines) == 2
    assert all(len(line) <= common.WRAP_AT for line in lines)
    assert lines[-1].endswith("…")


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize("person", ["Persona 1", "Persona 2"])
def test_profile_shows_openalex_venues_when_linked(person: str) -> None:
    import common

    expected = common.load("openalex_venues").filter(pl.col("person") == person)
    charts = [
        trace
        for spec in chart_specs(render_profile_app(person))
        for trace in spec["data"]
        if trace.get("hovertemplate", "").startswith("Trabajos en OpenAlex")
    ]
    if expected.is_empty():
        assert not charts
    else:
        (bars,) = charts
        assert dict(zip(bars["y"], numbers(bars["x"]), strict=True)) == dict(
            expected.select("venue", "works").iter_rows()
        )


def test_year_span_joins_consecutive_years() -> None:
    from advisor import year_span

    assert year_span([]) == "sin año"
    assert year_span([2024]) == "2024"
    assert year_span([2019, 2021, 2022, 2023]) == "2019, 2021–2023"


@pytest.mark.usefixtures("tables")
def test_profile_lists_every_project() -> None:
    import common
    from advisor import MARKDOWN_SPECIAL

    titles = common.load("project_titles").filter(pl.col("person") == "Persona 1")["title"]
    assert not titles.is_empty(), "el catálogo falso debe darle proyectos a Persona 1"
    markdown = " ".join(m.value for m in render_profile_app("Persona 1").markdown)
    for title in titles:
        assert f"**{title.translate(MARKDOWN_SPECIAL)}**" in markdown


def test_project_lines_escape_markdown() -> None:
    from advisor import project_lines

    projects = pl.DataFrame(
        {
            "title": ["Un *título*"],
            "years": [[2024]],
            "role": ["[Responsable](https://ejemplo.test)"],
        }
    )
    (line,) = project_lines(projects)
    # Ni el título ni el rol pueden volverse formato o un enlace.
    assert "](" not in line
    assert line == r"- **Un \*título\*** · 2024 · \[Responsable\]\(https://ejemplo\.test\)"


def test_thesis_lines_name_advisors_and_count_the_rest() -> None:
    from advisor import thesis_lines

    theses = pl.DataFrame(
        {
            "title": ["Un *título*"],
            "level": ["Doctorado"],
            "director_1": ["Ana López"],
            "director_2": [None],
            "committee": [["Luis Pérez", "Eva [Ruiz]"]],
            "others": [1],
        }
    )
    (line,) = thesis_lines(theses)
    assert line == (
        r"- **Un \*título\*** · Doctorado · Dirige: Ana López · "
        r"Comité: Luis Pérez, Eva \[Ruiz\] · y 1 persona más"
    )


@pytest.mark.usefixtures("tables")
def test_profile_lists_theses_by_role() -> None:
    import common
    from advisor import MARKDOWN_SPECIAL

    theses = common.load("thesis_titles")
    directed = theses.filter(pl.col("director_1") == "Persona 1")
    committee = theses.filter(pl.col("committee").list.contains("Persona 1"))
    assert not directed.is_empty() and not committee.is_empty()
    app = render_profile_app("Persona 1")
    labels = [e.label for e in app.expander]
    assert f"Como Director 1 ({directed.height})" in labels
    assert f"En comité tutorial ({committee.height})" in labels
    markdown = " ".join(m.value for m in app.markdown)
    for title in directed["title"]:
        assert f"**{title.translate(MARKDOWN_SPECIAL)}**" in markdown


def assert_venue_chart(app: AppTest, venues: pl.DataFrame) -> None:
    """La gráfica de revistas de la página dibuja `venues` revista por revista, apilada."""
    import common

    (chart,) = [
        spec
        for spec in chart_specs(app)
        if spec["data"]
        and all(
            trace["type"] == "bar" and set(trace["y"]) <= set(venues["venue"])
            for trace in spec["data"]
        )
    ]
    # Apiladas: con barras lado a lado, una revista con los dos tipos se vería partida.
    assert chart["layout"]["barmode"] in {"stack", "relative"}
    drawn = {
        (venue, trace["name"]): value
        for trace in chart["data"]
        for venue, value in zip(trace["y"], numbers(trace["x"]), strict=True)
    }
    expected = {
        (venue, kind): outputs
        for venue, kind, outputs in venues.select("venue", "kind", "outputs").iter_rows()
    }
    assert {kind for _, kind in expected} == set(common.KIND_COLORS), (
        "el catálogo falso debe tener artículos y congresos"
    )
    # Pocas revistas en el catálogo falso: caben todas, así que se compara revista por
    # revista y tipo por tipo.
    assert drawn == expected


@pytest.mark.usefixtures("tables")
def test_lab_venues_stack_both_kinds() -> None:
    import common

    app = open_page("laboratorios")
    lab = app.selectbox[0].value
    assert_venue_chart(app, common.load("venues").filter(pl.col("lab") == lab))


@pytest.mark.usefixtures("tables")
def test_cic_venues_stack_both_kinds() -> None:
    import common

    # Por omisión, todos los laboratorios: las filas con laboratorio nulo.
    venues = common.load("venues").filter(pl.col("lab").is_null())
    assert_venue_chart(open_page("produccion"), venues)
