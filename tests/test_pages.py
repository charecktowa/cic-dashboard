"""Cada página corre de principio a fin con la instantánea falsa."""

import datetime
import json
import math
from collections.abc import Iterator
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


def search_finder(app: AppTest, idea: str) -> AppTest:
    app.text_area(key="idea").set_value(idea)
    next(
        button for button in app.form_submit_button if button.label == "Buscar afinidades"
    ).click().run()
    return app


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page: str) -> None:
    app = open_page(page)
    assert not app.exception, app.exception


@pytest.mark.usefixtures("tables")
def test_finder_ranks_the_lab_that_uses_the_terms() -> None:
    app = search_finder(open_page("encuentra"), "análisis de sentimiento con deep learning")
    assert not app.exception, app.exception
    subheaders = [s.value for s in app.subheader]
    assert "Laboratorios afines" in subheaders
    # La barra más larga es la del laboratorio que match() pone primero.
    import common

    _, labs, _ = common.match("análisis de sentimiento con deep learning")
    bars = chart_specs(app)[0]["data"][0]
    affinities = numbers(bars["x"])
    assert bars["y"][affinities.index(max(affinities))] == labs["lab"][0]


@pytest.mark.usefixtures("tables")
def test_finder_warns_when_nothing_matches() -> None:
    app = search_finder(open_page("encuentra"), "zzzz qqqq")
    assert not app.exception, app.exception
    assert app.warning


def test_match_puts_the_right_lab_first(tables: dict[str, pl.DataFrame]) -> None:
    import common

    recognized, labs, _ = common.match("deep learning y hate speech")
    assert "deep learning" in recognized
    assert labs["lab"][0] == "Procesamiento de Lenguaje Natural"


@pytest.mark.usefixtures("tables")
def test_finder_opens_a_profile_without_leaving_the_page() -> None:
    app = search_finder(open_page("encuentra"), "deep learning")
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
    # Sin nadie elegido, ofrece las páginas que ayudan a encontrar a alguien.
    assert [link.proto.page for link in app.get("page_link")] == ["asesores", "encuentra"]


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
    app = search_finder(open_page("encuentra"), "deep learning")
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


def test_load_rereads_a_table_when_its_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import common

    monkeypatch.setattr(common, "DATA", tmp_path)
    table = tmp_path / "prueba.parquet"
    pl.DataFrame({"v": [1]}).write_parquet(table)
    assert common.load("prueba")["v"].to_list() == [1]
    # Un data/ nuevo copiado encima: otro contenido y otra fecha de modificación.
    pl.DataFrame({"v": [2]}).write_parquet(table)
    stat = table.stat()
    os.utime(table, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert common.load("prueba")["v"].to_list() == [2]


def expected_terms(where: str, app: AppTest) -> pl.DataFrame:
    """Los términos que la gráfica debe dibujar, con su lift."""
    import common

    if where == "perfil":
        counts = common.load("term_person").filter(pl.col("person") == PERSON_WITH_TERMS)
        return common.distinctive(counts.select("term", "titles"), 12)
    return common.load("signature").filter(pl.col("lab") == app.selectbox[0].value)


# En el catálogo falso, solo una persona externa usa términos más que el promedio.
PERSON_WITH_TERMS = "Externa 1"


@pytest.mark.usefixtures("tables")
@pytest.mark.parametrize("where", ["laboratorios", "perfil"])
def test_distinctive_terms_label_their_lift_on_a_fixed_scale(where: str) -> None:
    import common

    app = render_profile_app(PERSON_WITH_TERMS) if where == "perfil" else open_page(where)
    bars = [
        (spec, trace)
        for spec in chart_specs(app)
        for trace in spec["data"]
        if trace.get("texttemplate") == "×%{text:.1f}"
    ]
    assert len(bars) == 1, where
    spec, trace = bars[0]
    lifts = dict(expected_terms(where, app).select("term", "lift").iter_rows())
    assert sorted(trace["y"]) == sorted(lifts)
    # Cada barra dice su lift, y su color es el logaritmo del lift entre ×1 y ×LIFT_MAX.
    expected = [lifts[term] for term in trace["y"]]
    assert numbers(trace["text"]) == pytest.approx(expected)
    shades = [math.log10(min(max(lift, 1), common.LIFT_MAX)) for lift in expected]
    assert numbers(trace["marker"]["color"]) == pytest.approx(shades)
    axis = spec["layout"]["coloraxis"]
    assert (axis["cmin"], axis["cmax"]) == (0, math.log10(common.LIFT_MAX))
    assert axis["showscale"] is False


@pytest.mark.usefixtures("tables")
def test_lab_members_keep_one_color_in_both_charts() -> None:
    import common

    app = open_page("laboratorios")
    lab = app.selectbox[0].value
    members = common.load("people").filter(pl.col("lab") == lab)["person"].to_list()
    traces = [
        trace for spec in chart_specs(app) for trace in spec["data"] if trace.get("name") in members
    ]
    colors: dict[str, set[str]] = {}
    for trace in traces:
        colors.setdefault(trace["name"], set()).add(trace["marker"]["color"])
    # Cada persona: una barra de tesis y una de productos, del mismo color.
    assert sorted(colors) == sorted(members)
    assert all(len(drawn) == 1 for drawn in colors.values())
    # Los colores siguen el orden de las filas: las trazas van en ese orden.
    rows = list(dict.fromkeys(trace["name"] for trace in traces))
    assert {person: drawn.pop() for person, drawn in colors.items()} == common.person_colors(rows)


def test_person_colors_follow_the_order_and_repeat_after_the_palette() -> None:
    import common

    people = [f"Persona {i}" for i in range(12)]
    palette = common.PERSON_COLORS
    assert common.person_colors(people) == dict(zip(people, palette + palette[:4], strict=True))


@pytest.mark.usefixtures("tables")
def test_about_links_to_the_repository_and_its_issues() -> None:
    import common

    app = open_page("acerca")
    text = " ".join(m.value for m in app.markdown)
    assert f"]({common.REPO_URL})" in text
    assert f"]({common.REPO_URL}/issues)" in text


@pytest.mark.usefixtures("tables")
def test_theses_by_language_add_up_to_every_thesis() -> None:
    import common

    theses = common.load("language").filter(pl.col("source") == "Tesis")
    traces = [
        trace
        for spec in chart_specs(open_page("temas"))
        for trace in spec["data"]
        if trace.get("name") in common.LANGUAGE_COLORS
    ]
    assert sorted(trace["name"] for trace in traces) == ["Español", "Inglés"]
    total = 0.0
    for trace in traces:
        assert trace["marker"]["color"] == common.LANGUAGE_COLORS[trace["name"]]
        # La barra "Todas" junta todas las tesis con idioma, con o sin nivel.
        bars = dict(zip(trace["y"], numbers(trace["x"]), strict=True))
        total += bars["Todas"]
    assert total == theses["titles"].sum()


@pytest.mark.usefixtures("tables")
def test_cic_venues_explain_how_many_products_they_cover() -> None:
    import common

    with_venue = common.load("venues").filter(pl.col("lab").is_null())["outputs"].sum()
    total = dict(common.load("totals").iter_rows())["Artículos y congresos"]
    text = " ".join(m.value for m in open_page("produccion").markdown)
    assert f"{with_venue:,} de {total:,} productos" in text


def openalex_venue_page(lab: str | None = None) -> AppTest:
    """Producción con la fuente OpenAlex y, si se da, un laboratorio."""
    app = open_page("produccion")
    if lab is not None:
        app.selectbox(key="venue_lab").set_value(lab).run()
    app.segmented_control(key="venue_source").set_value("OpenAlex").run()
    assert not app.exception, app.exception
    return app


@pytest.mark.usefixtures("tables")
def test_cic_venues_can_come_from_openalex() -> None:
    import common

    bars = [
        trace
        for spec in chart_specs(openalex_venue_page())
        for trace in spec["data"]
        if trace.get("marker", {}).get("color") == common.PURPLE
    ]
    assert len(bars) == 1
    assert bars[0]["y"] == ["IEEE Access"]
    assert numbers(bars[0]["x"]) == [9]


@pytest.mark.usefixtures("tables")
def test_a_lab_without_openalex_profiles_says_so() -> None:
    import common

    linked = common.load("openalex_cic_venues")["lab"].drop_nulls().to_list()
    unlinked = next(lab for lab in common.load("lab_stats")["lab"] if lab not in linked)
    app = openalex_venue_page(unlinked)
    assert any("No hay trabajos de OpenAlex" in info.value for info in app.info)


@pytest.fixture
def semantic_finder(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Encuentra tu lab con llave e índice, y una API falsa que anota cada llamada."""
    import common
    import dashscope_client
    import title_documents

    documents = title_documents.collect_title_documents(common.DATA)
    width = 4
    index = documents.with_columns(
        embedding=pl.Series([[1.0, 0.0, 0.0, 0.0]] * documents.height, dtype=pl.List(pl.Float32)),
        model=pl.lit(dashscope_client.MODEL),
        dimension=pl.lit(dashscope_client.DIMENSION, dtype=pl.Int16),
        snapshot=pl.lit(common.snapshot()),
    )
    path = common.DATA / "title_embeddings.parquet"
    index.write_parquet(path)
    calls: list[str] = []

    def embed(query: str) -> list[float]:
        calls.append(f"embedding: {query}")
        return [1.0] + [0.0] * (width - 1)

    def rerank(query: str, titles: tuple[str, ...]) -> list[tuple[int, float]]:
        calls.append(f"rerank: {query}")
        return [(i, 1.0 - i / 1000) for i in range(len(titles))]

    monkeypatch.setenv("DASHSCOPE_API_KEY", "llave-de-prueba")
    monkeypatch.setattr(dashscope_client, "embed_query", embed)
    monkeypatch.setattr(dashscope_client, "rerank_titles", rerank)
    yield calls
    path.unlink()


@pytest.mark.usefixtures("tables")
def test_finder_uses_semantic_search_within_the_limits(semantic_finder: list[str]) -> None:
    app = search_finder(open_page("encuentra"), "visión por computadora con redes neuronales")
    assert not app.exception, app.exception
    assert any("Alibaba Cloud" in c.value for c in app.caption)
    assert semantic_finder


@pytest.mark.usefixtures("tables")
def test_finder_falls_back_to_terms_past_the_limit(
    semantic_finder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from search_limits import SearchLimiter

    monkeypatch.setattr(SearchLimiter, "allow", lambda self, client: "hora")
    app = search_finder(open_page("encuentra"), "visión por computadora con redes neuronales")
    assert not app.exception, app.exception
    assert any("límite de búsquedas por hora" in i.value for i in app.info)
    assert "Laboratorios afines" in [s.value for s in app.subheader]
    # Pasado el tope no sale ninguna llamada cobrada.
    assert semantic_finder == []


@pytest.mark.usefixtures("tables")
def test_opening_a_profile_does_not_repeat_the_search(
    semantic_finder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import semantic_search

    rankings: list[str] = []
    original = semantic_search.hybrid_rank_titles

    def counted(index: semantic_search.TitleIndex, query: str, vector: list[float]) -> pl.DataFrame:
        rankings.append(query)
        return original(index, query, vector)

    monkeypatch.setattr(semantic_search, "hybrid_rank_titles", counted)
    app = search_finder(open_page("encuentra"), "robots móviles que aprenden a caminar")
    opener = next(b for b in app.button if str(b.key).startswith("open-"))
    opener.click().run()
    assert not app.exception, app.exception
    assert rankings == ["robots móviles que aprenden a caminar"]


@pytest.mark.parametrize(
    ("n", "label"), [(0, "0 títulos"), (1, "1 título"), (1500, "1,500 títulos")]
)
def test_count_label_agrees_with_the_number(n: int, label: str) -> None:
    import common

    assert common.count_label(n, "título") == label


@pytest.mark.usefixtures("tables")
def test_finder_warns_about_the_provider_before_any_search(semantic_finder: list[str]) -> None:
    app = open_page("encuentra")
    assert any("se envían a Alibaba Cloud" in c.value for c in app.caption)
    # Abrir la página no manda nada.
    assert semantic_finder == []


@pytest.mark.usefixtures("tables")
def test_reruns_never_call_the_provider_again(semantic_finder: list[str]) -> None:
    import streamlit as st

    app = search_finder(open_page("encuentra"), "robots móviles que aprenden a caminar")
    sent = list(semantic_finder)
    # Aunque la caché expire, abrir un perfil reutiliza los resultados de la sesión.
    st.cache_data.clear()
    opener = next(b for b in app.button if str(b.key).startswith("open-"))
    opener.click().run()
    assert not app.exception, app.exception
    assert semantic_finder == sent


@pytest.mark.usefixtures("tables")
def test_an_index_from_another_snapshot_falls_back_to_terms(semantic_finder: list[str]) -> None:
    import datetime

    import common

    path = common.DATA / "title_embeddings.parquet"
    stale = pl.read_parquet(path).with_columns(snapshot=pl.lit(datetime.date(2000, 1, 1)))
    stale.write_parquet(path)
    app = search_finder(open_page("encuentra"), "visión por computadora con redes neuronales")
    assert not app.exception, app.exception
    assert any("otra instantánea" in i.value for i in app.info)
    assert "Laboratorios afines" in [s.value for s in app.subheader]
    assert semantic_finder == []


@pytest.mark.usefixtures("tables")
def test_a_broken_index_falls_back_to_terms(semantic_finder: list[str]) -> None:
    import common

    path = common.DATA / "title_embeddings.parquet"
    pl.read_parquet(path).drop("title").write_parquet(path)
    app = search_finder(open_page("encuentra"), "visión por computadora con redes neuronales")
    assert not app.exception, app.exception
    assert any("No se pudo completar la búsqueda semántica" in w.value for w in app.warning)
    assert semantic_finder == []


@pytest.mark.usefixtures("tables")
def test_finder_says_when_no_title_mentions_a_word(semantic_finder: list[str]) -> None:
    app = search_finder(open_page("encuentra"), "visión por computadora para odontología")
    assert not app.exception, app.exception
    notices = [i.value for i in app.info if "Ningún título del catálogo menciona" in i.value]
    # Con su acento, como la escribió el visitante.
    assert notices and "«odontología»" in notices[0]


@pytest.mark.usefixtures("tables")
def test_finder_lists_up_to_five_related_works(
    semantic_finder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import semantic_search

    # Lo que recommend() le entrega a la página, para compararlo con lo que se muestra.
    advisors: list[pl.DataFrame] = []
    original = semantic_search.recommend

    def recorded(*args: object, **kwargs: object) -> pl.DataFrame:
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("group_by") == "person":
            advisors.append(result)
        return result

    monkeypatch.setattr(semantic_search, "recommend", recorded)
    app = search_finder(open_page("encuentra"), "robots móviles que aprenden a caminar")
    assert not app.exception, app.exception
    shown = [e for e in app.expander if e.label.startswith("Obras afines")]
    (people,) = advisors
    cards = people.head(len(shown))
    assert shown and len(shown) == min(5, people.height)
    for expander, row in zip(shown, cards.iter_rows(named=True), strict=True):
        # Cada tarjeta, en orden, muestra las obras afines de su asesor, de la mejor a la peor.
        assert expander.label == f"Obras afines ({row['evidence_count']})"
        lines = expander.text[0].value.splitlines()
        assert lines == [f"• {work}" for work in row["related"]]
        assert 1 <= len(lines) <= 5


@pytest.fixture
def neighbor_chat(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Un chat falso que propone un área con trabajos en el catálogo falso y otra sin ellos."""
    import idea_assistant

    asked: list[str] = []

    def chat_json(instructions: str, prompt: str) -> dict:
        asked.append(prompt)
        return {
            "sugerencias": [
                {"tema": "Redes neuronales", "busqueda": "deep learning y redes neuronales"},
                {"tema": "Sin trabajos", "busqueda": "prótesis para odontología pediátrica"},
            ]
        }

    monkeypatch.setattr(idea_assistant, "chat_json", chat_json)
    return asked


@pytest.mark.usefixtures("tables")
def test_neighbor_areas_are_opt_in_and_launch_a_search(
    semantic_finder: list[str], neighbor_chat: list[str]
) -> None:
    import streamlit as st

    st.cache_data.clear()
    app = search_finder(open_page("encuentra"), "redes neuronales para odontología")
    assert not app.exception, app.exception
    # El chat no se llama hasta que el visitante lo pide.
    assert neighbor_chat == []
    ask = next(b for b in app.button if "Sugerir áreas cercanas" in b.label)
    ask.click().run()
    assert len(neighbor_chat) == 1 and "odontología" in neighbor_chat[0]
    # Solo la sugerencia con trabajos en el catálogo se ofrece.
    offered = [b for b in app.button if str(b.key).startswith("neighbor-")]
    assert [b.label for b in offered] == ["Redes neuronales"]
    sent = len(semantic_finder)
    offered[0].click().run()
    assert not app.exception, app.exception
    # La búsqueda nueva usa la sugerencia y sí llama a la API.
    assert app.session_state["idea"] == "deep learning y redes neuronales"
    assert len(semantic_finder) > sent


@pytest.mark.usefixtures("tables")
def test_no_neighbor_button_when_every_word_is_covered(
    semantic_finder: list[str], neighbor_chat: list[str]
) -> None:
    app = search_finder(open_page("encuentra"), "deep learning y redes neuronales")
    assert not [b for b in app.button if "Sugerir áreas cercanas" in b.label]


@pytest.mark.usefixtures("tables")
def test_neighbor_areas_count_toward_the_limits(
    semantic_finder: list[str], neighbor_chat: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from search_limits import SearchLimiter

    app = search_finder(open_page("encuentra"), "redes neuronales para odontología")
    # La búsqueda pasó; para cuando pide sugerencias, el sitio ya llegó a su tope.
    monkeypatch.setattr(SearchLimiter, "allow", lambda self, client: "hora")
    next(b for b in app.button if "Sugerir áreas cercanas" in b.label).click().run()
    assert not app.exception, app.exception
    assert neighbor_chat == []
    assert any("límite de búsquedas por hora" in i.value for i in app.info)


@pytest.fixture
def summary_chat(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Un chat falso que resume y elige la primera obra de la lista que recibe."""
    import idea_assistant

    asked: list[str] = []

    def chat_json(instructions: str, prompt: str) -> dict:
        asked.append(prompt)
        first = next(line[2:] for line in prompt.splitlines() if line.startswith("- "))
        return {"resumen": "Trabaja temas cercanos a tu idea.", "obra": first}

    monkeypatch.setattr(idea_assistant, "chat_json", chat_json)
    return asked


def open_first_profile(app: AppTest) -> AppTest:
    next(b for b in app.button if str(b.key).startswith("open-")).click().run()
    assert not app.exception, app.exception
    return app


@pytest.mark.usefixtures("tables")
def test_profile_from_the_finder_explains_the_fit(
    semantic_finder: list[str], summary_chat: list[str]
) -> None:
    import streamlit as st

    st.cache_data.clear()
    app = search_finder(open_page("encuentra"), "robots móviles que aprenden a caminar")
    assert summary_chat == []
    open_first_profile(app)
    assert any("Por qué podría encajar" in m.value for m in app.markdown)
    shown = [t.value for t in app.text]
    assert "Trabaja temas cercanos a tu idea." in shown
    # La obra seleccionada es una de las obras afines que se le enviaron al chat.
    ((prompt),) = summary_chat
    assert any(f"- {text}" in prompt for text in shown if " · " in text)
    assert any("generado por IA" in c.value for c in app.caption)
    # Volver a abrirlo no llama otra vez al chat.
    open_first_profile(app)
    assert len(summary_chat) == 1


@pytest.mark.usefixtures("tables")
def test_profile_from_the_term_search_has_no_ai_summary(summary_chat: list[str]) -> None:
    app = open_first_profile(search_finder(open_page("encuentra"), "deep learning"))
    assert not any("Por qué podría encajar" in m.value for m in app.markdown)
    assert summary_chat == []
