"""El perfil de una persona asesora. Lo usan la página Asesores y el buscador."""

import plotly.express as px
import polars as pl
import streamlit as st

from common import (
    BLUE,
    CORAL,
    KIND_COLORS,
    LABELS,
    ORANGE,
    ROLE_COLORS,
    ROLES,
    TEAL,
    TEAL_SCALE,
    distinctive,
    every_year,
    explain,
    load,
    load_optional,
    show,
    sparkline_help,
    year_axis,
    year_range,
)

PURPLE = "#9b7fd9"
OPENALEX_URL = "https://openalex.org"
WORK_TYPES = {
    "article": "Artículo",
    "conference-paper": "Congreso",
    "book-chapter": "Capítulo de libro",
    "book": "Libro",
    "preprint": "Preprint",
    "review": "Revisión",
    "dataset": "Datos",
}
# Las tesis no se muestran, tampoco las de OpenAlex (ver openalex.HIDDEN_TYPES).
HIDDEN_WORK_TYPES = ["dissertation"]
# Caracteres que Markdown interpretaría dentro de un título.
MARKDOWN_SPECIAL = str.maketrans({c: f"\\{c}" for c in "\\`*_{}[]<>()#+-.!|"})
WORK_COLORS = {
    "Artículos": BLUE,
    "Congresos": CORAL,
    "Tesis dirigidas": TEAL,
    "Comités tutoriales": ORANGE,
    "Proyectos": PURPLE,
}


def catalog_lines(works: pl.DataFrame) -> list[str]:
    """Títulos del catálogo del CIC como lista Markdown, con año y tipo."""
    lines = []
    for work in works.iter_rows(named=True):
        parts = [f"**{work['title'].translate(MARKDOWN_SPECIAL)}**"]
        if work["year"]:
            parts.append(str(work["year"]))
        parts.append(work["kind"])
        lines.append("1. " + " · ".join(parts))
    return lines


def openalex_lines(works: pl.DataFrame) -> list[str]:
    """Trabajos de OpenAlex como lista Markdown, con año, tipo, citas y DOI."""
    lines = []
    for work in works.iter_rows(named=True):
        parts = [f"**{work['title'].translate(MARKDOWN_SPECIAL)}**"]
        if work["year"]:
            parts.append(str(work["year"]))
        parts.append(WORK_TYPES.get(work["type"], work["type"] or "Otro"))
        if work["venue"]:
            parts.append(f"*{work['venue'].translate(MARKDOWN_SPECIAL)}*")
        parts.append(f"{work['cited_by_count']:,} citas")
        if work["doi"]:
            parts.append(f"[DOI]({work['doi']})")
        if work["is_oa"]:
            parts.append(":material/lock_open: acceso abierto")
        lines.append("1. " + " · ".join(parts))
    return lines


def render_profile(person: str) -> None:
    """Cifras, tipos de trabajo, producción por año, temas y colaboración de una persona."""
    profile = load("people").filter(pl.col("person") == person).row(0, named=True)
    per_year = load("person_year").filter(pl.col("person") == person)
    sparkline = every_year(per_year.group_by("year").agg(pl.sum("outputs")), "outputs")

    st.caption(profile["lab"] or "Sin laboratorio en el CIC (participa como persona externa)")
    with st.container(horizontal=True, gap="small"):
        st.metric("Tesis como Director 1", profile["director_1"], border=True)
        st.metric("Tesis como Director 2", profile["director_2"], border=True)
        st.metric("Comités tutoriales", profile["committee"], border=True)
        st.metric(
            "Productos",
            profile["outputs"],
            f"{profile['first_year']}–{profile['last_year']}" if profile["first_year"] else None,
            delta_color="off",
            delta_arrow="off",
            border=True,
            chart_data=sparkline["outputs"].to_list() or None,
            chart_type="bar",
            help=sparkline_help(sparkline),
        )
        st.metric(
            "Proyectos",
            profile["projects"],
            f"{profile['projects_led']} como responsable",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )

    overview, by_year, topics, publications = st.tabs(
        ["Panorama", "Producción por año", "Temas y colaboración", "Publicaciones (OpenAlex)"]
    )
    with overview:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**¿A qué dedica su trabajo?**")
            work = pl.DataFrame(
                {
                    "what": list(WORK_COLORS),
                    "n": [
                        profile["articles"],
                        profile["conferences"],
                        profile["director_1"] + profile["director_2"],
                        profile["committee"],
                        profile["projects"],
                    ],
                }
            )
            fig = px.bar(
                work,
                x="n",
                y="what",
                orientation="h",
                color="what",
                color_discrete_map=WORK_COLORS,
                category_orders={"what": list(WORK_COLORS)},
                labels={"n": "Cantidad", "what": ""},
            )
            fig.update_layout(height=300, showlegend=False)
            show(fig)
        with right:
            st.markdown("**Tesis por nivel y rol**")
            mix = load("advisor_roles").filter(pl.col("person") == person)
            if mix.is_empty():
                st.caption("Sin tesis en el catálogo.")
            else:
                fig = px.bar(
                    mix,
                    x="level",
                    y="theses",
                    color="role",
                    color_discrete_map=ROLE_COLORS,
                    category_orders={"role": ROLES},
                    labels=LABELS,
                )
                fig.update_layout(height=300, legend={"orientation": "h", "y": -0.25})
                show(fig)
        render_catalog_titles(person)
    with by_year:
        if per_year.is_empty():
            st.caption("Sin artículos ni congresos con año en el catálogo.")
        else:
            first, last = year_range(per_year["year"], key=f"years-{person}")
            scoped = per_year.filter(pl.col("year").is_between(first, last))
            fig = px.bar(
                scoped.sort("year"),
                x="year",
                y="outputs",
                color="kind",
                color_discrete_map=KIND_COLORS,
                labels=LABELS,
            )
            fig.update_layout(height=320, legend={"orientation": "h", "y": 1.1})
            year_axis(fig, scoped["year"].unique().sort().to_list())
            show(fig)
        st.caption(
            "Las tesis no tienen año en el catálogo del CIC, así que no pueden ir en esta línea "
            "de tiempo; sus totales están en **Panorama**."
        )
    with topics:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Temas característicos**")
            counts = load("term_person").filter(pl.col("person") == person).select("term", "titles")
            terms = distinctive(counts, 12).sort("score")
            if terms.is_empty():
                st.caption("Sin títulos suficientes para calcular temas.")
            else:
                fig = px.bar(
                    terms,
                    x="titles",
                    y="term",
                    orientation="h",
                    color="lift",
                    color_continuous_scale=TEAL_SCALE,
                    hover_data={"lift": ":.1f", "score": False},
                    labels=LABELS,
                )
                fig.update_layout(height=340, coloraxis_showscale=False)
                show(fig)
        with right:
            st.markdown("**Con quién coincide en tesis**")
            pairs = load("person_pairs").filter(pl.col("person") == person).sort("theses")
            if pairs.is_empty():
                st.caption("Sin colaboraciones registradas.")
            else:
                fig = px.bar(
                    pairs,
                    x="theses",
                    y="other",
                    orientation="h",
                    color_discrete_sequence=[ORANGE],
                    labels=LABELS,
                )
                fig.update_layout(height=340)
                show(fig)
    with publications:
        render_openalex(person)
    explain(
        """
- **¿A qué dedica su trabajo?** compara artículos en revista, trabajos en congreso, tesis que
  dirige (como Director 1 o 2), comités tutoriales y proyectos.
- **Temas característicos**: términos de los títulos de sus tesis dirigidas, artículos,
  congresos y proyectos que esta persona usa más que el promedio del CIC.
- **Con quién coincide**: personas que participaron en las mismas tesis, como directores o
  en el comité. Útil para pensar en un posible codirector.
"""
    )


def render_catalog_titles(person: str) -> None:
    """Las 5 publicaciones más recientes del catálogo del CIC y, desplegables, hasta 30."""
    titles = load_optional("output_titles")
    works = (
        titles.filter(pl.col("person") == person).sort("year", descending=True, nulls_last=True)
        if titles is not None
        else None
    )
    if works is None or works.is_empty():
        return
    st.markdown("**Publicaciones en el catálogo del CIC**")
    st.markdown("\n".join(catalog_lines(works.head(5))))
    if works.height <= 5:
        return
    with st.expander(f"Ver las {works.height} publicaciones del catálogo"):
        st.markdown("\n".join(catalog_lines(works.head(30))))
        if works.height > 30:
            st.caption(f"Se muestran las 30 más recientes de {works.height}.")


def render_openalex(person: str) -> None:
    """Métricas, citas por año y trabajos más citados según OpenAlex."""
    authors = load_optional("openalex_authors")
    found = authors.filter(pl.col("person") == person) if authors is not None else None
    if found is None or found.is_empty():
        st.caption("Esta persona no tiene un perfil de OpenAlex vinculado.")
        return
    author = found.row(0, named=True)

    with st.container(horizontal=True, gap="small"):
        st.metric("Trabajos en OpenAlex", f"{author['works_count']:,}", border=True)
        st.metric("Citas", f"{author['cited_by_count']:,}", border=True)
        st.metric(
            "Índice h",
            author["h_index"],
            border=True,
            help="h trabajos con al menos h citas cada uno.",
        )
        st.metric(
            "Índice i10",
            author["i10_index"],
            border=True,
            help="Trabajos con al menos 10 citas.",
        )
    if author["topics"]:
        st.caption(f"Temas según OpenAlex: {author['topics']}")
    ids = [i for i in author["openalex_ids"].split("|") if i]
    if ids:
        st.link_button(
            "Ver perfil en OpenAlex",
            f"{OPENALEX_URL}/{ids[0]}",
            icon=":material/open_in_new:",
        )

    years = load_optional("openalex_years")
    if years is not None:
        cited = years.filter((pl.col("person") == person) & (pl.col("citations") > 0)).sort("year")
        if not cited.is_empty():
            fig = px.bar(
                cited,
                x="year",
                y="citations",
                color_discrete_sequence=[BLUE],
                labels={"year": "Año", "citations": "Citas recibidas"},
                title="Citas recibidas por año",
            )
            fig.update_layout(height=280)
            year_axis(fig, cited["year"].unique().sort().to_list())
            show(fig)

    works = load_optional("openalex_works")
    # Una instantánea anterior al filtro de openalex.py aún puede traer tesis.
    top = (
        works.filter((pl.col("person") == person) & ~pl.col("type").is_in(HIDDEN_WORK_TYPES)).sort(
            "cited_by_count", descending=True
        )
        if works is not None
        else None
    )
    if top is not None and not top.is_empty():
        st.markdown("**Trabajos más citados**")
        st.markdown("\n".join(openalex_lines(top.head(10))))
        if top.height > 10:
            with st.expander(f"Ver los {top.height} trabajos guardados"):
                st.markdown("\n".join(openalex_lines(top.slice(10))))
    st.caption(
        f"Fuente: [OpenAlex]({OPENALEX_URL}), datos de dominio público (CC0). OpenAlex cuenta "
        "todo lo que tiene indexado, no solo lo registrado en el sitio del CIC, así que sus "
        "cifras pueden diferir de las demás pestañas. Si un perfil está mal vinculado, avísanos."
    )


@st.dialog("Perfil", width="large")
def profile_dialog(person: str) -> None:
    st.markdown(f"### {person}")
    render_profile(person)
