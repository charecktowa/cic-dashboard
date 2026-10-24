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
    explain,
    load,
    show,
)

PURPLE = "#9b7fd9"
WORK_COLORS = {
    "Artículos": BLUE,
    "Congresos": CORAL,
    "Tesis dirigidas": TEAL,
    "Comités tutoriales": ORANGE,
    "Proyectos": PURPLE,
}


def render_profile(person: str) -> None:
    """Cifras, tipos de trabajo, producción por año, temas y colaboración de una persona."""
    profile = load("people").filter(pl.col("person") == person).row(0, named=True)
    per_year = load("person_year").filter(pl.col("person") == person)
    sparkline = (
        per_year.filter(pl.col("year").is_between(2005, 2025))
        .group_by("year")
        .agg(pl.sum("outputs"))
        .sort("year")["outputs"]
        .to_list()
    )

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
            chart_data=sparkline or None,
            chart_type="bar",
        )
        st.metric(
            "Proyectos",
            profile["projects"],
            f"{profile['projects_led']} como responsable",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )

    overview, by_year, topics = st.tabs(["Panorama", "Producción por año", "Temas y colaboración"])
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
    with by_year:
        if per_year.is_empty():
            st.caption("Sin artículos ni congresos con año en el catálogo.")
        else:
            fig = px.bar(
                per_year.sort("year"),
                x="year",
                y="outputs",
                color="kind",
                color_discrete_map=KIND_COLORS,
                labels=LABELS,
            )
            fig.update_layout(height=320, legend={"orientation": "h", "y": 1.1})
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


@st.dialog("Perfil", width="large")
def profile_dialog(person: str) -> None:
    st.markdown(f"### {person}")
    render_profile(person)
