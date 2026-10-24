import plotly.express as px
import polars as pl
import streamlit as st

from common import (
    LABELS,
    ORANGE,
    ROLE_COLORS,
    ROLES,
    TEAL_SCALE,
    distinctive,
    explain,
    footer,
    load,
    show,
)

people = load("people")
roles = load("advisor_roles")

st.title("Asesores")
st.markdown(
    "Las personas que dirigen tesis y participan en comités tutoriales del CIC. Se incluye a "
    "todo el personal investigador con laboratorio en el CIC y a personas externas que "
    "participaron en al menos tres tesis. Los nombres y cifras salen del sitio público del CIC."
)

# --- Ranking ----------------------------------------------------------------

st.subheader("¿Quién participa en más tesis?")
left, middle, right = st.columns([2, 3, 2])
levels = sorted(roles["level"].unique())
picked_levels = left.multiselect("Nivel", levels, default=levels)
rank_by = middle.segmented_control(
    "Ordenar por", ["Todos los roles", *ROLES], default="Todos los roles", required=True
)
top_n = right.slider("Cuántas personas", 5, 30, 12)

scoped = roles.filter(pl.col("level").is_in(picked_levels))
ranking = (
    scoped.group_by("person")
    .agg(
        rank=pl.sum("theses")
        if rank_by == "Todos los roles"
        else pl.col("theses").filter(pl.col("role") == rank_by).sum(),
        total=pl.sum("theses"),
    )
    .sort("rank", "total", descending=True)
    .head(top_n)
)
bars = (
    scoped.join(ranking, on="person")
    .group_by("person", "role", "rank", "total")
    .agg(pl.sum("theses"))
)
fig = px.bar(
    bars,
    x="theses",
    y="person",
    color="role",
    orientation="h",
    color_discrete_map=ROLE_COLORS,
    category_orders={"role": ROLES, "person": ranking["person"].to_list()},
    hover_data={"total": True, "rank": False},
    labels=LABELS,
)
fig.update_layout(height=26 * top_n + 120, legend={"orientation": "h", "y": 1.08})
show(fig)
explain(
    """
- **Director 1**: quien dirige la tesis. Determina el laboratorio de la tesis.
- **Director 2**: codirector o codirectora.
- **Comité tutorial**: quien revisa y evalúa la tesis sin dirigirla.

Cada persona cuenta **una vez por tesis** con su rol más fuerte (el comité de la página de
la tesis también lista a los directores), así que el largo de la barra es el número de
tesis distintas en que participó.
"""
)

# --- Profile ----------------------------------------------------------------

st.subheader("Perfil")
names = people.sort("person")["person"].to_list()
default = st.session_state.get("profile", people.sort("director_1", descending=True)["person"][0])
person = st.selectbox(
    "Busca a una persona",
    names,
    index=names.index(default) if default in names else 0,
    help="Escribe para filtrar la lista.",
)
st.session_state["profile"] = person
profile = people.filter(pl.col("person") == person).row(0, named=True)
years = (
    load("person_year")
    .filter((pl.col("person") == person) & pl.col("year").is_between(2005, 2025))
    .sort("year")
)

st.markdown(f"### {person}")
st.caption(profile["lab"] or "Sin laboratorio en el CIC (participa como persona externa)")
cols = st.columns(5)
cols[0].metric("Tesis como Director 1", profile["director_1"], border=True)
cols[1].metric("Tesis como Director 2", profile["director_2"], border=True)
cols[2].metric("Comités tutoriales", profile["committee"], border=True)
cols[3].metric(
    "Productos",
    profile["outputs"],
    f"{profile['first_year']}–{profile['last_year']}" if profile["first_year"] else None,
    delta_color="off",
    delta_arrow="off",
    border=True,
    chart_data=years["outputs"].to_list() or None,
    chart_type="bar",
)
cols[4].metric(
    "Proyectos",
    profile["projects"],
    f"{profile['projects_led']} como responsable",
    delta_color="off",
    delta_arrow="off",
    border=True,
)

left, middle, right = st.columns(3, gap="large")
with left:
    st.markdown("**Tesis por nivel y rol**")
    mix = roles.filter(pl.col("person") == person)
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
        fig.update_layout(height=320, legend={"orientation": "h", "y": -0.25})
        show(fig)
with middle:
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
        fig.update_layout(height=320, coloraxis_showscale=False)
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
        fig.update_layout(height=320)
        show(fig)
explain(
    """
- **Temas característicos**: términos de los títulos de sus tesis dirigidas, artículos,
  congresos y proyectos que esta persona usa más que el promedio del CIC.
- **Con quién coincide**: personas que participaron en las mismas tesis, como directores o
  en el comité. Útil para pensar en un posible codirector.
- La línea de **Productos** muestra sus artículos y congresos por año (2005–2025).
"""
)

# --- Publishing vs supervising ---------------------------------------------

st.subheader("Publicar vs. dirigir")
st.markdown(
    "Cada punto es una persona investigadora del CIC. A la derecha, quien más publica; "
    "arriba, quien más tesis dirige; el tamaño es el número de proyectos que lidera. "
    "Haz clic en un laboratorio de la leyenda para ocultarlo."
)
researchers = people.drop_nulls("lab")
fig = px.scatter(
    researchers,
    x="outputs",
    y="director_1",
    size="projects_led",
    size_max=35,
    color="lab",
    hover_name="person",
    hover_data={"lab": True, "outputs": True, "director_1": True, "projects_led": True},
    labels={**LABELS, "outputs": "Productos (artículos y congresos)"},
)
fig.update_traces(marker={"line": {"width": 0.5, "color": "white"}, "sizemin": 4})
fig.update_layout(height=560)
show(fig)
footer()
