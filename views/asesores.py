import plotly.express as px
import polars as pl
import streamlit as st

from advisor import render_publishing_vs_supervising
from common import (
    LABELS,
    ROLE_COLORS,
    ROLES,
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
    "participaron en al menos tres tesis. Los nombres y cifras salen del sitio público del CIC. "
    "Aquí se compara a todas; para conocer a una sola, abre su perfil."
)

st.page_link(
    "views/perfil.py",
    label="Ver el perfil de una persona",
    icon=":material/person_search:",
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

# --- Publishing vs supervising ---------------------------------------------

st.subheader("Publicar vs. dirigir")
st.markdown(
    "Cada punto es una persona investigadora del CIC. A la derecha, quien más publica; "
    "arriba, quien más tesis dirige; el tamaño es el número de proyectos que lidera. "
    "Haz clic en las siglas de un laboratorio, debajo de la gráfica, para ocultarlo."
)
render_publishing_vs_supervising()
footer()
