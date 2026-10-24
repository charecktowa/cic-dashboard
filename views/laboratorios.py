import plotly.express as px
import polars as pl
import streamlit as st

from advisor import profile_link
from common import (
    BLUE,
    KIND_COLORS,
    LAB_METRICS,
    LABELS,
    TEAL,
    TEAL_SCALE,
    every_year,
    explain,
    fmt_share,
    footer,
    lab_colors,
    load,
    show,
    show_venues,
    sparkline_help,
    year_axis,
    year_range,
)

lab_stats = load("lab_stats")
outputs = load("outputs_by_year")
signature = load("signature")
people = load("people")

st.title("Laboratorios")
st.markdown(
    "El CIC tiene 15 laboratorios. Aquí puedes ordenarlos por distintas métricas, abrir el "
    "perfil de cada uno y compararlos entre sí. Cada laboratorio conserva su color en todas "
    "las gráficas."
)

# --- Ranking ----------------------------------------------------------------

st.subheader("Ranking")
metric = st.segmented_control(
    "Ordenar por",
    LAB_METRICS,
    default="theses",
    required=True,
    format_func=lambda metric: LABELS[metric],
)
ranked = lab_stats.sort(metric, descending=True, nulls_last=True)
fig = px.bar(
    ranked,
    x=metric,
    y="lab",
    orientation="h",
    color="lab",
    color_discrete_map=lab_colors(),
    # Un color por laboratorio parte las barras en series: sin esto Plotly las
    # ordenaría por serie y no por el ranking.
    category_orders={"lab": ranked["lab"].to_list()},
    hover_data={
        "lab": False,
        "researchers": True,
        "theses": True,
        "doctorate_share": ":.0%",
        "outputs": True,
        "recent_outputs": True,
        "projects": True,
    },
    labels=LABELS,
)
fig.update_layout(showlegend=False, height=28 * ranked.height + 80)
show(fig)
explain(
    """
- **Tesis**: una tesis pertenece al laboratorio de su **Director 1**. Si el Director 1 no
  tiene perfil en el CIC, la tesis no se asigna a ningún laboratorio.
- **Productos**: artículos en revista y trabajos en congreso. Un producto con autores de dos
  laboratorios cuenta para ambos.
- **Personas investigadoras**: quienes tienen ese laboratorio como adscripción principal en
  el sitio del CIC.
- **Productos por persona investigadora** ayuda a comparar laboratorios grandes con chicos.
"""
)

# --- Lab profile ------------------------------------------------------------

st.subheader("Perfil de un laboratorio")
lab = st.selectbox("Laboratorio", ranked["lab"].to_list(), help="Ordenados según el ranking.")
row = lab_stats.filter(pl.col("lab") == lab).row(0, named=True)
lab_years = every_year(
    outputs.filter(pl.col("lab") == lab).group_by("year").agg(pl.sum("outputs")), "outputs"
)

cols = st.columns(4)
cols[0].metric("Personas investigadoras", f"{row['researchers']:,}", border=True)
cols[1].metric(
    "Tesis",
    f"{row['theses']:,}",
    f"{fmt_share(row['doctorate_share'])} de doctorado",
    delta_color="off",
    delta_arrow="off",
    border=True,
)
cols[2].metric(
    "Productos",
    f"{row['outputs']:,}",
    f"{row['recent_outputs']:,} en los últimos 5 años",
    delta_color="off",
    delta_arrow="off",
    border=True,
    chart_data=lab_years["outputs"].to_list(),
    chart_type="area",
    help=sparkline_help(lab_years),
)
cols[3].metric("Proyectos", f"{row['projects']:,}", border=True)

tab_terms, tab_people, tab_years, tab_venues = st.tabs(
    ["Temas distintivos", "Personas", "Producción por año", "Dónde publica"]
)
with tab_terms:
    terms = signature.filter(pl.col("lab") == lab).sort("score")
    if terms.is_empty():
        st.info("Este laboratorio tiene muy pocos títulos para calcular temas distintivos.")
    else:
        fig = px.bar(
            terms,
            x="here",
            y="term",
            orientation="h",
            color="lift",
            color_continuous_scale=TEAL_SCALE,
            hover_data={"everywhere": True, "lift": ":.1f", "score": False},
            labels=LABELS,
        )
        fig.update_layout(height=22 * terms.height + 100)
        show(fig)
        st.caption(
            "Términos frecuentes en los títulos de este laboratorio y raros en el resto del "
            "CIC. El color indica cuántas veces más que el promedio aparece cada término."
        )
with tab_people:
    members = people.filter(pl.col("lab") == lab).sort("director_1")
    if members.is_empty():
        st.info("No hay personas investigadoras con este laboratorio como adscripción.")
    else:
        long = members.unpivot(
            on=["director_1", "outputs"], index="person", variable_name="what", value_name="n"
        ).with_columns(
            pl.col("what").replace({"director_1": "Tesis como Director 1", "outputs": "Productos"})
        )
        fig = px.bar(
            long,
            x="n",
            y="person",
            facet_col="what",
            orientation="h",
            color="what",
            color_discrete_sequence=[TEAL, BLUE],
            labels={**LABELS, "n": "", "what": ""},
        )
        fig.update_xaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(showlegend=False, height=26 * members.height + 100)
        show(fig)
        st.markdown("**Ver el perfil completo de:**")
        with st.container(horizontal=True, gap="small"):
            for name in members.sort("person")["person"]:
                profile_link(name)
with tab_years:
    per_kind = (
        outputs.filter(pl.col("lab") == lab)
        .group_by("year", "kind")
        .agg(pl.sum("outputs"))
        .sort("year")
    )
    if per_kind.is_empty():
        st.info("Este laboratorio no tiene productos con año en el catálogo.")
    else:
        first, last = year_range(per_kind["year"], key=f"years-{lab}")
        scoped = per_kind.filter(pl.col("year").is_between(first, last))
        fig = px.bar(
            scoped,
            x="year",
            y="outputs",
            color="kind",
            color_discrete_map=KIND_COLORS,
            labels=LABELS,
        )
        year_axis(fig, scoped["year"].unique().sort().to_list())
        show(fig)
with tab_venues:
    if not show_venues(load("venues").filter(pl.col("lab") == lab), 12):
        st.info("Este laboratorio no tiene productos con revista o congreso registrado.")

# --- Compare ----------------------------------------------------------------

st.subheader("Compara laboratorios")
st.markdown(
    "Elige hasta cuatro laboratorios. Cada eje va de 0 a 100, donde 100 es el laboratorio "
    "con el valor más alto del CIC en esa métrica."
)
picked = st.multiselect(
    "Laboratorios a comparar",
    ranked["lab"].to_list(),
    default=ranked["lab"].head(3).to_list(),
    max_selections=4,
)
if picked:
    radar_metrics = LAB_METRICS + ["doctorate_share"]
    scaled = lab_stats.with_columns(
        [(pl.col(m) / pl.col(m).max() * 100).round() for m in radar_metrics]
    )
    radar = (
        scaled.filter(pl.col("lab").is_in(picked))
        .unpivot(on=radar_metrics, index="lab", variable_name="metric", value_name="value")
        .with_columns(pl.col("metric").replace(LABELS))
    )
    fig = px.line_polar(
        radar,
        r="value",
        theta="metric",
        color="lab",
        color_discrete_map=lab_colors(),
        line_close=True,
        range_r=[0, 100],
        labels={"value": "Escala 0–100", "metric": "Métrica", **LABELS},
    )
    fig.update_traces(fill="toself", opacity=0.6)
    fig.update_layout(height=480, legend={"orientation": "h", "y": -0.1})
    show(fig)

# --- Topic map --------------------------------------------------------------

st.subheader("Mapa de temas")
st.markdown(
    "Cada laboratorio con sus seis temas más distintivos. **Haz clic en un laboratorio** para "
    "acercarte y en el centro para volver."
)
tree = signature.sort("score", descending=True).group_by("lab", maintain_order=True).head(6)
fig = px.sunburst(
    tree,
    path=[px.Constant("CIC"), "lab", "term"],
    values="here",
    color="lab",
    color_discrete_map=lab_colors(),
    hover_data={"here": True},
    labels=LABELS,
)
fig.update_traces(hovertemplate="<b>%{label}</b><br>Títulos: %{value}<extra></extra>")
fig.update_layout(height=650)
show(fig)
footer()
