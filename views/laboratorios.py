import plotly.express as px
import polars as pl
import streamlit as st

from common import (
    BLUE,
    KIND_COLORS,
    LAB_METRICS,
    LABELS,
    TEAL,
    TEAL_SCALE,
    explain,
    fmt_share,
    footer,
    load,
    show,
)

lab_stats = load("lab_stats")
outputs = load("outputs_by_year")
signature = load("signature")
people = load("people")

st.title("Laboratorios")
st.markdown(
    "El CIC tiene 15 laboratorios. Aquí puedes ordenarlos por distintas métricas, abrir el "
    "perfil de cada uno y compararlos entre sí."
)

# --- Ranking ----------------------------------------------------------------

st.subheader("Ranking")
metric = st.segmented_control(
    "Ordenar por",
    LAB_METRICS,
    default="theses",
    required=True,
    format_func=LABELS.get,
)
ranked = lab_stats.sort(metric, descending=True, nulls_last=True)
fig = px.bar(
    ranked,
    x=metric,
    y="lab",
    orientation="h",
    color=metric,
    color_continuous_scale=TEAL_SCALE,
    hover_data={
        "researchers": True,
        "theses": True,
        "doctorate_share": ":.0%",
        "outputs": True,
        "recent_outputs": True,
        "projects": True,
    },
    labels=LABELS,
)
fig.update_layout(coloraxis_showscale=False, height=28 * ranked.height + 80)
fig.update_yaxes(categoryorder="total ascending")
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
lab_years = (
    outputs.filter((pl.col("lab") == lab) & pl.col("year").is_between(2005, 2025))
    .group_by("year")
    .agg(pl.sum("outputs"))
    .sort("year")
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
        st.caption("Abre la página **Asesores** para ver el perfil completo de cada persona.")
with tab_years:
    per_kind = (
        outputs.filter(pl.col("lab") == lab)
        .group_by("year", "kind")
        .agg(pl.sum("outputs"))
        .sort("year")
    )
    fig = px.bar(
        per_kind, x="year", y="outputs", color="kind", color_discrete_map=KIND_COLORS, labels=LABELS
    )
    show(fig)
with tab_venues:
    top_venues = (
        load("venues")
        .filter(pl.col("lab") == lab)
        .group_by("venue")
        .agg(pl.sum("outputs"), kind=pl.col("kind").sort_by("outputs").last())
        .sort("outputs", descending=True)
        .head(12)
        .sort("outputs")
    )
    fig = px.bar(
        top_venues,
        x="outputs",
        y="venue",
        orientation="h",
        color="kind",
        color_discrete_map=KIND_COLORS,
        labels=LABELS,
    )
    fig.update_layout(height=24 * top_venues.height + 100)
    show(fig)

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
    color_discrete_sequence=px.colors.qualitative.Set3,
    hover_data={"here": True},
    labels=LABELS,
)
fig.update_traces(hovertemplate="<b>%{label}</b><br>Títulos: %{value}<extra></extra>")
fig.update_layout(height=650)
show(fig)
footer()
