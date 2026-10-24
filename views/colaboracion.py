import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from common import LABELS, explain, footer, load, show

pairs = load("lab_pairs")

st.title("Colaboración")
st.markdown(
    "La investigación rara vez se hace en un solo laboratorio. Aquí se ve qué laboratorios "
    "trabajan juntos y qué personas coinciden más en tesis."
)

# --- Heatmap ----------------------------------------------------------------

st.subheader("¿Qué laboratorios colaboran?")
link = st.segmented_control(
    "Tipo de vínculo",
    pairs["link"].unique(maintain_order=True).to_list(),
    default="Artículos y congresos",
    required=True,
)
counts = pairs.filter(pl.col("link") == link)
labs_here = sorted(set(counts["lab_a"]) | set(counts["lab_b"]))
grid = (
    pl.DataFrame({"lab_a": labs_here})
    .join(pl.DataFrame({"lab_b": labs_here}), how="cross")
    .join(counts, on=["lab_a", "lab_b"], how="left")
    .with_columns(pl.col("items").fill_null(0))
    .pivot(on="lab_b", index="lab_a", values="items")
    .sort("lab_a")
)
directed = link == "Codirección de tesis"
fig = px.imshow(
    grid.select(labs_here).to_numpy(),
    x=labs_here,
    y=grid["lab_a"].to_list(),
    color_continuous_scale="Oranges",
    labels={
        "x": "Laboratorio del Director 2" if directed else "Con",
        "y": "Laboratorio del Director 1" if directed else "Laboratorio",
        "color": "Compartidos",
    },
    aspect="auto",
)
fig.update_layout(height=34 * len(labs_here) + 220)
fig.update_xaxes(tickangle=-40)
show(fig)
explain(
    """
Cada celda cuenta los elementos que comparten dos laboratorios:

- **Artículos y congresos**: productos con autores de ambos laboratorios.
- **Proyectos**: proyectos registrados en ambos laboratorios.
- **Codirección de tesis**: tesis cuyo Director 1 es de la fila y cuyo Director 2 es de la
  columna. La diagonal cuenta codirecciones dentro de un mismo laboratorio.
"""
)

# --- Sankey -----------------------------------------------------------------

st.subheader("Flujo de codirección")
st.markdown(
    "A la izquierda, el laboratorio del **Director 1**; a la derecha, el del **Director 2**. "
    "El grosor de cada banda es el número de tesis. Pasa el cursor sobre una banda para ver "
    "el detalle."
)
codir = pairs.filter(pl.col("link") == "Codirección de tesis")
min_theses = st.slider("Mostrar solo flujos con al menos estas tesis", 1, 10, 2)
codir = codir.filter(pl.col("items") >= min_theses)
left_labs = sorted(codir["lab_a"].unique())
right_labs = sorted(codir["lab_b"].unique())
palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
colors = {
    lab: palette[i % len(palette)] for i, lab in enumerate(sorted(set(left_labs) | set(right_labs)))
}
fig = go.Figure(
    go.Sankey(
        node={
            "label": left_labs + right_labs,
            "color": [colors[lab] for lab in left_labs + right_labs],
            "pad": 12,
            "thickness": 14,
        },
        link={
            "source": [left_labs.index(a) for a in codir["lab_a"]],
            "target": [len(left_labs) + right_labs.index(b) for b in codir["lab_b"]],
            "value": codir["items"].to_list(),
            "color": [
                colors[a].replace("rgb", "rgba").replace(")", ",0.45)") for a in codir["lab_a"]
            ],
            "hovertemplate": "%{source.label} → %{target.label}<br>%{value} tesis<extra></extra>",
        },
    )
)
fig.update_layout(height=620)
show(fig)

# --- People pairs -----------------------------------------------------------

st.subheader("Duplas que más coinciden en tesis")
st.markdown(
    "Pares de personas que más veces participaron juntas en una misma tesis, como directores "
    "o en el comité tutorial."
)
duos = (
    load("person_pairs")
    .filter(pl.col("person") < pl.col("other"))
    .with_columns(duo=pl.col("person") + " · " + pl.col("other"))
    .sort("theses", descending=True)
    .head(15)
    .sort("theses")
)
fig = px.bar(
    duos,
    x="theses",
    y="duo",
    orientation="h",
    color="theses",
    color_continuous_scale="Oranges",
    labels={**LABELS, "duo": "Dupla", "theses": "Tesis en común"},
)
fig.update_layout(coloraxis_showscale=False, height=26 * duos.height + 80)
fig.update_yaxes(title=None)
show(fig)
footer()
