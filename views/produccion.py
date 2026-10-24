import plotly.express as px
import polars as pl
import streamlit as st

from common import (
    ALL_LABS,
    BLUE,
    CORAL,
    CURRENT_YEAR,
    KIND_COLORS,
    LABELS,
    TEAL,
    explain,
    footer,
    lab_colors,
    load,
    number,
    ratio,
    show,
    show_venues,
    year_axis,
)

outputs = load("outputs_by_year")
lab_names = sorted(load("lab_stats")["lab"])

st.title("Producción")
st.markdown(
    "Los **productos** son artículos en revista y trabajos presentados en congresos. Aquí se "
    "ve cómo ha cambiado la producción del CIC, cómo creció cada laboratorio, sus proyectos y "
    "dónde publica."
)

# --- Over time --------------------------------------------------------------

st.subheader("Productos por año")
first_year = int(number(outputs["year"].min()))
left, middle, right = st.columns([2, 2, 1])
years = left.slider(
    "Años", first_year, CURRENT_YEAR, (max(first_year, CURRENT_YEAR - 20), CURRENT_YEAR)
)
kinds = sorted(outputs["kind"].unique())
picked_kinds = middle.multiselect("Tipo", kinds, default=kinds)
split = right.segmented_control(
    "Separar por", ["Tipo", "Laboratorio"], default="Tipo", required=True
)

scoped = outputs.filter(pl.col("year").is_between(*years) & pl.col("kind").is_in(picked_kinds))
if split == "Tipo":
    by_year = scoped.filter(pl.col("lab").is_null()).group_by("year", "kind").agg(pl.sum("outputs"))
    fig = px.line(
        by_year.sort("year"),
        x="year",
        y="outputs",
        color="kind",
        markers=True,
        color_discrete_map=KIND_COLORS,
        labels=LABELS,
    )
else:
    by_year = scoped.drop_nulls("lab").group_by("year", "lab").agg(pl.sum("outputs"))
    fig = px.area(
        by_year.sort("year", "lab"),
        x="year",
        y="outputs",
        color="lab",
        color_discrete_map=lab_colors(),
        labels=LABELS,
    )
year_axis(fig, scoped["year"].unique().sort().to_list())
show(fig)
st.caption(
    "Un producto con autores de dos laboratorios cuenta una vez para cada uno al separar por "
    "laboratorio. El año en curso es parcial."
)

# --- Race -------------------------------------------------------------------

st.subheader("Cómo creció cada laboratorio")
st.markdown(
    "Productos **acumulados** de cada laboratorio desde el año 2000. Presiona ▶ para ver la "
    "animación o arrastra el control de años."
)
race = (
    outputs.drop_nulls("lab")
    .filter(pl.col("year").is_between(2000, CURRENT_YEAR))
    .group_by("lab", "year")
    .agg(pl.sum("outputs"))
)
grid = (
    pl.DataFrame({"lab": lab_names})
    .join(pl.DataFrame({"year": range(2000, CURRENT_YEAR + 1)}), how="cross")
    .join(race.with_columns(pl.col("year").cast(pl.Int64)), on=["lab", "year"], how="left")
    .with_columns(pl.col("outputs").fill_null(0))
    .sort("lab", "year")
    .with_columns(total=pl.col("outputs").cum_sum().over("lab"))
)
order = grid.filter(pl.col("year") == CURRENT_YEAR).sort("total")["lab"].to_list()
fig = px.bar(
    grid,
    x="total",
    y="lab",
    orientation="h",
    animation_frame="year",
    color="lab",
    color_discrete_map=lab_colors(),
    range_x=[0, number(grid["total"].max()) * 1.05],
    category_orders={"lab": order},
    labels={**LABELS, "total": "Productos acumulados"},
)
fig.update_layout(showlegend=False, height=28 * len(lab_names) + 180)
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 400
show(fig)

# --- Projects ---------------------------------------------------------------

st.subheader("Proyectos de investigación")
by_scope, by_team = load("projects_by_scope"), load("projects_by_team")
first, last = int(number(by_scope["year"].min())), int(number(by_scope["year"].max()))
left, middle, right = st.columns([2, 2, 1])
project_lab = left.selectbox("Laboratorio", [ALL_LABS, *lab_names], key="project_lab")
project_years = middle.slider("Años", first, last, (first, last), key="project_years")
undated = right.toggle("Incluir proyectos sin año", value=True)


def scope(frame: pl.DataFrame) -> pl.DataFrame:
    lab_filter = (
        pl.col("lab").is_null() if project_lab == ALL_LABS else pl.col("lab") == project_lab
    )
    year_filter = pl.col("year").is_between(*project_years) | (pl.col("year").is_null() & undated)
    return frame.filter(lab_filter & year_filter)


scoped, team = scope(by_scope), scope(by_team)
total = scoped["projects"].sum()
several = scoped.filter(pl.col("scope") == "Varios laboratorios")["projects"].sum()
staffed = team.filter(pl.col("members") > 0)
average_team = ratio((staffed["members"] * staffed["projects"]).sum(), staffed["projects"].sum())

cols = st.columns(3)
cols[0].metric("Proyectos", f"{total:,}", border=True)
cols[1].metric("Con varios laboratorios", f"{ratio(several, total):.0%}", border=True)
cols[2].metric("Equipo promedio", f"{average_team:.1f} personas", border=True)

left, right = st.columns([2, 1])
with left:
    per_year = scoped.drop_nulls("year").group_by("year", "scope").agg(pl.sum("projects"))
    fig = px.bar(
        per_year.sort("year"),
        x="year",
        y="projects",
        color="scope",
        color_discrete_map={"Un laboratorio": TEAL, "Varios laboratorios": CORAL},
        labels=LABELS,
        title="Proyectos por año",
    )
    year_axis(fig, per_year["year"].unique().sort().to_list())
    show(fig)
with right:
    sizes = staffed.group_by("members").agg(pl.sum("projects")).sort("members")
    fig = px.bar(
        sizes,
        x="members",
        y="projects",
        color_discrete_sequence=[BLUE],
        labels=LABELS,
        title="Tamaño del equipo",
    )
    fig.update_xaxes(type="category")
    show(fig)

# --- Venues -----------------------------------------------------------------

st.subheader("¿Dónde publica el CIC?")
venues = load("venues")
left, middle, right = st.columns([2, 2, 1])
venue_lab = left.selectbox("Laboratorio", [ALL_LABS, *lab_names], key="venue_lab")
venue_kind = middle.segmented_control(
    "Tipo", ["Todos", *sorted(venues["kind"].unique())], default="Todos", required=True
)
venue_top = right.slider("Cuántas", 5, 40, 15, step=5)

scoped = venues.filter(
    pl.col("lab").is_null() if venue_lab == ALL_LABS else pl.col("lab") == venue_lab
)
if venue_kind != "Todos":
    scoped = scoped.filter(pl.col("kind") == venue_kind)
if not show_venues(scoped, venue_top):
    st.info("No hay productos con revista o congreso registrado para esta selección.")
with_venue = venues.filter(pl.col("lab").is_null())["outputs"].sum()
all_outputs = dict(load("totals").iter_rows())["Artículos y congresos"]
explain(
    f"""
- **Qué cuenta:** los artículos en revista y los trabajos en congreso del catálogo del CIC
  que tienen anotada su revista o congreso: {with_venue:,} de {all_outputs:,} productos.
- **Cada producto cuenta una vez** en el total del CIC, aunque lo firmen varias personas del
  CIC. En un laboratorio cuenta si el catálogo lo asocia con ese laboratorio, así que un
  producto de dos laboratorios aparece en ambos y la suma de los laboratorios pasa del total.
- **Una revista con varias formas de escribirse** (con o sin acentos, mayúsculas, "&" o
  "and") se cuenta como una sola y se muestra con su nombre más frecuente.
- **Todos los años juntos.** Pasa el cursor sobre una barra para ver el primer y el último
  año con un producto ahí. Con el tipo en **Todos**, cada barra se divide en artículos y
  congresos.

Si estás por publicar, esta lista es una buena pista de dónde suelen enviar sus trabajos en
tu laboratorio.
"""
)
footer()
