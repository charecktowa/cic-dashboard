import plotly.express as px
import polars as pl
import streamlit as st

import text
from common import (
    CORAL,
    CURRENT_YEAR,
    LABELS,
    TEAL,
    explain,
    footer,
    load,
    number,
    ratio,
    rising_terms,
    show,
)

term_year, term_lab = load("term_year"), load("term_lab")
SUGGESTIONS = {
    "IA clásica vs. moderna": "redes neuronales, deep learning, fuzzy, algoritmos geneticos",
    "Lenguaje": "nlp, natural language, lenguaje natural, sentiment",
    "Seguridad": "seguridad, criptografia, blockchain, malware",
    "Hardware": "fpga, embedded, sensores, iot",
}


def use_suggestion() -> None:
    picked = st.session_state.get("suggestion")
    if picked:
        st.session_state["terms"] = SUGGESTIONS[picked]


st.title("Temas")
st.markdown(
    "Todos los títulos del CIC (tesis, artículos, congresos y proyectos) partidos en palabras "
    "y pares de palabras. Con eso basta para seguir un tema en el tiempo, sin ningún modelo "
    "de inteligencia artificial."
)

# --- Follow a term ----------------------------------------------------------

st.subheader("Sigue un tema")
if "terms" not in st.session_state:
    st.session_state["terms"] = "deep learning, redes neuronales, blockchain, fuzzy"
st.pills("Ideas para comparar:", list(SUGGESTIONS), key="suggestion", on_change=use_suggestion)
left, right = st.columns([3, 2])
raw = left.text_input("Términos separados por comas, en español o inglés", key="terms")
sources = sorted(term_year["source"].unique())
picked = right.multiselect("Fuentes", sources, default=sources)
wanted = list(dict.fromkeys(" ".join(text.words(t)) for t in raw.split(",") if t.strip()))

hits = term_year.filter(pl.col("term").is_in(wanted) & pl.col("source").is_in(picked))
if hits.is_empty():
    st.warning(
        "Ningún título tiene esos términos. Prueba otra ortografía o una sola palabra.",
        icon=":material/search_off:",
    )
else:
    dated = hits.drop_nulls("year").group_by("term", "year").agg(pl.sum("titles"))
    if not dated.is_empty():
        # Los años sin coincidencias cuentan como cero para que las líneas no salten.
        all_years = pl.DataFrame(
            {"year": range(int(number(dated["year"].min())), int(number(dated["year"].max())) + 1)},
            schema={"year": pl.Int32},
        )
        dated = (
            dated.select("term")
            .unique()
            .join(all_years, how="cross")
            .join(dated, on=["term", "year"], how="left")
            .with_columns(pl.col("titles").fill_null(0))
        )
    by_lab = (
        term_lab.filter(pl.col("term").is_in(wanted) & pl.col("source").is_in(picked))
        .group_by("term", "lab")
        .agg(pl.sum("titles"))
    )
    left, right = st.columns([3, 2])
    with left:
        fig = px.line(
            dated.sort("year"),
            x="year",
            y="titles",
            color="term",
            markers=True,
            labels=LABELS,
            category_orders={"term": wanted},
            title="Títulos por año",
        )
        show(fig)
    with right:
        fig = px.bar(
            by_lab,
            x="titles",
            y="lab",
            color="term",
            orientation="h",
            labels=LABELS,
            category_orders={"term": wanted},
            title="Por laboratorio",
        )
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(showlegend=False)
        show(fig)
explain(
    """
Cada título se pasa a minúsculas, sin acentos, y se quitan palabras vacías (*de*, *para*,
*the*, *of*…). Quedan palabras sueltas y pares de palabras vecinas, como *“deep learning”* o
*“redes neuronales”*. Un término solo se publica si aparece en **al menos tres títulos**.

Las tesis no tienen año en el catálogo, así que solo aparecen en la gráfica por laboratorio.
"""
)

# --- Rising and fading ------------------------------------------------------

st.subheader("¿Qué crece y qué se apaga?")
st.markdown(
    "Compara qué tanto aparece cada término en los títulos recientes contra la ventana "
    "anterior del mismo tamaño. Copia un término en el buscador de arriba para ver su curva."
)
left, right, note = st.columns([2, 2, 1])
window = left.slider("Ventana (años)", 2, 8, 4)
min_titles = right.slider("Mínimo de títulos", 3, 30, 8)
note.markdown(
    f"Ventana reciente: **{CURRENT_YEAR - window + 1}–{CURRENT_YEAR}**  \n"
    f"Anterior: **{CURRENT_YEAR - 2 * window + 1}–{CURRENT_YEAR - window}**"
)
rising = rising_terms(window, min_titles)
shown = (
    pl.concat([rising.head(12), rising.tail(12)])
    .unique("term")
    .with_columns(
        direction=pl.when(pl.col("change") > 0).then(pl.lit("Crece")).otherwise(pl.lit("Baja"))
    )
)
fig = px.bar(
    shown,
    x="change",
    y="term",
    color="direction",
    orientation="h",
    color_discrete_map={"Crece": TEAL, "Baja": CORAL},
    hover_data={"change": ":+.2f", "recent": True, "before": True, "direction": False},
    labels={
        **LABELS,
        "change": "Cambio en la proporción de títulos (puntos porcentuales)",
        "recent": "Títulos recientes",
        "before": "Títulos anteriores",
    },
)
fig.update_yaxes(categoryorder="total ascending")
fig.update_layout(height=18 * shown.height + 140, legend={"orientation": "h", "y": 1.05})
show(fig)
explain(
    """
Para cada término se calcula qué porcentaje de los títulos fechados (artículos, congresos y
proyectos) lo menciona en la ventana reciente, y se le resta ese porcentaje en la ventana
anterior. Un **+0.5** significa que medio punto porcentual más de los títulos lo usan ahora.

El año en curso es parcial, y el catálogo puede tardar en registrar lo más reciente.
"""
)

# --- Language ---------------------------------------------------------------

st.subheader("¿Español o inglés?")
language = load("language")
theses_lang = language.filter(pl.col("source") == "Tesis")
thesis_share = ratio(theses_lang["english"].sum(), theses_lang["titles"].sum())
by_year = (
    language.drop_nulls("year")
    .filter(pl.col("year").is_between(CURRENT_YEAR - 25, CURRENT_YEAR) & (pl.col("titles") >= 5))
    .with_columns(share=pl.col("english") / pl.col("titles"))
    .sort("year")
)
left, right = st.columns([1, 5])
left.metric("Tesis con título en inglés", f"{thesis_share:.0%}", border=True)
with right:
    fig = px.line(
        by_year,
        x="year",
        y="share",
        color="source",
        markers=True,
        line_shape="spline",
        hover_data={"titles": True},
        labels={**LABELS, "share": "Títulos en inglés"},
    )
    fig.update_yaxes(tickformat=".0%")
    show(fig)
explain(
    """
El idioma de un título se adivina por sus palabras cortas más comunes (*de*, *la*, *para*
contra *the*, *of*, *for*). Los títulos donde empatan no se cuentan. Las tesis no tienen año,
por eso aparecen como una sola proporción.
"""
)
footer()
