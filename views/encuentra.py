import plotly.express as px
import polars as pl
import streamlit as st

import text
from advisor import profile_dialog
from common import LABELS, TEAL_SCALE, explain, footer, lab_colors, load, match, show

EXAMPLES = {
    "Noticias falsas": "detección de noticias falsas en redes sociales con lenguaje natural",
    "Visión": "visión por computadora y redes neuronales convolucionales para imágenes médicas",
    "Robots": "robots móviles, control y navegación autónoma",
    "Ciberseguridad": "ciberseguridad, criptografía y detección de intrusiones",
    "Cuántica": "computación cuántica y algoritmos cuánticos",
    "Blockchain": "blockchain y contratos inteligentes",
    "Educación": "aprendizaje automático aplicado a la educación",
}


def use_example() -> None:
    picked = st.session_state.get("example")
    if picked:
        st.session_state["idea"] = EXAMPLES[picked]


st.title("Encuentra tu laboratorio y tu asesor")
st.markdown(
    "Escribe tu idea de tesis o lo que te gustaría investigar, en español o en inglés. "
    "El tablero busca esos términos en los títulos de **tesis, artículos, congresos y "
    "proyectos** del CIC y te dice qué laboratorios y qué personas han trabajado en temas "
    "parecidos."
)

st.pills(
    "¿Sin idea todavía? Prueba un ejemplo:",
    list(EXAMPLES),
    key="example",
    on_change=use_example,
)
idea = st.text_area(
    "Tu idea",
    key="idea",
    placeholder="Por ejemplo: detección de emociones en texto con deep learning",
    height=90,
)

if not idea.strip():
    st.info(
        "Escribe algunas palabras arriba o elige un ejemplo. Entre más específico, mejor: "
        "*“segmentación de imágenes satelitales”* da mejores pistas que *“inteligencia "
        "artificial”*.",
        icon=":material/edit:",
    )
    footer()
    st.stop()

recognized, labs, people = match(idea)
ignored = [t for t in text.terms(idea) if " " not in t and t not in recognized]

st.markdown("**Términos reconocidos en el catálogo**")
if recognized:
    term_df = dict(load("term_df").iter_rows())
    with st.container(horizontal=True, gap="small"):
        for term in recognized[:16]:
            st.badge(f"{term} · {term_df[term]:,} títulos", color="primary")
        for term in ignored[:8]:
            st.badge(f"{term} · sin títulos", color="gray")
else:
    st.warning(
        "Ningún término de tu idea aparece en al menos tres títulos del catálogo. Prueba con "
        "sinónimos, en inglés (muchos títulos lo están) o con palabras más generales.",
        icon=":material/search_off:",
    )
    footer()
    st.stop()

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Laboratorios afines")
    top_labs = labs.head(6).sort("affinity")
    fig = px.bar(
        top_labs,
        x="affinity",
        y="lab",
        orientation="h",
        color="affinity",
        color_continuous_scale=TEAL_SCALE,
        range_color=[0, 100],
        hover_data={"matched": True, "affinity": True},
        labels=LABELS,
    )
    fig.update_layout(coloraxis_showscale=False, height=60 * top_labs.height + 60)
    fig.update_xaxes(range=[0, 105])
    show(fig)
with right:
    st.subheader("Personas afines")
    top_people = people.head(8).sort("affinity")
    fig = px.bar(
        top_people,
        x="affinity",
        y="person",
        orientation="h",
        color="lab",
        color_discrete_map=lab_colors(),
        hover_data={"lab": True, "matched": True, "affinity": True},
        labels=LABELS,
        # Un color por laboratorio parte las barras en series; sin esto Plotly las
        # ordenaría por serie y no por afinidad.
        category_orders={"person": people.head(8)["person"].to_list()},
    )
    fig.update_layout(height=45 * top_people.height + 100, legend={"orientation": "h", "y": -0.2})
    fig.update_xaxes(range=[0, 105])
    show(fig)

st.subheader("Conoce a las primeras tres personas")
for column, row in zip(st.columns(3), people.head(3).iter_rows(named=True), strict=False):
    profile = load("people").filter(pl.col("person") == row["person"]).row(0, named=True)
    with column.container(border=True, height="stretch"):
        st.markdown(f"**{row['person']}**")
        st.caption(profile["lab"] or "Sin laboratorio en el CIC")
        st.markdown(
            f"{profile['director_1']} tesis como Director 1 · "
            f"{profile['outputs']} productos · {profile['projects_led']} proyectos liderados"
        )
        st.caption(f"En común contigo: {row['matched']}")
        if st.button(
            "Ver perfil completo",
            key=f"open-{row['person']}",
            icon=":material/open_in_full:",
        ):
            profile_dialog(row["person"])

st.subheader("¿Tu tema está creciendo?")
trend = (
    load("term_year")
    .filter(pl.col("term").is_in(recognized[:5]))
    .drop_nulls("year")
    .group_by("term", "year")
    .agg(pl.sum("titles"))
    .sort("year")
)
fig = px.area(
    trend.filter(pl.col("year") >= 2000),
    x="year",
    y="titles",
    color="term",
    labels=LABELS,
    category_orders={"term": recognized[:5]},
)
fig.update_layout(height=300)
show(fig)
st.caption(
    "Títulos de artículos, congresos y proyectos con cada término, por año (las tesis no "
    "tienen año en el catálogo). Si quieres más detalle, ve a la página **Temas**."
)

explain(
    """
**Qué se compara.** Tu texto se parte en palabras y pares de palabras (*“redes neuronales”*),
sin acentos ni palabras vacías como *de*, *para* o *the*. Solo cuentan los términos que
aparecen en al menos tres títulos del catálogo.

**Afinidad de un laboratorio.** Por cada término se toma la fracción de títulos del
laboratorio que lo usan y se multiplica por lo **raro** que es el término en todo el CIC
(un término que casi nadie usa pesa más que *“sistema”*). Se suman todos los términos y se
expresa como porcentaje del laboratorio con mayor puntaje.

**Afinidad de una persona.** Igual, pero con log(1 + títulos) en vez de la fracción, para
no favorecer solo a quien tiene cientos de títulos. Los títulos de una persona son las
tesis que dirige (Director 1 o 2), sus artículos y congresos, y sus proyectos.

**Limitaciones.** Solo se leen **títulos**, no resúmenes; y se buscan **palabras**, no
significados, así que *“aprendizaje profundo”* y *“deep learning”* son términos distintos:
escribe ambos si quieres cubrir los dos idiomas. Úsalo como punto de partida para
conversar con posibles asesores, no como una recomendación definitiva.
""",
    label="¿Cómo funciona la búsqueda?",
)
footer()
