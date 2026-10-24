import polars as pl
import streamlit as st

from common import CURRENT_YEAR, footer, load, rising_terms

st.title("El CIC en datos")
st.markdown(
    "Un mapa interactivo de la investigación del **Centro de Investigación en Computación "
    "(CIC) del IPN**: qué se investiga, en qué laboratorios, con quién y cómo ha cambiado con "
    "los años. Está pensado para estudiantes que buscan **tema de tesis, laboratorio o "
    "asesor**, y para cualquiera con curiosidad por la computación que se hace en el CIC."
)

outputs = load("outputs_by_year").filter(pl.col("lab").is_null())
per_year = (
    outputs.filter(pl.col("year").is_between(CURRENT_YEAR - 15, CURRENT_YEAR - 1))
    .group_by("year")
    .agg(pl.sum("outputs"))
    .sort("year")
)
# Una fila que se parte sola: cinco tarjetas en escritorio, dos por fila en un celular.
with st.container(horizontal=True, gap="small"):
    for metric, value in load("totals").iter_rows():
        sparkline = per_year["outputs"].to_list() if metric == "Artículos y congresos" else None
        st.metric(
            metric,
            f"{value:,}",
            border=True,
            chart_data=sparkline,
            chart_type="area",
            help="La línea muestra los productos por año en los últimos 15 años."
            if sparkline
            else None,
        )

st.subheader("¿Por dónde empiezo?")
cards = [
    (
        "views/encuentra.py",
        ":material/explore:",
        "Encuentra tu lab",
        "Escribe tu idea de tesis y descubre qué laboratorios y asesores trabajan en temas "
        "parecidos.",
    ),
    (
        "views/laboratorios.py",
        ":material/science:",
        "Laboratorios",
        "Compara los 15 laboratorios: tesis, artículos, proyectos y los temas que los distinguen.",
    ),
    (
        "views/asesores.py",
        ":material/school:",
        "Asesores",
        "Quién dirige más tesis, en qué temas trabaja cada persona y con quién colabora.",
    ),
    (
        "views/temas.py",
        ":material/trending_up:",
        "Temas",
        "Sigue un tema en el tiempo y mira qué está creciendo y qué se está apagando.",
    ),
]
for column, (page, icon, title, blurb) in zip(st.columns(4), cards, strict=True):
    with column.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        st.caption(blurb)
        st.page_link(page, label="Abrir", icon=icon)

st.subheader("¿Sabías que…?")
labs = load("lab_stats")
top_lab = labs.sort("theses", descending=True).row(0, named=True)
people = load("people")
top_person = people.sort("director_1", descending=True).row(0, named=True)
rising = rising_terms(4, 8).filter(pl.col("term").str.contains(" "))
top_rising = rising.tail(3)["term"].reverse().to_list()
pairs = load("lab_pairs").filter(pl.col("link") == "Artículos y congresos")
top_pair = pairs.sort("items", descending=True).row(0, named=True)
recent_lang = load("language").filter(
    pl.col("year").is_between(CURRENT_YEAR - 5, CURRENT_YEAR)
    & pl.col("source").is_in(["Artículo", "Congreso"])
)
english = recent_lang["english"].sum() / max(recent_lang["titles"].sum(), 1)
doctorate = labs["doctorate"].sum() / max(labs["theses"].sum(), 1)

facts = [
    (
        ":material/emoji_events:",
        f"**{top_lab['lab']}** es el laboratorio con más tesis dirigidas: "
        f"**{top_lab['theses']:,}**.",
    ),
    (
        ":material/trending_up:",
        "Los temas que más crecieron en los títulos de los últimos 4 años son "
        + ", ".join(f"**{t}**" for t in top_rising)
        + ".",
    ),
    (
        ":material/handshake:",
        f"Los laboratorios que más artículos comparten son **{top_pair['lab_a']}** y "
        f"**{top_pair['lab_b']}** ({top_pair['items']:,}).",
    ),
    (
        ":material/translate:",
        f"El **{english:.0%}** de los artículos y congresos de los últimos 5 años tiene título "
        "en inglés.",
    ),
    (
        ":material/school:",
        f"**{top_person['person']}** es Director 1 de **{top_person['director_1']:,}** tesis, "
        "más que nadie en el catálogo.",
    ),
    (
        ":material/workspace_premium:",
        f"Una de cada {round(1 / doctorate) if doctorate else '—'} tesis con laboratorio "
        f"asignado es de doctorado (**{doctorate:.0%}**).",
    ),
]
for row in range(0, len(facts), 3):
    for column, (icon, fact) in zip(st.columns(3), facts[row : row + 3], strict=True):
        with column.container(border=True, height="stretch"):
            st.markdown(f"{icon} {fact}")

with st.expander("Cómo usar las gráficas", icon=":material/touch_app:"):
    st.markdown(
        """
- **Pasa el cursor** (o toca, en el celular) sobre una barra o un punto para ver sus números.
- **Haz clic en la leyenda** para ocultar una serie; **doble clic** para dejar solo esa.
- Usa los botones **+** y **−** de la esquina de cada gráfica para acercarte; la **casita** la
  regresa a su tamaño.
- Cada sección tiene un desplegable **¿Cómo se calcula?** que explica de dónde sale cada número.
"""
    )

footer()
