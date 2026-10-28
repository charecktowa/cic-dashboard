import polars as pl
import streamlit as st

from common import CURRENT_YEAR, every_year, footer, load, ratio, rising_terms

st.title("El CIC en datos")
st.markdown(
    "La investigación del **Centro de Investigación en Computación (CIC) del IPN** en datos: "
    "qué se investiga, en qué laboratorios, con quién y cómo ha cambiado con los años. Para "
    "estudiantes que buscan **tema de tesis, laboratorio o asesor**, y para quien tenga "
    "curiosidad por la computación que se hace en el CIC."
)

outputs = load("outputs_by_year").filter(pl.col("lab").is_null())
per_year = every_year(
    outputs.filter(pl.col("year").is_between(CURRENT_YEAR - 15, CURRENT_YEAR - 1))
    .group_by("year")
    .agg(pl.sum("outputs")),
    "outputs",
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

st.subheader("¿Por dónde empezar?")
cards = [
    (
        "views/encuentra.py",
        ":material/explore:",
        "Encuentra tu lab",
        "A partir de una idea de tesis, qué laboratorios y asesores trabajan temas parecidos.",
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
        "Cómo cambia cada tema con los años: qué temas crecen y cuáles van a la baja.",
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
english = ratio(recent_lang["english"].sum(), recent_lang["titles"].sum())
doctorate = ratio(labs["doctorate"].sum(), labs["theses"].sum())
top_work = load("openalex_works").sort("cited_by_count", descending=True).row(0, named=True)
# Cada dupla aparece dos veces (A con B y B con A): basta una.
pairs_people = load("person_pairs").filter(pl.col("person") < pl.col("other"))
top_duo = pairs_people.sort("theses", descending=True).row(0, named=True)
# `rising` va del cambio más negativo al más positivo: el primero es el que más cayó.
top_falling = rising.row(0, named=True)


def titles_before_and_now(term: str) -> str | None:
    """Los títulos con `term` en la ventana anterior y en la reciente, como "12 → 27"."""
    row = rising.filter(pl.col("term") == term)
    return f"{row['before'][0]} → {row['recent'][0]}" if row.height else None


# El contraste con lo que reemplaza a las redes neuronales a secas: deep learning y los LLM.
contrast = [
    f"{label} ({counts})"
    for label, term in (
        ("**deep learning** subió", "deep learning"),
        ("los **LLM** aparecieron", "large language"),
    )
    if (counts := titles_before_and_now(term))
]

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
    (
        ":material/format_quote:",
        f"El trabajo más citado de alguien del CIC es «{top_work['title']}» "
        f"({top_work['year']}), de **{top_work['person']}**: "
        f"**{top_work['cited_by_count']:,}** citas en OpenAlex.",
    ),
    (
        ":material/group:",
        f"**{top_duo['person']}** y **{top_duo['other']}** coinciden en "
        f"**{top_duo['theses']:,}** tesis: ninguna otra dupla del CIC se repite tanto.",
    ),
    (
        ":material/trending_down:",
        f"El tema que más cayó en los títulos de los últimos 4 años es **{top_falling['term']}** "
        f"({top_falling['before']} → {top_falling['recent']} títulos)"
        + (", mientras " + " y ".join(contrast) if contrast else "")
        + ".",
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
