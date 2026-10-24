"""Carga de datos, etiquetas y utilidades que comparten todas las páginas."""

import datetime
import math
import os
import textwrap
from pathlib import Path
from typing import SupportsFloat, cast

import polars as pl
import streamlit as st

import text

# Las pruebas apuntan a una instantánea falsa con CIC_DATA_DIR.
DATA = Path(os.environ.get("CIC_DATA_DIR", Path(__file__).parent / "data"))
CURRENT_YEAR = datetime.date.today().year
ROLES = ["Director 1", "Director 2", "Comité tutorial"]
# Tonos intermedios: se leen igual sobre el fondo claro y sobre el oscuro.
TEAL, BLUE, ORANGE, CORAL = "#2a9d8f", "#4a90d9", "#f4a261", "#e76f51"
ROLE_COLORS = {"Director 1": TEAL, "Director 2": BLUE, "Comité tutorial": ORANGE}
KIND_COLORS = {"Artículo": BLUE, "Congreso": CORAL}
# Escalas sin extremos casi blancos ni casi negros, que desaparecerían en un modo u otro.
TEAL_SCALE = ["#7ccfc0", "#1f8a7e"]
ORANGE_SCALE = ["#f6b17a", "#c2410c"]
# Un color fijo por laboratorio del CIC. Tonos intermedios y separados entre sí (ΔE ≥ 21)
# para que se distingan sobre el fondo claro y el oscuro y entre pantallas sin calibración.
# Van por nombre y no por posición: si el catálogo agrega o renombra un laboratorio, los
# demás conservan el suyo.
LAB_COLORS = {
    "Ciberseguridad": "#2a9d8f",
    "Ciencia de Datos y Tecnología de Software": "#4a90d9",
    "Ciencias Cognitivas Computacionales": "#e76f51",
    "Ciencias Matemáticas y Computacionales": "#9b7fd9",
    "Ciencias de la Información Cuántica": "#f4a261",
    "Cómputo Inteligente": "#e15f99",
    "Inteligencia Artificial": "#8cd17d",
    "Microtecnología y Sistemas Embebidos": "#9c755f",
    "Procesamiento Digital de Señales": "#edc948",
    "Procesamiento Inteligente de Información Geoespacial": "#6e899c",
    "Procesamiento de Lenguaje Natural": "#ff9da7",
    "Redes y Ciencia de Datos": "#b07aa1",
    "Robótica y Mecatrónica": "#b68100",
    "Simulación y Modelado": "#c77dff",
    "Sistemas Inteligentes para la Automatización": "#c46a1b",
}
ALL_LABS = "Todos los laboratorios"
# Marcas en un eje de años y años visibles por omisión: caben en un celular.
YEAR_TICKS = 20
AUTHOR = "Andrea Gutiérrez"
AUTHOR_URL = "https://github.com/charecktowa"
# El aviso del sitio del CIC pide citar la fuente completa y su dirección electrónica.
SOURCE = "Centro de Investigación en Computación (CIC) del Instituto Politécnico Nacional"
SOURCE_URL = "https://www.cic.ipn.mx"
# El botón de PNG es inofensivo; la selección por caja o lazo no aporta aquí.
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

LABELS = {
    "lab": "Laboratorio",
    "year": "Año",
    "outputs": "Productos",
    "kind": "Tipo",
    "theses": "Tesis",
    "researchers": "Personas investigadoras",
    "doctorate": "Tesis de doctorado",
    "doctorate_share": "Proporción de doctorado",
    "recent_outputs": "Productos, últimos 5 años",
    "projects": "Proyectos",
    "projects_led": "Proyectos que lidera",
    "outputs_per_researcher": "Productos por persona investigadora",
    "term": "Término",
    "titles": "Títulos",
    "person": "Persona",
    "other": "Colega",
    "role": "Rol",
    "level": "Nivel",
    "venue": "Revista o congreso",
    "source": "Fuente",
    "scope": "Alcance",
    "members": "Personas en el proyecto",
    "here": "Títulos en este laboratorio",
    "everywhere": "Títulos en todo el catálogo",
    "lift": "Veces más que el promedio",
    "first": "Primer año",
    "last": "Último año",
    "total": "Total de tesis",
    "items": "Elementos compartidos",
    "affinity": "Afinidad (%)",
    "matched": "Términos en común",
    "director_1": "Tesis como Director 1",
    "score": "Puntaje",
}
# Los ejes de categorías se explican solos; su etiqueta solo se usa en los tooltips.
UNTITLED_AXES = {"Laboratorio", "Término", "Persona", "Colega", "Revista o congreso", "Año"}
# Siglas para ejes estrechos, como el mapa de calor en un celular.
LAB_CODE_OVERRIDES = {"Ciberseguridad": "Ciber", "Ciencias de la Información Cuántica": "CIQ"}
LAB_CODE_SKIP = {"de", "del", "la", "las", "el", "los", "en", "para", "a"}
# Un nombre más largo se parte en dos líneas para no robarle ancho a las barras.
WRAP_AT = 26
# Alto mínimo por barra cuando hay etiquetas de dos líneas, para que no se encimen.
WRAPPED_ROW = 42
LAB_METRICS = [
    "theses",
    "outputs",
    "recent_outputs",
    "projects",
    "researchers",
    "outputs_per_researcher",
]


@st.cache_data
def load(name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / f"{name}.parquet")


def load_optional(name: str) -> pl.DataFrame | None:
    """Como load(), pero None si el archivo no existe (por ejemplo, sin datos de OpenAlex)."""
    return load(name) if (DATA / f"{name}.parquet").exists() else None


def snapshot() -> datetime.date:
    return load("meta")["snapshot"][0]


def assign_lab_colors(labs: list[str]) -> dict[str, str]:
    """El color de LAB_COLORS para cada laboratorio conocido.

    Uno nuevo toma, en orden alfabético, un color que no use ningún otro laboratorio
    presente; los colores solo se repiten si hay más laboratorios que colores.
    """
    colors = {lab: LAB_COLORS[lab] for lab in labs if lab in LAB_COLORS}
    palette = list(LAB_COLORS.values())
    free = [c for c in palette if c not in colors.values()] or palette
    for i, lab in enumerate(sorted(set(labs) - set(colors))):
        colors[lab] = free[i % len(free)]
    return colors


@st.cache_data
def lab_colors() -> dict[str, str]:
    """Un color estable por laboratorio: el mismo en todas las gráficas de la app."""
    return assign_lab_colors(load("lab_stats")["lab"].to_list())


def translucent(color: str, alpha: float) -> str:
    """Un color "#rrggbb" como rgba, por ejemplo para las bandas de un Sankey."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def lab_code(lab: str) -> str:
    """Siglas de un laboratorio: "Robótica y Mecatrónica" → "RyM"."""
    if lab in LAB_CODE_OVERRIDES:
        return LAB_CODE_OVERRIDES[lab]
    words = [w for w in lab.split() if w.lower() not in LAB_CODE_SKIP]
    return "".join(w if w.lower() in {"y", "e"} else w[0].upper() for w in words)


def wrap(label: str) -> str:
    return "<br>".join(textwrap.wrap(label, WRAP_AT)) if len(label) > WRAP_AT else label


def show(fig, **kwargs):
    """Dibuja una figura de Plotly con el estilo común del tablero."""
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend_title_text="",
        hoverlabel={"align": "left"},
        # Sin arrastre, deslizar el dedo sobre una gráfica mueve la página en vez de
        # hacer zoom; los botones + y − de la barra siguen acercando.
        dragmode=False,
    )
    for axis in [fig.layout.xaxis, fig.layout.yaxis]:
        if axis.title.text in UNTITLED_AXES:
            axis.title.text = None
    # En barras horizontales, los nombres largos van en dos líneas; el tooltip
    # sigue mostrando el nombre completo.
    categories = {
        label
        for trace in fig.data
        if trace.type == "bar" and trace.orientation == "h" and trace.y is not None
        for label in trace.y
        if isinstance(label, str)
    }
    if any(len(label) > WRAP_AT for label in categories):
        fig.update_yaxes(tickvals=list(categories), ticktext=[wrap(c) for c in categories])
        fig.update_layout(height=max(fig.layout.height or 0, WRAPPED_ROW * len(categories) + 100))
    return st.plotly_chart(fig, config=PLOT_CONFIG, width="stretch", **kwargs)


def year_axis(fig, years: list[int]) -> None:
    """Una marca legible por año, sin encimarlas en un celular.

    Con hasta YEAR_TICKS años se etiqueta cada barra; con más, se salta años de forma
    uniforme sin pasar de YEAR_TICKS marcas, y el último año siempre lleva la suya.
    """
    if not years:
        return
    # Los saltos cubren de la primera a la penúltima marca; la última es el año final.
    step = 1 if len(years) <= YEAR_TICKS else math.ceil((len(years) - 1) / (YEAR_TICKS - 1))
    ticks = years[::step]
    if ticks[-1] != years[-1]:
        # A medio salto o menos de la anterior, las dos etiquetas se encimarían.
        if len(years) - 1 - years.index(ticks[-1]) <= step / 2:
            ticks.pop()
        ticks.append(years[-1])
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        tickangle=-60,
        tickfont_size=9,
        automargin=True,
    )


def year_range(years: pl.Series, key: str) -> tuple[int, int]:
    """Deslizador de años sobre `years`, de inicio en los últimos YEAR_TICKS años.

    Con un solo año no hay nada que elegir y no se dibuja.
    """
    first, last = int(number(years.min())), int(number(years.max()))
    if first < last:
        control, _ = st.columns([3, 2])
        first, last = control.slider(
            "Años", first, last, (max(first, last - YEAR_TICKS + 1), last), key=key
        )
    return first, last


def every_year(totals: pl.DataFrame, column: str) -> pl.DataFrame:
    """Los totales de `column` por año, con los años sin nada en cero.

    Una gráfica sin eje, como la miniatura de una tarjeta, pegaría dos años separados por
    un hueco y parecerían seguidos.
    """
    if totals.is_empty():
        return totals
    first, last = int(number(totals["year"].min())), int(number(totals["year"].max()))
    years = pl.DataFrame({"year": range(first, last + 1)}, schema={"year": totals.schema["year"]})
    return (
        years.join(totals.select("year", column), on="year", how="left")
        .with_columns(pl.col(column).fill_null(0))
        .sort("year")
    )


def sparkline_help(per_year: pl.DataFrame) -> str | None:
    """Qué años cubre la miniatura de una tarjeta, que no tiene eje."""
    if per_year.is_empty():
        return None
    first, last = int(number(per_year["year"].min())), int(number(per_year["year"].max()))
    return f"Un valor por año, de {first} a {last}. Los años sin productos cuentan como cero."


def explain(markdown: str, label: str = "¿Cómo se calcula?") -> None:
    with st.expander(label, icon=":material/help:"):
        st.markdown(markdown)


def footer() -> None:
    st.divider()
    st.markdown(f"Hecho con ❤️ por **{AUTHOR}** · [@charecktowa]({AUTHOR_URL})")
    st.caption(
        f"Fuente: {SOURCE}, [{SOURCE_URL.removeprefix('https://')}]({SOURCE_URL}), "
        f"consultado el {snapshot():%d/%m/%Y}. Proyecto independiente y sin fines de lucro, "
        "sin relación oficial con el IPN ni el CIC."
    )


def idf() -> pl.DataFrame:
    """Qué tan informativo es cada término: los raros pesan más que los comunes."""
    titles = load("meta")["titles"][0]
    return load("term_df").select("term", "titles", idf=(pl.lit(titles) / pl.col("titles")).log())


def match(idea: str) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    """Términos reconocidos de una idea, y laboratorios y personas afines.

    Un laboratorio suma, por cada término, la fracción de sus títulos que lo
    usan multiplicada por su rareza. Una persona suma log(1 + títulos) por la
    rareza, para no premiar solo a quien tiene muchos títulos.
    """
    weights = idf().filter(pl.col("term").is_in(text.terms(idea)))
    recognized = weights.sort("titles", descending=True)["term"].to_list()

    labs = (
        load("term_lab")
        .group_by("term", "lab")
        .agg(pl.sum("titles"))
        .join(weights.select("term", "idf"), on="term")
        .join(load("lab_titles").rename({"titles": "lab_titles"}), on="lab")
        .group_by("lab")
        .agg(
            score=(pl.col("titles") / pl.col("lab_titles") * pl.col("idf")).sum(),
            matched=pl.col("term").sort_by("titles", descending=True).head(4).str.join(", "),
        )
    )
    people = (
        load("term_person")
        .join(weights.select("term", "idf"), on="term")
        .group_by("person")
        .agg(
            score=(pl.col("titles").log1p() * pl.col("idf")).sum(),
            matched=pl.col("term").sort_by("titles", descending=True).head(4).str.join(", "),
        )
        .join(load("people").select("person", "lab"), on="person")
    )

    def scale(frame: pl.DataFrame) -> pl.DataFrame:
        top = frame["score"].max() or 1
        return frame.with_columns(affinity=(pl.col("score") / top * 100).round()).sort(
            "score", descending=True
        )

    return recognized, scale(labs), scale(people)


@st.cache_data
def rising_terms(window: int, min_titles: int) -> pl.DataFrame:
    """Cambio en la proporción de títulos con cada término entre dos ventanas."""
    period = (
        pl.when(pl.col("year") > CURRENT_YEAR - window)
        .then(pl.lit("recent"))
        .when(pl.col("year") > CURRENT_YEAR - 2 * window)
        .then(pl.lit("before"))
    )
    sizes = dict(
        load("doc_year")
        .with_columns(period=period)
        .drop_nulls("period")
        .group_by("period")
        .agg(pl.sum("titles"))
        .iter_rows()
    )
    return (
        load("term_year")
        .with_columns(period=period)
        .drop_nulls("period")
        .group_by("term")
        .agg(
            recent=pl.col("titles").filter(pl.col("period") == "recent").sum(),
            before=pl.col("titles").filter(pl.col("period") == "before").sum(),
        )
        .filter((pl.col("recent") + pl.col("before")) >= min_titles)
        .with_columns(
            change=(
                pl.col("recent") / sizes.get("recent", 1)
                - pl.col("before") / sizes.get("before", 1)
            )
            * 100
        )
        .sort("change")
    )


def distinctive(counts: pl.DataFrame, n: int) -> pl.DataFrame:
    """Los términos más característicos de un conjunto de conteos (term, titles)."""
    total = load("meta")["titles"][0]
    own = max(counts["titles"].sum(), 1)
    return (
        counts.join(load("term_df").rename({"titles": "everywhere"}), on="term")
        .with_columns(lift=(pl.col("titles") / own) / (pl.col("everywhere") / total))
        .filter(pl.col("lift") > 1)
        .with_columns(score=pl.col("titles") * pl.col("lift").log())
        .sort("score", descending=True)
        .head(n)
    )


def number(value: object) -> float:
    """Polars tipa sus agregados como cualquier escalar; en estas tablas son números."""
    return float(cast(SupportsFloat, value))


def ratio(part: object, whole: object) -> float:
    """part / whole, sin dividir entre cero cuando no hay nada."""
    return number(part) / max(number(whole), 1)


def fmt_share(value: float | None) -> str:
    return "—" if value is None or math.isnan(value) else f"{value:.0%}"
