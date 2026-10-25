"""Búsqueda de laboratorios y asesores a partir de una idea de tesis."""

import re
import threading
from dataclasses import dataclass
from functools import partial

import plotly.express as px
import polars as pl
import streamlit as st

import text
from advisor import profile_dialog
from common import (
    DATA,
    LABELS,
    TEAL_SCALE,
    count_label,
    explain,
    footer,
    lab_colors,
    load,
    match,
    show,
    snapshot,
)
from dashscope_client import (
    MAX_QUERY_CHARS,
    MAX_RERANK_DOCUMENTS,
    ChatError,
    EmbeddingError,
    RerankError,
    embed_query,
    is_configured,
    rerank_titles,
)
from idea_assistant import Suggestion, Summary, advisor_summary, neighbor_areas
from search_limits import SEARCHES_AT_ONCE, SearchLimiter, client_key
from semantic_search import (
    EVIDENCE_LIMIT,
    TitleIndex,
    apply_rerank,
    hybrid_rank_titles,
    lexical_variants,
    prepare_index,
    recommend,
    uncovered_words,
)

EXAMPLES = {
    "Noticias falsas": "detección de noticias falsas en redes sociales con lenguaje natural",
    "Visión": "visión por computadora y redes neuronales convolucionales para imágenes médicas",
    "Robots": "robots móviles, control y navegación autónoma",
    "Ciberseguridad": "ciberseguridad, criptografía y detección de intrusiones",
    "Cuántica": "computación cuántica y algoritmos cuánticos",
    "Blockchain": "blockchain y contratos inteligentes",
    "Educación": "aprendizaje automático aplicado a la educación",
}


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_query_embedding(query: str) -> list[float]:
    """Evita repetir la llamada externa si Streamlit vuelve a ejecutar la página."""
    return embed_query(query)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_title_rerank(query: str, titles: tuple[str, ...]) -> list[tuple[int, float]]:
    """Evita repetir el reranking cuando Streamlit vuelve a ejecutar la página."""
    return rerank_titles(query, titles)


INDEX_PATH = DATA / "title_embeddings.parquet"


def index_version() -> int | None:
    """La versión del índice (su fecha de modificación); None si todavía no existe."""
    return INDEX_PATH.stat().st_mtime_ns if INDEX_PATH.exists() else None


# Uno solo y compartido: la versión anterior sale en cuanto cambia el archivo.
@st.cache_resource(max_entries=1, show_spinner=False)
def title_index(version: int) -> TitleIndex:
    """El índice preparado una vez para todas las sesiones; `version` distingue la caché."""
    return prepare_index(pl.read_parquet(INDEX_PATH))


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_variants(query: str, version: int) -> dict[str, list[str]]:
    """Variantes léxicas de la idea en el vocabulario del índice."""
    return lexical_variants(query, title_index(version).corpus)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_ranking(query: str, version: int) -> pl.DataFrame:
    """Los títulos que pueden respaldar una recomendación, ya ordenados por la búsqueda híbrida.

    Se calcula una vez por idea: abrir un perfil o cambiar de pestaña no lo repite.
    """
    ranked = hybrid_rank_titles(title_index(version), query, cached_query_embedding(query))
    return ranked.head(max(EVIDENCE_LIMIT, MAX_RERANK_DOCUMENTS))


@st.cache_resource
def search_slots() -> threading.BoundedSemaphore:
    """Cuántas búsquedas semánticas corren a la vez; las demás esperan su turno."""
    return threading.BoundedSemaphore(SEARCHES_AT_ONCE)


@st.cache_resource
def search_limiter() -> SearchLimiter:
    """Un solo contador para todas las sesiones del proceso."""
    return SearchLimiter()


LIMIT_MESSAGES = {
    "visitante": "Hiciste muchas búsquedas en la última hora",
    "hora": "El tablero alcanzó su límite de búsquedas por hora",
    "día": "El tablero alcanzó su límite de búsquedas del día",
}


def use_example() -> None:
    picked = st.session_state.get("example")
    if picked:
        st.session_state["idea"] = EXAMPLES[picked]
        st.session_state.pop("results", None)


@dataclass(frozen=True)
class Results:
    """Lo que encontró una búsqueda enviada. Vive en la sesión: volver a dibujar la página
    (abrir un perfil, por ejemplo) lo reutiliza y nunca llama otra vez a DashScope."""

    query: str
    recognized: list[str]
    labs: pl.DataFrame
    people: pl.DataFrame
    variants: dict[str, list[str]]
    semantic: bool
    # (tipo, texto, ícono), con tipo "info" o "warning", para mostrar sobre los resultados.
    notices: list[tuple[str, str, str]]
    # Las palabras de la idea que ningún título menciona, como las escribió el visitante.
    uncovered: tuple[str, ...] = ()


def as_written(query: str, term: str) -> str:
    """El término como lo escribió el visitante, con sus acentos ("odontología")."""
    for raw in re.findall(r"[\w-]+", query):
        if text.words(raw) == [term]:
            return raw
    return term


def recognized_terms(query: str) -> list[str]:
    """Los términos de la idea que aparecen en el catálogo."""
    known = set(load("term_df")["term"])
    return [term for term in text.terms(query) if term in known]


def term_results(query: str, notices: list[tuple[str, str, str]]) -> Results:
    """La búsqueda local por términos: no llama a ningún servicio."""
    recognized, labs, people = match(query)
    return Results(query, recognized, labs, people, {}, False, notices)


def run_search(query: str) -> Results:
    """Busca una idea enviada. Es la única ruta que puede llamar a DashScope."""
    version = index_version()
    if version is None:
        info = "Aún no se generó el índice semántico; usamos la búsqueda por términos."
        return term_results(query, [("info", info, ":material/info:")])
    if not is_configured():
        info = "DashScope no está configurado en el servidor; usamos la búsqueda por términos."
        return term_results(query, [("info", info, ":material/info:")])
    limit = search_limiter().allow(client_key(st.context.headers, st.context.ip_address))
    if limit:
        info = (
            f"{LIMIT_MESSAGES[limit]}; mostramos la búsqueda por términos, que no tiene "
            "límite. Vuelve a intentar más tarde."
        )
        return term_results(query, [("info", info, ":material/hourglass_top:")])

    notices: list[tuple[str, str, str]] = []
    try:
        with st.spinner("Buscando títulos relacionados…"), search_slots():
            index = title_index(version)
            if index.snapshot != snapshot():
                # Vínculos de otra instantánea: pueden nombrar títulos o personas que ya no
                # están. Hasta reconstruirlo, la búsqueda por términos.
                info = (
                    "El índice semántico es de otra instantánea de datos; usamos la búsqueda "
                    "por términos hasta reconstruirlo."
                )
                return term_results(query, [("info", info, ":material/update:")])
            variants = cached_variants(query, version)
            ranked = cached_ranking(query, version)
            candidate_titles = tuple(ranked.head(MAX_RERANK_DOCUMENTS)["title"].to_list())
            try:
                ranked = apply_rerank(ranked, cached_title_rerank(query, candidate_titles))
            except RerankError:
                warning = (
                    "No se pudo ordenar los títulos con el reranker; usamos el ranking híbrido."
                )
                notices.append(("warning", warning, ":material/cloud_off:"))
            labs = recommend(index, ranked, group_by="lab")
            people = recommend(index, ranked, group_by="person")
    except (EmbeddingError, ValueError):
        warning = (
            "No se pudo completar la búsqueda semántica. Mostramos los resultados por "
            "coincidencia de términos."
        )
        return term_results(query, [("warning", warning, ":material/cloud_off:")])
    uncovered = tuple(
        as_written(query, term) for term in uncovered_words(query, index.corpus, variants)
    )
    if uncovered:
        missing = ", ".join(f"«{word}»" for word in uncovered)
        info = (
            f"Ningún título del catálogo menciona {missing}. Las sugerencias se basan en el "
            "resto de tu idea y en temas parecidos: tómalas como un punto de partida."
        )
        notices.append(("info", info, ":material/search_off:"))
    return Results(query, recognized_terms(query), labs, people, variants, True, notices, uncovered)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_neighbors(query: str, uncovered: tuple[str, ...], version: int) -> list[Suggestion]:
    """Las áreas vecinas de una idea; `version` las renueva si cambia el índice."""
    return neighbor_areas(query, list(uncovered), title_index(version).corpus)


def suggest_neighbors(results: Results) -> None:
    """Pide al chat las áreas vecinas, si caben en los topes, y las guarda en la sesión."""
    version = index_version()
    limit = search_limiter().allow(client_key(st.context.headers, st.context.ip_address))
    if limit or version is None:
        st.session_state["neighbors"] = (results.query, LIMIT_MESSAGES.get(limit or "", ""), [])
        return
    try:
        # Sin spinner: un callback dibuja arriba de la página, no junto al botón.
        with search_slots():
            found = cached_neighbors(results.query, results.uncovered, version)
    except ChatError:
        found = []
    st.session_state["neighbors"] = (results.query, "", found)


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def cached_summary(query: str, person: str, works: tuple[str, ...]) -> Summary | None:
    """El resumen de por qué una persona encaja con una idea, una vez por idea y persona."""
    return advisor_summary(query, person, list(works))


def render_fit(query: str, person: str, works: tuple[str, ...]) -> None:
    """Por qué esta persona encaja con la idea. Solo aparece al abrir su perfil desde aquí.

    Se guarda en la sesión: interactuar dentro del diálogo no vuelve a llamar al chat ni a
    contar en los topes.
    """
    stored: dict[tuple[str, str], Summary | str | None] = st.session_state.setdefault(
        "summaries", {}
    )
    if (query, person) not in stored:
        limit = search_limiter().allow(client_key(st.context.headers, st.context.ip_address))
        if limit:
            stored[query, person] = LIMIT_MESSAGES[limit]
        else:
            try:
                with st.spinner("Resumiendo por qué encaja con tu idea…"), search_slots():
                    stored[query, person] = cached_summary(query, person, works)
            except ChatError:
                stored[query, person] = None
    summary = stored[query, person]
    with st.container(border=True):
        st.markdown("**Por qué podría encajar con tu idea**")
        if isinstance(summary, Summary):
            # Texto plano: lo escribió un modelo y no debe interpretarse como formato.
            st.text(summary.text)
            st.caption("Obra seleccionada")
            st.text(summary.work)
            st.caption(
                "Resumen generado por IA (Qwen) a partir de sus obras afines a tu idea; puede "
                "equivocarse. Revisa sus trabajos antes de escribirle."
            )
        elif isinstance(summary, str):
            st.caption(f"{summary}; el resumen con IA no está disponible por ahora.")
        else:
            st.caption("No pudimos generar el resumen esta vez.")


def search_neighbor(query: str) -> None:
    """Lanza la búsqueda de un área vecina como si el visitante la hubiera escrito."""
    st.session_state["idea"] = query
    st.session_state["pending_search"] = query


st.title("Encuentra tu laboratorio y tu asesor")
st.markdown(
    "Escribe tu idea de tesis o lo que te gustaría investigar, en español o en inglés. "
    "El tablero compara la idea con títulos de **tesis dirigidas, artículos, congresos y "
    "proyectos** del CIC, y muestra las obras que respaldan cada sugerencia."
)

st.pills(
    "¿Sin idea todavía? Prueba un ejemplo:",
    list(EXAMPLES),
    key="example",
    on_change=use_example,
)

with st.form("finder"):
    idea = st.text_area(
        "Tu idea",
        key="idea",
        placeholder="Por ejemplo: detección de emociones en texto con deep learning",
        height=90,
        max_chars=MAX_QUERY_CHARS,
    )
    # El aviso va antes del botón: nada sale del servidor sin que el visitante lo haya leído.
    if index_version() is not None and is_configured():
        st.caption(
            "Al buscar, tu idea y hasta 100 títulos públicos se envían a Alibaba Cloud para "
            "calcular similitud y ordenar los resultados. No incluyas información personal o "
            "confidencial."
        )
    submitted = st.form_submit_button("Buscar afinidades", type="primary", icon=":material/search:")

if submitted:
    if not idea.strip():
        st.info("Escribe tu idea o elige un ejemplo antes de buscar.", icon=":material/edit:")
        footer()
        st.stop()
    st.session_state["results"] = run_search(idea.strip())
elif pending := st.session_state.pop("pending_search", None):
    st.session_state["results"] = run_search(pending)

results: Results | None = st.session_state.get("results")
if results is None:
    st.info(
        "Escribe algunas palabras arriba o elige un ejemplo. Entre más específico, mejor: "
        "*“segmentación de imágenes satelitales”* da mejores pistas que *“inteligencia "
        "artificial”*.",
        icon=":material/edit:",
    )
    footer()
    st.stop()

query, recognized, labs, people = results.query, results.recognized, results.labs, results.people
variants, semantic = results.variants, results.semantic
if idea.strip() != query:
    st.caption(
        "Los resultados corresponden a la última búsqueda enviada. "
        "Presiona Buscar para actualizarlos."
    )
for kind, message, icon in results.notices:
    (st.warning if kind == "warning" else st.info)(message, icon=icon)

# El chat solo se llama si el visitante lo pide: sin clic, no hay costo.
if results.semantic and results.uncovered:
    asked, limit_message, neighbors = st.session_state.get("neighbors", ("", "", []))
    if asked != query:
        st.button(
            "Sugerir áreas cercanas que sí están en el catálogo (IA)",
            icon=":material/auto_awesome:",
            on_click=suggest_neighbors,
            args=(results,),
        )
    elif limit_message:
        st.info(f"{limit_message}; intenta más tarde.", icon=":material/hourglass_top:")
    elif not neighbors:
        st.info(
            "No encontramos un área cercana con trabajos en el catálogo. Prueba con otras "
            "palabras de tu idea.",
            icon=":material/search_off:",
        )
    else:
        st.markdown("**Áreas cercanas con trabajos en el CIC** · sugeridas por IA")
        with st.container(horizontal=True, gap="small"):
            for number, suggestion in enumerate(neighbors):
                st.button(
                    suggestion.topic,
                    key=f"neighbor-{number}",
                    help=suggestion.query,
                    icon=":material/travel_explore:",
                    on_click=search_neighbor,
                    args=(suggestion.query,),
                )

term_counts = dict(load("term_df").iter_rows())
ignored = [term for term in text.terms(query) if " " not in term and term not in recognized]

st.markdown("**Términos reconocidos en el catálogo**")
if recognized:
    with st.container(horizontal=True, gap="small"):
        for term in recognized[:16]:
            st.badge(f"{term} · {term_counts.get(term, 0):,} títulos", color="primary")
        for term in ignored[:8]:
            if term in variants:
                matches = ", ".join(variants[term][:3])
                st.badge(f"{term} ≈ {matches}", color="primary")
            else:
                st.badge(f"{term} · sin títulos", color="gray")
elif semantic:
    st.caption(
        "No hay coincidencias literales suficientes; los resultados usan similitud semántica."
    )
else:
    st.warning(
        "Ningún término de tu idea aparece en al menos tres títulos del catálogo. Prueba con "
        "sinónimos, en inglés (muchos títulos lo están) o con palabras más generales.",
        icon=":material/search_off:",
    )
    footer()
    st.stop()

chart_labels = {**LABELS, "matched": "Obra más afín" if semantic else LABELS["matched"]}
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Laboratorios afines")
    top_labs = labs.head(5).sort("affinity")
    if top_labs.is_empty():
        st.info("No encontramos laboratorios con títulos relacionados.")
    else:
        fig = px.bar(
            top_labs,
            x="affinity",
            y="lab",
            orientation="h",
            color="affinity",
            color_continuous_scale=TEAL_SCALE,
            range_color=[0, 100],
            hover_data={"matched": True, "affinity": True},
            labels=chart_labels,
        )
        fig.update_layout(coloraxis_showscale=False, height=60 * top_labs.height + 60)
        fig.update_xaxes(range=[0, 105])
        show(fig)
with right:
    st.subheader("Personas afines")
    top_people = people.head(5).sort("affinity")
    if top_people.is_empty():
        st.info("No encontramos asesores con títulos relacionados.")
    else:
        fig = px.bar(
            top_people,
            x="affinity",
            y="person",
            orientation="h",
            color="lab",
            color_discrete_map=lab_colors(),
            hover_data={"lab": True, "matched": True, "affinity": True},
            labels=chart_labels,
            category_orders={"person": people.head(5)["person"].to_list()},
        )
        fig.update_layout(
            height=45 * top_people.height + 100, legend={"orientation": "h", "y": -0.2}
        )
        fig.update_xaxes(range=[0, 105])
        show(fig)

st.subheader("Cinco asesores y las obras relacionadas")
people_rows = list(people.head(5).iter_rows(named=True))
for start in range(0, len(people_rows), 2):
    columns = st.columns(2, gap="large")
    for column, row in zip(columns, people_rows[start : start + 2], strict=False):
        profile = load("people").filter(pl.col("person") == row["person"]).row(0, named=True)
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{row['person']}**")
            st.caption(profile["lab"] or "Sin laboratorio en el CIC")
            st.markdown(
                f"{profile['director_1']} tesis como Director 1 · "
                f"{profile['outputs']} productos · {profile['projects_led']} proyectos liderados"
            )
            if semantic:
                st.caption("Obra más afín")
                st.text(f"{row['evidence_source']}: {row['evidence_title']}")
                total = row["evidence_count"]
                with st.expander(f"Obras afines ({total})", icon=":material/library_books:"):
                    # Texto plano: un título con * o _ no debe volverse formato.
                    st.text("\n".join(f"• {work}" for work in row["related"]))
                    if total > len(row["related"]):
                        st.caption(
                            f"Los {len(row['related'])} más afines de "
                            f"{count_label(total, 'título')} relacionados con tu idea."
                        )
            else:
                st.caption(f"En común contigo: {row['matched']}")
            if st.button(
                "Ver perfil completo",
                key=f"open-{row['person']}",
                icon=":material/open_in_full:",
            ):
                # El resumen con IA solo existe con obras afines, es decir, en la búsqueda
                # semántica; desde la búsqueda por términos, el perfil de siempre.
                fit = (
                    partial(render_fit, query, row["person"], tuple(row["related"]))
                    if semantic
                    else None
                )
                profile_dialog(row["person"], intro=fit)

st.subheader("¿Tu tema está creciendo?")
if recognized:
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
else:
    st.caption("La tendencia por término solo aparece cuando hay coincidencias literales.")

explain(
    """
**Qué se compara.** Cuando el índice y la API están disponibles, la idea se transforma en un
vector que recupera títulos de significado parecido, incluso entre idiomas. BM25 añade
coincidencias literales para términos técnicos y el ranking combina ambas señales. Un reranker
especializado vuelve a ordenar hasta 100 títulos candidatos según su relación con la idea. Si
el reranker falla, se conserva el ranking híbrido; si falla la búsqueda híbrida, la página usa
la búsqueda local por palabras y pares de palabras.

**Qué significan los resultados.** La afinidad es relativa al resultado mejor posicionado en
esta búsqueda; no es una probabilidad de éxito ni una evaluación de la calidad del asesor.
Cada recomendación muestra la obra por la que esa persona entró a la lista: casi siempre su
obra más afín, salvo cuando esa obra ya respaldó a otra persona y entra por la siguiente. En
**Obras afines** están hasta cinco de sus trabajos más relacionados con tu idea, del más afín
al menos. Primero, cada obra respalda a una
sola persona y a un solo laboratorio, el de quien la dirige o encabeza, para que un proyecto
con muchos integrantes no llene la lista; los demás pueden aparecer por otra obra suya. Si
después sobran lugares, una obra ya usada puede respaldar a alguien más, como un coasesor que
no tiene otra obra relacionada.

**El orden.** Va primero quien trabaja tu tema de fondo: cuenta la relevancia de sus tres
obras más afines, la mejor completa, la segunda a la mitad y la tercera a un cuarto. Así una
persona con un solo artículo parecido no queda por encima de quien tiene varios, pero dos
obras poco afines tampoco ganan a una muy cercana.

En la búsqueda semántica, si ningún título del catálogo menciona una palabra de tu idea de 6
letras o más, la página lo dice.

**Limitaciones.** Solo se leen títulos, no resúmenes. Las tesis se vinculan con Director 1 y
Director 2; los comités no cuentan como experiencia de dirección. Es una herramienta para
explorar posibles laboratorios y asesores, no una recomendación definitiva.
""",
    label="¿Cómo funciona la búsqueda?",
)
footer()
