"""Búsqueda de laboratorios y asesores a partir de una idea de tesis."""

import plotly.express as px
import polars as pl
import streamlit as st

import text
from advisor import profile_dialog
from common import LABELS, TEAL_SCALE, explain, footer, lab_colors, load, load_optional, match, show
from dashscope_client import (
    MAX_QUERY_CHARS,
    MAX_RERANK_DOCUMENTS,
    EmbeddingError,
    RerankError,
    embed_query,
    is_configured,
    rerank_titles,
)
from search_limits import SearchLimiter, client_key
from semantic_search import (
    apply_rerank,
    hybrid_rank_titles,
    lexical_variants,
    recommend,
    validate_index,
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
        st.session_state.pop("searched_idea", None)


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
    submitted = st.form_submit_button("Buscar afinidades", type="primary", icon=":material/search:")

if submitted:
    if not idea.strip():
        st.info("Escribe tu idea o elige un ejemplo antes de buscar.", icon=":material/edit:")
        footer()
        st.stop()
    st.session_state["searched_idea"] = idea.strip()
    # Solo una búsqueda enviada cuenta; volver a dibujar la página no llama a la API.
    if is_configured():
        client = client_key(st.context.headers, st.context.ip_address)
        st.session_state["limit_reason"] = search_limiter().allow(client)

query = st.session_state.get("searched_idea", "")
if not query:
    st.info(
        "Escribe algunas palabras arriba o elige un ejemplo. Entre más específico, mejor: "
        "*“segmentación de imágenes satelitales”* da mejores pistas que *“inteligencia "
        "artificial”*.",
        icon=":material/edit:",
    )
    footer()
    st.stop()

if len(query) > MAX_QUERY_CHARS:
    st.error(f"La consulta no puede superar {MAX_QUERY_CHARS} caracteres.")
    footer()
    st.stop()
if idea.strip() != query:
    st.caption(
        "Los resultados corresponden a la última búsqueda enviada. "
        "Presiona Buscar para actualizarlos."
    )

term_counts = dict(load("term_df").iter_rows())
recognized = [term for term in text.terms(query) if term in term_counts]
ignored = [term for term in text.terms(query) if " " not in term and term not in recognized]
index = load_optional("title_embeddings")
variants = (
    lexical_variants(
        query,
        index.select("title_id", "title")
        .unique(subset="title_id", keep="first", maintain_order=True)["title"]
        .to_list(),
    )
    if index is not None
    else {}
)
semantic = False
limit_reason = st.session_state.get("limit_reason")

if index is not None and is_configured() and limit_reason:
    st.info(
        f"{LIMIT_MESSAGES[limit_reason]}; mostramos la búsqueda por términos, que no tiene "
        "límite. Vuelve a intentar más tarde.",
        icon=":material/hourglass_top:",
    )
    recognized, labs, people = match(query)
elif index is not None and is_configured():
    try:
        validate_index(index)
        with st.spinner("Buscando títulos relacionados…"):
            query_vector = cached_query_embedding(query)
            ranked = hybrid_rank_titles(index, query, query_vector)
            candidate_titles = tuple(ranked.head(MAX_RERANK_DOCUMENTS)["title"].to_list())
            try:
                rerank_results = cached_title_rerank(query, candidate_titles)
                ranked = apply_rerank(ranked, rerank_results)
            except RerankError:
                st.warning(
                    "No se pudo ordenar los títulos con el reranker; usamos el ranking híbrido.",
                    icon=":material/cloud_off:",
                )
            labs = recommend(index, ranked, group_by="lab")
            people = recommend(index, ranked, group_by="person")
        semantic = True
    except (EmbeddingError, ValueError):
        st.warning(
            "No se pudo completar la búsqueda semántica. Mostramos los resultados por "
            "coincidencia de términos.",
            icon=":material/cloud_off:",
        )
        recognized, labs, people = match(query)
else:
    if index is None:
        st.info("Aún no se generó el índice semántico; usamos la búsqueda por términos.")
    elif not is_configured():
        st.info("DashScope no está configurado en el servidor; usamos la búsqueda por términos.")
    recognized, labs, people = match(query)

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

if semantic:
    st.caption(
        "La idea y hasta 100 títulos públicos se envían a Alibaba Cloud para calcular similitud "
        "y ordenar los resultados. "
        "No incluyas información personal o confidencial."
    )

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
                st.caption(f"Evidencia relacionada: {row['evidence_count']} títulos")
            else:
                st.caption(f"En común contigo: {row['matched']}")
            if st.button(
                "Ver perfil completo",
                key=f"open-{row['person']}",
                icon=":material/open_in_full:",
            ):
                profile_dialog(row["person"])

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
Cada recomendación muestra la obra que más contribuyó a que apareciera.

**Limitaciones.** Solo se leen títulos, no resúmenes. Las tesis se vinculan con Director 1 y
Director 2; los comités no cuentan como experiencia de dirección. Es una herramienta para
explorar posibles laboratorios y asesores, no una recomendación definitiva.
""",
    label="¿Cómo funciona la búsqueda?",
)
footer()
