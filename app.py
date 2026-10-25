"""El CIC en datos: un tablero público para estudiantes del CIC.

Solo lee data/*.parquet, escrito por export.py. No se conecta a ninguna base de
datos y todas las gráficas se construyen con conteos agregados.

    uv run streamlit run app.py
"""

import logging

import streamlit as st


@st.cache_resource
def configure_logging() -> None:
    """Lo que registra la app (búsquedas, topes, fallas de DashScope) sale en `docker logs`.

    Va bajo el logger "cic", con fecha y nivel. Una sola vez por proceso: el caché evita que
    dos sesiones que arrancan a la vez agreguen el handler dos veces. Nunca se registra la
    idea del visitante ni su IP.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    app_log = logging.getLogger("cic")
    app_log.addHandler(handler)
    app_log.setLevel(logging.INFO)


configure_logging()
st.set_page_config(page_title="El CIC en datos", page_icon="🔭", layout="wide")

pages = [
    st.Page("views/inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("views/encuentra.py", title="Encuentra tu lab", icon=":material/explore:"),
    st.Page("views/laboratorios.py", title="Laboratorios", icon=":material/science:"),
    st.Page("views/asesores.py", title="Asesores", icon=":material/school:"),
    st.Page("views/perfil.py", title="Perfil de asesor", icon=":material/person_search:"),
    st.Page("views/temas.py", title="Temas", icon=":material/trending_up:"),
    st.Page("views/colaboracion.py", title="Colaboración", icon=":material/hub:"),
    st.Page("views/produccion.py", title="Producción", icon=":material/article:"),
    st.Page("views/acerca.py", title="Acerca de", icon=":material/info:"),
]
st.navigation(pages, position="top").run()
