"""El CIC en datos: un tablero público para estudiantes del CIC.

Solo lee data/*.parquet, escrito por export.py. No se conecta a ninguna base de
datos y todas las gráficas se construyen con conteos agregados.

    uv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="El CIC en datos", page_icon="🔭", layout="wide")

pages = [
    st.Page("views/inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("views/encuentra.py", title="Encuentra tu lab", icon=":material/explore:"),
    st.Page("views/laboratorios.py", title="Laboratorios", icon=":material/science:"),
    st.Page("views/asesores.py", title="Asesores", icon=":material/school:"),
    st.Page("views/temas.py", title="Temas", icon=":material/trending_up:"),
    st.Page("views/colaboracion.py", title="Colaboración", icon=":material/hub:"),
    st.Page("views/produccion.py", title="Producción", icon=":material/article:"),
    st.Page("views/acerca.py", title="Acerca de", icon=":material/info:"),
]
st.navigation(pages, position="top").run()
