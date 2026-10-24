import streamlit as st

from advisor import render_profile
from common import footer, load

st.title("Perfil de asesor")
st.markdown(
    "Busca a una persona investigadora o asesora del CIC para ver las tesis en que participa, "
    "dónde y cuánto publica, sus temas, con quién colabora y su perfil de OpenAlex."
)

# Ligado a la URL (?persona=...): un perfil se puede compartir o abrir desde otra página.
person = st.selectbox(
    "Nombre",
    load("people").sort("person")["person"].to_list(),
    index=None,
    placeholder="Escribe un nombre o apellido",
    key="persona",
    bind="query-params",
)
if person is None:
    st.info(
        "¿No sabes a quién buscar? La página **Asesores** compara a todas las personas del CIC, "
        "y **Encuentra tu lab** sugiere nombres a partir de tu idea de tesis.",
        icon=":material/person_search:",
    )
    with st.container(horizontal=True, gap="small"):
        st.page_link("views/asesores.py", label="Ir a Asesores", icon=":material/school:")
        st.page_link("views/encuentra.py", label="Ir a Encuentra tu lab", icon=":material/explore:")
else:
    st.markdown(f"### {person}")
    render_profile(person)
footer()
