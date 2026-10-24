import streamlit as st

from common import AUTHOR, AUTHOR_URL, REPO_URL, SOURCE, SOURCE_URL, footer, load, snapshot

totals = dict(load("totals").iter_rows())

st.title("Acerca de este tablero")
st.markdown(
    f"""
### ¿De dónde salen los datos?

Del sitio del **{SOURCE}**, [{SOURCE_URL}]({SOURCE_URL}): perfiles de personal
investigador, laboratorios, tesis, proyectos, artículos y congresos, consultados el
**{snapshot():%d/%m/%Y}**. El catálogo tiene {totals["Tesis"]:,} tesis, {totals["Proyectos"]:,}
proyectos y {totals["Artículos y congresos"]:,} artículos y congresos.

Las métricas de citas y los trabajos más citados de cada persona vienen de
[OpenAlex](https://openalex.org), cuyos datos son de dominio público (CC0). El vínculo entre
cada persona del CIC y su perfil de OpenAlex se revisó a mano; si ves uno equivocado, avísanos.

El sitio del CIC permite reproducir su contenido con fines no lucrativos, sin mutilarlo y citando la
fuente completa. Este tablero no reproduce las páginas: calcula conteos a partir de ellas,
cita su origen en cada página y no tiene fines de lucro.

### ¿Qué **no** se publica?

- **Ningún dato de estudiantes**: ni nombres, ni correos, ni quién hizo qué tesis. No es
  posible descargar el catálogo desde aquí.
- De las **tesis** se muestran el título, el nivel, quién las dirige y su comité tutorial,
  pero nunca el alumno. Solo se nombra a quien ya aparece en este tablero; el resto de
  quienes participaron se cuenta sin nombre. El título es el mismo que publica el sitio del
  CIC, donde sí aparece el alumno.
- Los títulos de **artículos y congresos del personal investigador** sí se muestran: son
  publicaciones públicas y aparecen con el nombre de la persona. Los de
  [OpenAlex](https://openalex.org) (CC0) llevan enlace a su DOI cuando está disponible y
  nunca incluyen tesis; los del catálogo del CIC, año y tipo.
- Los títulos de los **proyectos de investigación** también se muestran en el perfil de cada
  persona, con sus años y su rol. Están en el sitio del CIC y todos sus integrantes son
  personal investigador.
- Los nombres que aparecen son de personal investigador y asesores, tal como figuran en el
  sitio del CIC, junto con cifras que cualquiera puede contar ahí.

### Cómo leer las cifras

- **Una tesis pertenece al laboratorio de su Director 1.** Si el Director 1 no tiene perfil
  en el CIC, la tesis no tiene laboratorio.
- **Los temas salen de los títulos**, no de resúmenes ni del texto completo. Se buscan
  palabras, no significados: *“aprendizaje profundo”* y *“deep learning”* son términos
  distintos.
- **El catálogo no está completo ni es perfecto.** Puede haber homónimos, nombres escritos de
  varias formas o publicaciones que el sitio no registra. El año en curso siempre es parcial.

### ¿Para qué sirve?

Para orientarte: descubrir qué se investiga en el CIC, en qué laboratorios y con quién, antes
de acercarte a un posible asesor. **No sustituye una conversación** con las personas ni la
información oficial del posgrado.

### Contacto y correcciones

Este tablero es un proyecto personal de **{AUTHOR}** ([@charecktowa]({AUTHOR_URL})). Su
código es abierto y está en [GitHub]({REPO_URL}). Si encuentras un error o tienes una idea,
[abre un *issue*]({REPO_URL}/issues) en el repositorio. Las correcciones de fondo (un nombre,
una tesis que falta) deben hacerse en el sitio del CIC; el tablero se actualiza a partir de ahí.
"""
)
footer()
