# El CIC en datos

Tablero interactivo y público sobre la investigación del Centro de Investigación en
Computación (CIC) del IPN, pensado para estudiantes que buscan tema de tesis, laboratorio o
asesor. Hecho con Streamlit y Plotly.

| Página | Qué ofrece |
| --- | --- |
| Inicio | Cifras generales, "¿Sabías que…?" calculados con los datos y accesos a cada sección. |
| Encuentra tu lab | Escribes tu idea de tesis y te sugiere laboratorios y personas afines, con acceso a su perfil. |
| Laboratorios | Ranking, perfil de cada laboratorio, comparación en radar y mapa de temas. |
| Asesores | Panorama de todas las personas: quién participa en más tesis y con qué rol, y "publicar vs. dirigir". |
| Perfil de asesor | Buscador y perfil de una persona: tesis, producción por año, temas, colaboración y OpenAlex. Se comparte con `?persona=`. |
| Temas | Seguir términos en el tiempo, temas que crecen o se apagan, español vs. inglés. |
| Colaboración | Mapa de calor entre laboratorios, flujo de codirección y duplas frecuentes. |
| Producción | Productos por año, crecimiento animado por laboratorio, proyectos y revistas. |
| Acerca de | Fuente, qué no se publica, cómo leer las cifras y limitaciones. |

## Qué se publica y qué no

- `export.py` lee la base `cic_scrape` y escribe en `data/` archivos Parquet **agregados**:
  conteos por año, laboratorio, rol, revista, término o persona investigadora. Las únicas
  excepciones son los títulos de artículos, congresos y proyectos del personal investigador,
  y los de tesis con sus directores y comité. No escribe ids ni datos de estudiantes: el
  catálogo ni siquiera guarda al alumno de cada tesis.
- La app solo lee `data/` y no tiene conexión a la base de datos. «Encuentra tu lab» puede
  enviar la idea de búsqueda a DashScope para calcular un embedding; el resto de la app no
  hace llamadas a servicios externos. No se ofrecen tablas de títulos ni botones de descarga.
- Los términos que aparecen en menos de tres títulos se descartan.
- Los títulos visibles son del personal investigador: sus artículos y congresos del catálogo
  del CIC, con año y tipo; los de [OpenAlex](https://openalex.org) (CC0), con enlace a su DOI
  cuando lo tienen; y sus proyectos de investigación, con años y rol. Todos sus integrantes
  son personal del CIC.
- Las tesis muestran título, nivel, directores y comité, **nunca al alumno**. Solo se nombra
  a quien ya aparece en el tablero; el resto se cuenta ("y 2 personas más"). Las tesis que
  indexa OpenAlex siguen ocultas, porque ahí el autor suele ser el alumno.

Lo que una gráfica muestra llega al navegador: los valores del tooltip se pueden leer y la
barra de Plotly guarda un PNG. Que todo esté agregado antes es lo que lo vuelve inofensivo.

## Estructura

```
export.py        base de datos → data/*.parquet (solo en tu máquina)
build_title_index.py  títulos públicos → vectores Qwen en data/ (solo en tu máquina)
dashscope_client.py   cliente DashScope y validación de sus respuestas
semantic_search.py    similitud vectorial y agrupación por asesor/laboratorio
idea_assistant.py     áreas vecinas y resúmenes con Qwen3.7 Flash, verificados
text.py          cómo se parte un texto en términos; lo usan export.py y la app
common.py        carga de datos, etiquetas en español y utilidades de las páginas
app.py           navegación
tests/           pruebas con un catálogo inventado (sin base de datos)
views/           una página por archivo
.streamlit/      configuración: sin menú de desarrollador ni trazas de error para visitantes
Dockerfile       imagen solo con la app (sin export.py, psycopg ni Lingua)
compose.yaml     servicio para el VPS, detrás de Nginx Proxy Manager
```

## 1. Construir la instantánea (en tu máquina)

Necesitas la base `cic_scrape` corriendo (en `theiss-match-cic`: `docker compose up -d db` y
`db/setup.sh`). La conexión se lee de `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`,
`POSTGRES_PASSWORD` y `POSTGRES_DB`; todas salvo la contraseña tienen valores por defecto para
el entorno local. Copia `.env.example` a `.env` (está en `.gitignore`) y pon ahí la contraseña.

```bash
uv run --env-file .env --group export python export.py
```

El grupo `export` añade `psycopg` y [Lingua](https://github.com/pemistahl/lingua-py), que
decide si cada título está en español o en inglés. La app no los necesita y la imagen no
los instala.
`data/` está en `.gitignore`: se copia al servidor aparte.

## 1a. Índice semántico de «Encuentra tu lab» (opcional)

Después de `export.py`, genera el índice de títulos con tu llave de DashScope:

```bash
uv run --env-file .env python build_title_index.py
```

La API recibe los títulos públicos únicos del catálogo; el índice local contiene el vector y
los vínculos a asesores, laboratorios y fuentes. Las tesis se vinculan solo con Director 1 y
Director 2; no se envían nombres del comité ni datos de estudiantes. Se excluyen los trabajos
de OpenAlex en esta primera versión porque el Parquet conserva una selección por citas.
`export.py` no borra `data/title_embeddings.parquet`. Después de cada exportación vuelve a
correr `build_title_index.py`: reutiliza los vectores ya calculados, solo paga los títulos
nuevos y actualiza los vínculos con asesores y laboratorios. Si lo olvidas, la página lo nota: el
índice guarda la fecha de su instantánea y, si no coincide con la de `data/`, usa la búsqueda
por términos hasta que lo reconstruyas. El archivo se copia al VPS con
los demás datos y no se versiona.

Al presionar **Buscar afinidades**, la idea escrita por el visitante se envía a Alibaba Cloud
para obtener su vector. BM25 también busca coincidencias literales en el catálogo y Reciprocal
Rank Fusion combina ambos rankings. La página limita la entrada a 500 caracteres y conserva
los resultados en caché temporal para evitar llamadas repetidas durante los reruns de Streamlit.
Si falta la llave o el índice, o si falla el servicio, «Encuentra tu lab» usa la búsqueda local
por términos. No se deben introducir datos personales o confidenciales en la idea.

Cada búsqueda semántica se cobra a la llave: unos $0.0007 con una idea corta y hasta $0.0017
con una de 500 caracteres, casi todo del reranker. `search_limits.py` pone tres topes: 20
búsquedas por hora por visitante (la IP que reporta Cloudflare), 300 por hora y 2,000 por día
en todo el sitio. Con ellos el gasto máximo ronda $3.40 al día; pasado un tope, la página usa
la búsqueda por términos. Los contadores viven en memoria: bastan porque la app corre en un
solo proceso, y se reinician con el contenedor. Configura además una alerta de gasto en la
consola de Alibaba Cloud.

Dos ayudas usan además Qwen3.7 Flash (~$0.03 por millón de tokens de entrada y $0.13 de
salida), por la ruta compatible con OpenAI del mismo servidor y con la misma llave, y solo
cuando el visitante las pide: sugerir áreas cercanas cuando una palabra no tiene trabajos en
el catálogo, y resumir por qué una persona encaja al abrir su perfil desde «Encuentra tu lab».
Cada una cuesta unos $0.0001, cuenta en los mismos topes y se verifica contra el catálogo
(`idea_assistant.py`): una sugerencia necesita un par de palabras con títulos, y el resumen
solo puede citar, por número, las obras que recibió.

## 1b. Métricas y publicaciones de OpenAlex (opcional)

`openalex.py` vincula a cada persona investigadora con su perfil de
[OpenAlex](https://openalex.org) y trae citas, índice h, citas por año, en qué revistas y
congresos publica y sus trabajos más citados. También cuenta las revistas de todo el CIC y
de cada laboratorio con una sola consulta por grupo, para que un trabajo con varios autores
del CIC cuente una vez. Los datos de OpenAlex son de dominio público (CC0). Corre en tu
máquina, después de `export.py`; el VPS nunca llama a OpenAlex.

Si tienes una [llave gratuita de OpenAlex](https://openalex.org/settings/api-key), ponla en
`OPENALEX_API_KEY` dentro de `.env`: es opcional y multiplica el presupuesto diario. Se manda
en la cabecera `Authorization`, nunca en la URL.

```bash
uv run --env-file .env python openalex.py match  # propone un autor de OpenAlex por persona
uv run --env-file .env python openalex.py fetch  # trae métricas y trabajos de los revisados
```

`match` escribe `openalex_authors.csv` con una columna `status`:

- `auto`: nombre parecido y afiliación al IPN. Suele estar bien, pero revísalo.
- `revisar`: dudoso, o dos personas apuntan al mismo autor. Corrige el id a mano.
- `corregido`: corregido a mano.
- `sin coincidencia`: sin candidato; busca a la persona en openalex.org si quieres agregarla.

Una persona partida en varios perfiles de OpenAlex lleva sus ids separados por `|`; `fetch`
los suma. El CSV está versionado para que las correcciones no se pierdan. Sin los archivos
`openalex_*`, la pestaña **Publicaciones** del perfil simplemente lo indica.

## 2. Probar en local

```bash
uv run --env-file .env streamlit run app.py
```

Sin `DASHSCOPE_API_KEY` o sin `data/title_embeddings.parquet`, la búsqueda usa el modo local
por términos.

## 3. Desplegar en el VPS

### Copiar el proyecto y los datos

```bash
rsync -av --exclude .venv --exclude .git --exclude __pycache__ --exclude '.*_cache' \
    --exclude .claude --exclude .env ./ vps:~/cic-dashboard/
```

Tu `.env` local no se copia: lleva la contraseña de Postgres y la llave de OpenAlex, que el
VPS no necesita. Para habilitar la búsqueda semántica, crea en el VPS un `.env` con
`DASHSCOPE_API_KEY` y `PROXY_NETWORK`; Compose entrega la clave solo al proceso servidor.
Las cachés de mypy, pytest y ruff (más de 150 MB) tampoco se copian.

### Conectar con Nginx Proxy Manager

El contenedor no publica puertos: se une a la red de Docker de Nginx Proxy Manager y NPM le
llega por nombre. Busca cómo se llama esa red en el VPS:

```bash
docker network ls
```

Si no es `npm_default`, añade `PROXY_NETWORK` al `.env` del VPS. Conserva ahí también
`DASHSCOPE_API_KEY`; no copies las credenciales de Postgres ni OpenAlex desde tu máquina.
Después, levanta el servicio:

```bash
docker compose up -d --build
```

En Nginx Proxy Manager, crea un **Proxy Host**:

- **Domain Names**: el subdominio que elijas, por ejemplo `cic.tudominio.com`.
- **Scheme** `http`, **Forward Hostname** `cic-dashboard`, **Forward Port** `8501`.
- Activa **Websockets Support** (Streamlit no funciona sin él) y **Block Common Exploits**.
- En **SSL**: tu certificado (Let's Encrypt con reto DNS de Cloudflare, o un Origin
  Certificate de Cloudflare), con **Force SSL** y **HTTP/2**.

### Cloudflare

- Registro DNS del subdominio **con proxy** (nube naranja).
- **SSL/TLS** en modo **Full (strict)**.
- **WebSockets** activado (viene activado por defecto en *Network*).
- Opcional: una regla de *rate limiting* o *Bot Fight Mode* si ves tráfico abusivo.

## Seguridad

### Lo que ya trae el proyecto

- **Sin base de datos ni datos de estudiantes.** Los títulos de tesis son públicos y el
  catálogo no guarda al alumno. La clave de DashScope se configura como variable de entorno
  del contenedor y solo se usa en el servidor Python.
- **Los Parquet no se sirven por la web.** `enableStaticServing = false` y `data/` no está en
  ninguna ruta pública; tampoco va dentro de la imagen, se monta aparte.
- **Contenedor endurecido**: usuario sin privilegios (uid 1000), disco de solo lectura,
  `cap_drop: ALL`, `no-new-privileges`, sin puertos publicados y 512 MB de memoria máxima.
- **Gasto acotado en DashScope**: la búsqueda semántica tiene topes por visitante, por hora y
  por día (`search_limits.py`). El límite por visitante usa `CF-Connecting-IP`, que solo es
  confiable si al servidor únicamente llega Cloudflare (punto 1 de la lista siguiente).
- **Lo que escriben los visitantes no ejecuta nada**: no hay SQL, `eval` ni
  `unsafe_allow_html`; el campo se limita a 500 caracteres y solo se envía al proveedor al
  presionar Buscar.
- **Sin detalles internos para visitantes**: ni trazas de error ni menú de desarrollador.
  La protección XSRF de Streamlit viene activada.

### Lo que depende del VPS

1. **Que solo Cloudflare llegue al servidor.** Si alguien conoce la IP del VPS puede saltarse
   Cloudflare. Abre los puertos 80 y 443 solo a los
   [rangos de Cloudflare](https://www.cloudflare.com/ips/), por ejemplo con `ufw`:

   ```bash
   for ip in $(curl -s https://www.cloudflare.com/ips-v4) $(curl -s https://www.cloudflare.com/ips-v6); do
       sudo ufw allow proto tcp from "$ip" to any port 80,443
   done
   ```

   Ojo: Docker publica puertos saltándose `ufw`, así que revisa que el `compose` de Nginx
   Proxy Manager no exponga 80/443 a todo internet, o usa un **Cloudflare Tunnel**, que no
   abre ningún puerto.
2. **El panel de Nginx Proxy Manager (puerto 81) no debe ser público.** Publícalo solo en
   `127.0.0.1:81` y entra con un túnel SSH (`ssh -L 8181:127.0.0.1:81 vps`).
3. **SSH solo con llave**, sin contraseña ni acceso directo de root
   (`PasswordAuthentication no`, `PermitRootLogin no`), y actualizaciones automáticas del
   sistema (`unattended-upgrades` en Debian/Ubuntu).
4. **Límite de visitas en Cloudflare.** El riesgo realista es la saturación: cada visitante
   abre un websocket. Una regla de *rate limiting* (por ejemplo, 60 peticiones por minuto por
   IP) y *Bot Fight Mode* lo cubren casi todo; el límite de memoria evita que tumbe el VPS.
5. **Permisos de los datos.** Que solo el usuario del contenedor los lea:

   ```bash
   sudo chown -R 1000:1000 ~/cic-dashboard/data && sudo chmod 700 ~/cic-dashboard/data && sudo chmod 600 ~/cic-dashboard/data/*
   ```

   Si tu usuario del VPS no es el uid 1000 (`id -u`), después necesitarás `sudo` para
   actualizar `data/`.

6. **Parches.** Reconstruye la imagen de vez en cuando para recibir las correcciones de
   Streamlit y de Python:

   ```bash
   docker compose build --pull && docker compose up -d
   ```

### ¿Por qué los datos no están cifrados?

La app necesita descifrarlos para dibujar las gráficas, así que la llave tendría que vivir en
el mismo servidor: quien saque el archivo del VPS también puede sacar la llave. Además, cada
gráfica le envía sus números al navegador. Cifrar solo tiene sentido para las copias que salen
del servidor, como los respaldos.

## 4. Actualizar los datos

```bash
uv run --env-file .env --group export python export.py
rsync -av --delay-updates data/ vps:~/cic-dashboard/data/
```

No hace falta reiniciar ni reconstruir nada: la app vuelve a leer cada tabla en cuanto
cambia la fecha de modificación de su archivo. `rsync` reemplaza cada archivo de golpe, así
que nunca se lee uno a medio copiar, y con `--delay-updates` deja listos todos los archivos
nuevos y los coloca juntos al final: una página no mezcla tablas viejas y nuevas mientras
dura la copia.

## Desarrollo

```bash
uv sync --all-groups          # app, export y herramientas
uv run pre-commit install     # los mismos chequeos que CI, en cada commit
uv run pytest                 # pruebas
```

Las pruebas no usan la base de datos ni `data/`: `tests/conftest.py` inventa un catálogo
pequeño y lo pasa por `export.build()`, el mismo camino que siguen los datos reales. Así
se prueban las páginas y, además, que **ningún título ni id llegue a la instantánea**.

CI (`.github/workflows/ci.yml`) corre ruff, mypy, bandit y pytest, y construye la imagen
de Docker. Dependabot propone actualizaciones mensuales de uv, GitHub Actions, pre-commit y
las imágenes base, con una semana de espera.

## Licencia y fuente

El código se distribuye bajo la [Licencia Apache 2.0](LICENSE); ver también
[NOTICE](NOTICE).

Los datos **no** forman parte del repositorio. Se calculan a partir del sitio del Centro de
Investigación en Computación del Instituto Politécnico Nacional,
[www.cic.ipn.mx](https://www.cic.ipn.mx), cuyo contenido puede reproducirse con fines no
lucrativos, sin mutilarlo y citando la fuente completa y su dirección electrónica. El
tablero cita esa fuente en cada página. Es un proyecto independiente y sin fines de lucro,
sin relación oficial con el IPN ni el CIC.
