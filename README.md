# El CIC en datos

Tablero interactivo y público sobre la investigación del Centro de Investigación en
Computación (CIC) del IPN, pensado para estudiantes que buscan tema de tesis, laboratorio o
asesor. Hecho con Streamlit y Plotly.

| Página | Qué ofrece |
| --- | --- |
| Inicio | Cifras generales, "¿Sabías que…?" calculados con los datos y accesos a cada sección. |
| Encuentra tu lab | Escribes tu idea de tesis y te sugiere laboratorios y personas afines, con acceso a su perfil. |
| Laboratorios | Ranking, perfil de cada laboratorio, comparación en radar y mapa de temas. |
| Asesores | Quién participa en más tesis y con qué rol, perfil por persona y "publicar vs. dirigir". |
| Temas | Seguir términos en el tiempo, temas que crecen o se apagan, español vs. inglés. |
| Colaboración | Mapa de calor entre laboratorios, flujo de codirección y duplas frecuentes. |
| Producción | Productos por año, crecimiento animado por laboratorio, proyectos y revistas. |
| Acerca de | Fuente, qué no se publica, cómo leer las cifras y limitaciones. |

## Qué se publica y qué no

- `export.py` lee la base `cic_scrape` y escribe en `data/` archivos Parquet **agregados**:
  conteos por año, laboratorio, rol, revista, término o persona investigadora. No escribe
  títulos, ids ni datos de estudiantes.
- La app solo lee `data/`. No tiene conexión a la base de datos ni credenciales, y no ofrece
  tablas de títulos ni botones de descarga.
- Los términos que aparecen en menos de tres títulos se descartan.

Lo que una gráfica muestra llega al navegador: los valores del tooltip se pueden leer y la
barra de Plotly guarda un PNG. Que todo esté agregado antes es lo que lo vuelve inofensivo.

## Estructura

```
export.py        base de datos → data/*.parquet (solo en tu máquina)
text.py          cómo se parte un texto en términos; lo usan export.py y la app
common.py        carga de datos, etiquetas en español y utilidades de las páginas
app.py           navegación
views/           una página por archivo
.streamlit/      configuración: sin menú de desarrollador ni trazas de error para visitantes
Dockerfile       imagen solo con la app (sin export.py ni psycopg)
compose.yaml     servicio para el VPS, detrás de Nginx Proxy Manager
```

## 1. Construir la instantánea (en tu máquina)

Necesitas la base `cic_scrape` corriendo (en `theiss-match-cic`: `docker compose up -d db` y
`db/setup.sh`). Las variables `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`,
`POSTGRES_PASSWORD` y `POSTGRES_DB` tienen valores por defecto para el entorno local.

```bash
uv run --group export python export.py
```

El grupo `export` añade `psycopg`; la app no lo necesita y la imagen no lo instala.
`data/` está en `.gitignore`: se copia al servidor aparte.

## 2. Probar en local

```bash
uv run streamlit run app.py
```

## 3. Desplegar en el VPS

### Copiar el proyecto y los datos

```bash
rsync -av --exclude .venv --exclude .git --exclude __pycache__ ./ vps:~/cic-dashboard/
```

### Conectar con Nginx Proxy Manager

El contenedor no publica puertos: se une a la red de Docker de Nginx Proxy Manager y NPM le
llega por nombre. Busca cómo se llama esa red en el VPS:

```bash
docker network ls
```

Si no es `npm_default`, crea un `.env` a partir de `.env.example` con el nombre correcto, y
levanta el servicio:

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

## 4. Actualizar los datos

```bash
uv run --group export python export.py
rsync -av data/ vps:~/cic-dashboard/data/
ssh vps 'cd ~/cic-dashboard && docker compose restart'
```

El reinicio es necesario porque `st.cache_data` guarda la instantánea en memoria.

## Calidad

```bash
uv run ruff check .
uv run ruff format --check .
```
