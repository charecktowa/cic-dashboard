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
  conteos por año, laboratorio, rol, revista, término o persona investigadora. La única
  excepción son los títulos de artículos y congresos del personal investigador. No escribe
  ids ni datos de estudiantes.
- La app solo lee `data/`. No tiene conexión a la base de datos ni credenciales, y no ofrece
  tablas de títulos ni botones de descarga.
- Los términos que aparecen en menos de tres títulos se descartan.
- Los títulos visibles son de artículos y congresos del personal investigador: los del
  catálogo del CIC, con año y tipo, y los de [OpenAlex](https://openalex.org) (CC0), con
  enlace a su DOI cuando lo tienen. Ningún título de tesis (tampoco las que indexa OpenAlex)
  ni de proyectos.

Lo que una gráfica muestra llega al navegador: los valores del tooltip se pueden leer y la
barra de Plotly guarda un PNG. Que todo esté agregado antes es lo que lo vuelve inofensivo.

## Estructura

```
export.py        base de datos → data/*.parquet (solo en tu máquina)
text.py          cómo se parte un texto en términos; lo usan export.py y la app
common.py        carga de datos, etiquetas en español y utilidades de las páginas
app.py           navegación
tests/           pruebas con un catálogo inventado (sin base de datos)
views/           una página por archivo
.streamlit/      configuración: sin menú de desarrollador ni trazas de error para visitantes
Dockerfile       imagen solo con la app (sin export.py ni psycopg)
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

El grupo `export` añade `psycopg`; la app no lo necesita y la imagen no lo instala.
`data/` está en `.gitignore`: se copia al servidor aparte.

## 1b. Métricas y publicaciones de OpenAlex (opcional)

`openalex.py` vincula a cada persona investigadora con su perfil de
[OpenAlex](https://openalex.org) y trae citas, índice h, citas por año y sus trabajos más
citados. Los datos de OpenAlex son de dominio público (CC0). Corre en tu máquina, después de
`export.py`; el VPS nunca llama a OpenAlex.

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

## Seguridad

### Lo que ya trae el proyecto

- **No hay nada valioso que robar.** Sin base de datos, sin credenciales, sin datos de
  estudiantes: solo conteos derivados del sitio público del CIC.
- **Los Parquet no se sirven por la web.** `enableStaticServing = false` y `data/` no está en
  ninguna ruta pública; tampoco va dentro de la imagen, se monta aparte.
- **Contenedor endurecido**: usuario sin privilegios (uid 1000), disco de solo lectura,
  `cap_drop: ALL`, `no-new-privileges`, sin puertos publicados y 512 MB de memoria máxima.
- **Lo que escriben los visitantes no ejecuta nada**: no hay SQL, `eval` ni
  `unsafe_allow_html`, y el texto del buscador se reduce a letras, números y guiones.
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
rsync -av data/ vps:~/cic-dashboard/data/
ssh vps 'cd ~/cic-dashboard && docker compose restart'
```

El reinicio es necesario porque `st.cache_data` guarda la instantánea en memoria.

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
