FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Solo las dependencias de la app: el grupo export (psycopg) se queda fuera.
# --no-build: solo wheels, ningún sdist corre su script de build en la imagen.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-build

COPY .streamlit ./.streamlit
COPY app.py advisor.py common.py dashscope_client.py idea_assistant.py search_limits.py semantic_search.py text.py ./
COPY views ./views

USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
