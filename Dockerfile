FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts

RUN pip install --no-cache-dir -e . \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# PORT is not set locally (docker-compose) but is injected by most PaaS hosts (Render, ...),
# which assign it dynamically and require the app to listen on it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8501\")}/_stcore/health')" || exit 1

# Run migrations before serving so a fresh database is never missing the current schema --
# safe to repeat on every restart, alembic upgrade head is a no-op once already at head.
CMD ["sh", "-c", "alembic upgrade head && streamlit run src/lit_review_assistant/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
