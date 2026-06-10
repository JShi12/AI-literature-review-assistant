FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/lit_review_assistant/app.py", "--server.address=0.0.0.0"]
