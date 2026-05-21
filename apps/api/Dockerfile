FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -r pyproject.toml

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]