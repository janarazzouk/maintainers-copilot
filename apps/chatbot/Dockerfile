FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./pyproject.toml

RUN uv pip install --system -r pyproject.toml

COPY app ./app

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]