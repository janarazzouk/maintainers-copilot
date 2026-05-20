FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -r pyproject.toml

COPY app ./app
COPY training ./training
COPY models ./models

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]