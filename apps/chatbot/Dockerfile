FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -r pyproject.toml

COPY app.py ./app.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]