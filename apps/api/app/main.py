from fastapi import FastAPI

app = FastAPI(title="Maintainer's Copilot API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}