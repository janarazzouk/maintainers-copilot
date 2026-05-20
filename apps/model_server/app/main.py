from fastapi import FastAPI

app = FastAPI(title="Maintainer's Copilot Model Server")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "model_server"}