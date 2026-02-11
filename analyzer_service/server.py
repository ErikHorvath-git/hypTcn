import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("analyzer")
APP = FastAPI(title="hypTcn Analyzer")


class TCNModel:
    def __init__(self):
        LOG.info("loading placeholder TCN model into memory")

    def infer(self, data: bytes) -> dict:
        score = min(1.0, len(data) / 1024.0)
        status = "suspicious" if score > 0.5 else "clean"
        return {"anomaly_score": round(score, 4), "status": status}


MODEL = TCNModel()


class AnalyzeResponse(BaseModel):
    anomaly_score: float
    status: str


@APP.on_event("startup")
def ready():
    LOG.info("analyzer service ready")


@APP.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty payload")

    result = MODEL.infer(data)
    return result


@APP.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("analyzer_service.server:APP", uds="/tmp/hyptcn.sock", log_level="info")
