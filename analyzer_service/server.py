import logging

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from analyzer_service.model import TCNModel
from analyzer_service.processor import PageBuffer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("analyzer")
APP = FastAPI(title="hypTcn Analyzer")

MODEL = TCNModel()
BUFFER = PageBuffer()


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

    sequence = BUFFER.append(data)
    if sequence is None:
        raise HTTPException(status_code=202, detail="waiting for buffer")

    tensor = torch.from_numpy(sequence).unsqueeze(0)
    tensor = tensor.to(torch.float32)

    score = MODEL.infer(tensor)
    status = "suspicious" if score > 0.5 else "clean"
    LOG.info("anomaly score: %.4f", score)

    return AnalyzeResponse(anomaly_score=score, status=status)


@APP.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("analyzer_service.server:APP", uds="/tmp/hyptcn.sock", log_level="info")
