"""
api.py — FastAPI backend that proxies requests to the SageMaker endpoint.

FIXES applied vs original:
  1. invoke_endpoint() now supports BOTH response shapes:
       - New:  {"predictions":[0], "probabilities":[[0.1,0.9]]}  (legacy api.py expected)
       - Old:  {"prediction":0, "fake_prob":0.9, "real_prob":0.1} (flat / inference.py direct)
     So it works with the old deployed model AND the fixed inference.py.
  2. Added read_timeout=60 on boto3 client to prevent silent hangs.
  3. /health now actually invokes the endpoint with a tiny payload to confirm
     it is truly live (not just reachable).
  4. ArticleInput.title is now Optional so CSV rows without titles work.
  5. Cleaner error messages surfaced to the frontend.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import boto3
from botocore.config import Config
import json
import pandas as pd
from io import StringIO
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "gunjan-ml-crp-project-name-endpoint")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "ap-south-1"))

logger.info(f"API v1.1 — endpoint: {ENDPOINT_NAME}  region: {AWS_REGION}")

# ── Boto3 client with explicit timeouts (prevents silent container hangs) ─────
_BOTO_CONFIG = Config(
    region_name=AWS_REGION,
    read_timeout=60,          # seconds to wait for SageMaker response
    connect_timeout=10,
    retries={"max_attempts": 2, "mode": "standard"},
)


def get_runtime():
    return boto3.client("sagemaker-runtime", config=_BOTO_CONFIG)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ArticleInput(BaseModel):
    title: Optional[str] = ""
    text:  str


class PredictionResponse(BaseModel):
    prediction:     int
    probability:    float    # P(FAKE)
    confidence:     str
    label:          str      # "FAKE" or "REAL"
    interpretation: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_confidence(probability: float) -> str:
    """Map fake probability to a human confidence label."""
    delta = abs(probability - 0.5)
    if delta > 0.4:
        return "Very High"
    if delta > 0.2:
        return "High"
    if delta > 0.1:
        return "Medium"
    return "Low"


def get_interpretation(prediction: int, probability: float) -> str:
    conf = get_confidence(probability)
    if prediction == 1:
        return (
            f"This article shows strong indicators of being fabricated or misleading. "
            f"{conf} confidence ({probability:.1%})."
        )
    return (
        f"This article appears to be credible and factual. "
        f"{conf} confidence ({1 - probability:.1%})."
    )


def invoke_endpoint(text: str) -> dict:
    """
    Call SageMaker and normalise the response into a flat dict with keys:
        pred_idx  (int)   0=REAL, 1=FAKE
        fake_prob (float)
        real_prob (float)
    Works with BOTH the old and new inference.py response shapes.
    """
    runtime = get_runtime()
    payload = json.dumps({"text": text})

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )
    except Exception as exc:
        logger.error(f"SageMaker InvokeEndpoint failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"SageMaker endpoint unreachable: {str(exc)}",
        )

    raw = json.loads(response["Body"].read().decode())
    logger.debug(f"Raw endpoint response: {raw}")

    # ── Shape A: new inference.py — flat keys ─────────────────────────────
    if "fake_prob" in raw:
        return {
            "pred_idx":  int(raw.get("prediction", 0)),
            "fake_prob": float(raw["fake_prob"]),
            "real_prob": float(raw.get("real_prob", 1 - raw["fake_prob"])),
        }

    # ── Shape B: legacy — {"predictions":[0], "probabilities":[[r,f]]} ───
    if "predictions" in raw and "probabilities" in raw:
        pred_idx  = int(raw["predictions"][0])
        prob_list = raw["probabilities"][0]
        fake_prob = prob_list[1] if len(prob_list) > 1 else prob_list[0]
        real_prob = prob_list[0]
        return {
            "pred_idx":  pred_idx,
            "fake_prob": float(fake_prob),
            "real_prob": float(real_prob),
        }

    # ── Shape C: HuggingFace default container format ─────────────────────
    # [{"label": "FAKE", "score": 0.97}, ...]
    if isinstance(raw, list) and raw and "label" in raw[0]:
        top = max(raw, key=lambda x: x["score"])
        fake_prob = top["score"] if top["label"].upper() == "FAKE" else 1 - top["score"]
        pred_idx  = 1 if top["label"].upper() == "FAKE" else 0
        return {
            "pred_idx":  pred_idx,
            "fake_prob": float(fake_prob),
            "real_prob": float(1 - fake_prob),
        }

    # ── Unknown shape — log and raise ─────────────────────────────────────
    logger.error(f"Unrecognised endpoint response shape: {raw}")
    raise HTTPException(
        status_code=500,
        detail=f"Unexpected response format from SageMaker endpoint: {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}",
    )


def _build_response(r: dict, title: str = "") -> dict:
    """Turn a normalised invoke_endpoint() result into the API response shape."""
    pred_idx  = r["pred_idx"]
    fake_prob = r["fake_prob"]
    return {
        "prediction":     pred_idx,
        "probability":    fake_prob,
        "confidence":     get_confidence(fake_prob),
        "label":          "FAKE" if pred_idx == 1 else "REAL",
        "interpretation": get_interpretation(pred_idx, fake_prob),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message":  "Fake News Detection API",
        "version":  "1.1.0",
        "endpoint": ENDPOINT_NAME,
        "region":   AWS_REGION,
    }


@app.get("/health")
async def health_check():
    """
    Actually invokes the endpoint with a minimal payload.
    Returns 200 only when the model responds correctly.
    """
    try:
        result = invoke_endpoint("health check")
        return {
            "status":   "healthy",
            "endpoint": ENDPOINT_NAME,
            "region":   AWS_REGION,
            "test_prediction": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Endpoint unhealthy: {str(exc)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(article: ArticleInput):
    combined_text = f"{article.title} {article.text}".strip()
    if not combined_text:
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")
    r = invoke_endpoint(combined_text)
    return _build_response(r, title=article.title or "")


@app.post("/predict/batch")
async def predict_batch(articles: List[ArticleInput]):
    if not articles:
        raise HTTPException(status_code=400, detail="Article list is empty.")

    results = []
    for i, article in enumerate(articles):
        combined_text = f"{article.title} {article.text}".strip()
        r = invoke_endpoint(combined_text)
        resp = _build_response(r, title=article.title or "")
        results.append({"row": i, "title": article.title or "", **resp})

    return {
        "total_articles": len(results),
        "predictions":    results,
        "summary": {
            "fake_detected":   sum(1 for r in results if r["prediction"] == 1),
            "real_detected":   sum(1 for r in results if r["prediction"] == 0),
            "high_confidence": sum(1 for r in results if r["confidence"] in ["High", "Very High"]),
        },
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file.")

    if "text" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'text' column. Optional: 'title'.",
        )
    if "title" not in df.columns:
        df["title"] = ""

    results = []
    for idx, row in df.iterrows():
        combined_text = f"{row['title']} {row['text']}".strip()
        r    = invoke_endpoint(combined_text)
        resp = _build_response(r, title=str(row.get("title", "")))
        results.append({
            "row":   int(idx),
            "title": str(row.get("title", ""))[:80],
            **resp,
        })

    return {
        "total_articles": len(results),
        "predictions":    results,
        "summary": {
            "fake_detected":   sum(1 for r in results if r["prediction"] == 1),
            "real_detected":   sum(1 for r in results if r["prediction"] == 0),
            "high_confidence": sum(1 for r in results if r["confidence"] in ["High", "Very High"]),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
