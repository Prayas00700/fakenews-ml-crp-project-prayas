"""
api.py — FastAPI backend that proxies requests to the SageMaker endpoint.

Your BERT model returns this shape:
{
  "label":       "FAKE" | "REAL",
  "prediction":  1 | 0,
  "fake_prob":   0.9963,
  "real_prob":   0.0037,
  "confidence":  0.9963,
  "predictions": [1],
  "probabilities": [[0.0037, 0.9963]],
  "probabilities_named": {"REAL": 0.0037, "FAKE": 0.9963}
}
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import boto3
from botocore.config import Config
import json
import pandas as pd
from io import StringIO
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "prayas-ml-crp-project-name-endpoint")
AWS_REGION    = os.getenv("AWS_REGION", "ap-south-1")

logger.info(f"API v2.0 — endpoint: {ENDPOINT_NAME}  region: {AWS_REGION}")

config           = Config(read_timeout=120, retries={"max_attempts": 2})
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=config)


class ArticleInput(BaseModel):
    text:  str
    title: Optional[str] = ""


def invoke_endpoint(text: str) -> dict:
    """Invoke SageMaker endpoint. Always returns a flat dict."""
    payload = {"text": text}   # your inference.py expects "text" key

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        raw  = response["Body"].read().decode("utf-8")
        result = json.loads(raw)

        # Handle wrapped list format: ['{...json...}', 'application/json']
        if isinstance(result, list) and len(result) == 2 and isinstance(result[0], str):
            result = json.loads(result[0])

        logger.info("SageMaker result: %s", result)
        return result

    except Exception as e:
        logger.error("SageMaker invocation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.95: return "Very High"
    if confidence >= 0.80: return "High"
    if confidence >= 0.60: return "Medium"
    return "Low"


def _interpretation(label: str, fake_prob: float) -> str:
    pct = fake_prob * 100
    if label == "FAKE":
        if pct >= 95:
            return f"Strong indicators of fabricated content ({pct:.1f}% fake probability). Do not share without verification."
        if pct >= 80:
            return f"Likely misinformation ({pct:.1f}% fake probability). Cross-check with trusted sources."
        return f"Possible misinformation ({pct:.1f}% fake probability). Investigate before sharing."
    else:
        real_pct = (1 - fake_prob) * 100
        if real_pct >= 95:
            return f"Strong signals of authentic content ({real_pct:.1f}% real probability). Still verify key claims."
        return f"Leans toward authentic ({real_pct:.1f}% real probability). Cross-check with primary sources."


def _build_response(r: dict, title: str = "") -> dict:
    """
    Normalise raw endpoint output into the schema the frontend expects.
    Your model already returns all keys — this just ensures nothing is missing.
    """
    label     = r.get("label", "REAL").upper()
    fake_prob = float(r.get("fake_prob", 0.0))
    real_prob = float(r.get("real_prob", 1 - fake_prob))
    confidence_raw = float(r.get("confidence", max(fake_prob, real_prob)))

    # The frontend's app.js uses: prediction, probability, label, confidence, interpretation
    return {
        "title":          title,
        "prediction":     1 if label == "FAKE" else 0,
        "label":          label,
        "probability":    round(fake_prob if label == "FAKE" else real_prob, 4),
        "fake_prob":      round(fake_prob, 4),
        "real_prob":      round(real_prob, 4),
        "confidence":     _confidence_label(confidence_raw),
        "confidence_raw": round(confidence_raw, 4),
        "interpretation": _interpretation(label, fake_prob),
    }


@app.get("/health")
def health():
    return {"status": "ok", "endpoint": ENDPOINT_NAME, "region": AWS_REGION}


@app.post("/predict")
async def predict_single(article: ArticleInput):
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")

    combined = f"{article.title} {article.text}".strip() if article.title else article.text.strip()
    r = invoke_endpoint(combined)
    return _build_response(r, title=article.title or "")


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
        raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")

    if "title" not in df.columns:
        df["title"] = ""

    results = []
    for idx, row in df.iterrows():
        try:
            combined = f"{row['title']} {row['text']}".strip()
            r    = invoke_endpoint(combined)
            resp = _build_response(r, title=str(row.get("title", "")))
            results.append({"row": int(idx), **resp})
        except Exception as e:
            logger.warning("Skipping row %d: %s", idx, e)

    fake_count = sum(1 for r in results if r["prediction"] == 1)
    high_conf  = sum(1 for r in results if r["confidence"] in ("High", "Very High"))

    return {
        "total_articles": len(results),
        "predictions":    results,
        "summary": {
            "fake_detected":    fake_count,
            "real_detected":    len(results) - fake_count,
            "high_confidence":  high_conf,
        },
    }
