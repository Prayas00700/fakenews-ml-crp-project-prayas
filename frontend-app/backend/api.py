"""
api.py — FastAPI backend that proxies requests to the SageMaker endpoint.
FIXED: Corrected response parsing to handle dictionary output from BERT inference.
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

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", version="1.1.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Environment
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "gunjan-ml-crp-project-name-endpoint")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

logger.info(f"API v1.1 — endpoint: {ENDPOINT_NAME}  region: {AWS_REGION}")

# SageMaker Runtime Client with 60s timeout
config = Config(read_timeout=60, retries={'max_attempts': 3})
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=config)

class ArticleInput(BaseModel):
    text: str
    title: Optional[str] = ""

def invoke_endpoint(text: str):
    """
    Invokes the SageMaker endpoint and handles the response.
    FIXED: Unwraps the ['{json_string}', 'content_type'] list format.
    """
    payload = {"inputs": text}
    
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        # Read the response body
        raw_body = response["Body"].read().decode("utf-8")
        result = json.loads(raw_body)
        
        # --- NEW FIX: Handle the ['{...}', 'application/json'] list format ---
        if isinstance(result, list) and len(result) == 2 and isinstance(result[0], str):
            logger.info("Detected nested JSON string in list. Unwrapping...")
            result = json.loads(result[0])
        
        logger.info(f"SageMaker Parsed Result: {result}")
        
        # Scenario A: Result is now the flat dict
        if isinstance(result, dict):
            return result
            
        # Scenario B: Standard Hugging Face List format
        if isinstance(result, list) and len(result) > 0:
            top = max(result, key=lambda x: x.get("score", 0))
            return {
                "label": top.get("label"),
                "confidence": top.get("score"),
                "prediction": 1 if top.get("label") == "FAKE" else 0
            }
            
        return result

    except Exception as e:
        logger.error(f"Error invoking SageMaker: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

def _build_response(r: dict, title: str = ""):
    """
    Standardizes the response for the frontend regardless of model version.
    """
    # Map varying keys (confidence vs score, label vs prediction) to a fixed schema
    label = r.get("label", "REAL")
    
    # Logic: if 'prediction' is missing, determine it from the label
    prediction_val = r.get("prediction")
    if prediction_val is None:
        prediction_val = 1 if label == "FAKE" else 0

    return {
        "title": title,
        "prediction": int(prediction_val),
        "label": label,
        "score": float(r.get("confidence") or r.get("score") or 0.0),
        "fake_prob": float(r.get("fake_prob", 0.0)),
        "real_prob": float(r.get("real_prob", 0.0))
    }

@app.get("/health")
def health():
    """Checks if the API is up and the SageMaker endpoint is reachable."""
    try:
        # Ping endpoint with a tiny string to verify live status
        invoke_endpoint("health check")
        return {"status": "healthy", "endpoint": ENDPOINT_NAME}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict")
async def predict_single(article: ArticleInput):
    """Endpoint for single article analysis."""
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")

    combined_text = f"{article.title} {article.text}".strip()
    
    r = invoke_endpoint(combined_text)
    return _build_response(r, title=article.title)

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Endpoint for batch analysis via CSV upload."""
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
            detail="CSV must contain a 'text' column.",
        )
    
    if "title" not in df.columns:
        df["title"] = ""

    results = []
    for idx, row in df.iterrows():
        try:
            combined_text = f"{row['title']} {row['text']}".strip()
            r = invoke_endpoint(combined_text)
            resp = _build_response(r, title=str(row.get("title", "")))
            results.append({
                "row": int(idx),
                **resp,
            })
        except Exception as e:
            logger.warning(f"Skipping row {idx} due to error: {e}")

    return {
        "total_articles": len(results),
        "predictions": results,
        "summary": {
            "fake_detected": sum(1 for r in results if r["prediction"] == 1),
            "real_detected": sum(1 for r in results if r["prediction"] == 0),
        }
    }