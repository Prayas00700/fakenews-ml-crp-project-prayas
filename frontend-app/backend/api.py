from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import boto3
import json
import pandas as pd
import numpy as np
from io import StringIO
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "fake-news-endpoint")
AWS_REGION    = os.getenv("AWS_REGION", "ap-south-1")

logger.info(f"API initialized using endpoint: {ENDPOINT_NAME} in region: {AWS_REGION}")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class ArticleInput(BaseModel):
    title: str
    text:  str

class PredictionResponse(BaseModel):
    prediction:      int           # 0 = Real, 1 = Fake
    probability:     float         # Probability of being FAKE
    confidence:      str           # Low / Medium / High / Very High
    label:           str           # "FAKE" or "REAL"
    interpretation:  str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_confidence(probability: float) -> str:
    if probability < 0.4 or probability > 0.6:
        if abs(probability - 0.5) > 0.4:
            return "Very High"
        return "High"
    if abs(probability - 0.5) > 0.2:
        return "Medium"
    return "Low"


def get_interpretation(prediction: int, probability: float) -> str:
    confidence = get_confidence(probability)
    if prediction == 1:
        return f"This article shows strong indicators of being fabricated or misleading. {confidence} confidence ({probability:.1%})."
    else:
        return f"This article appears to be credible and factual. {confidence} confidence ({1 - probability:.1%})."


def invoke_endpoint(text: str) -> dict:
    """Call SageMaker endpoint with article text."""
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    payload = json.dumps({"text": text})
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )
    return json.loads(response["Body"].read().decode())


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict":       "/predict",
            "predict_batch": "/predict/batch",
            "predict_csv":   "/predict/csv",
            "health":        "/health",
        },
    }


@app.get("/health")
async def health_check():
    try:
        boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        return {"status": "healthy", "endpoint": ENDPOINT_NAME, "region": AWS_REGION}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(article: ArticleInput):
    """Predict whether a single article is fake or real."""
    try:
        combined_text = f"{article.title} {article.text}".strip()
        result        = invoke_endpoint(combined_text)

        prediction   = result["predictions"][0]
        probabilities = result["probabilities"][0]
        fake_prob    = probabilities[1] if len(probabilities) > 1 else probabilities[0]

        return PredictionResponse(
            prediction     = prediction,
            probability    = fake_prob,
            confidence     = get_confidence(fake_prob),
            label          = "FAKE" if prediction == 1 else "REAL",
            interpretation = get_interpretation(prediction, fake_prob),
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(articles: List[ArticleInput]):
    """Predict for a list of articles."""
    try:
        results = []
        for i, article in enumerate(articles):
            combined_text = f"{article.title} {article.text}".strip()
            result        = invoke_endpoint(combined_text)

            prediction    = result["predictions"][0]
            probabilities = result["probabilities"][0]
            fake_prob     = probabilities[1] if len(probabilities) > 1 else probabilities[0]

            results.append({
                "row":           i,
                "title":         article.title,
                "prediction":    prediction,
                "probability":   fake_prob,
                "confidence":    get_confidence(fake_prob),
                "label":         "FAKE" if prediction == 1 else "REAL",
                "interpretation": get_interpretation(prediction, fake_prob),
            })

        return {
            "total_articles": len(results),
            "predictions":    results,
            "summary": {
                "fake_detected": sum(1 for r in results if r["prediction"] == 1),
                "real_detected": sum(1 for r in results if r["prediction"] == 0),
                "high_confidence": sum(1 for r in results if r["confidence"] in ["High", "Very High"]),
            },
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict from a CSV file with 'title' and 'text' columns."""
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be CSV format")

        contents = await file.read()
        df       = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Support both column name styles
        if "title" not in df.columns and "text" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain at least a 'text' column (and optionally 'title')",
            )

        if "title" not in df.columns:
            df["title"] = ""

        results = []
        for idx, row in df.iterrows():
            combined_text = f"{row['title']} {row['text']}".strip()
            result        = invoke_endpoint(combined_text)

            prediction    = result["predictions"][0]
            probabilities = result["probabilities"][0]
            fake_prob     = probabilities[1] if len(probabilities) > 1 else probabilities[0]

            results.append({
                "row":           int(idx),
                "title":         str(row.get("title", ""))[:80],
                "prediction":    prediction,
                "probability":   fake_prob,
                "confidence":    get_confidence(fake_prob),
                "label":         "FAKE" if prediction == 1 else "REAL",
                "interpretation": get_interpretation(prediction, fake_prob),
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

    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
