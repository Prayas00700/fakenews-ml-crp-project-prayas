"""
api.py — FastAPI backend with SageMaker inference + KernelSHAP explainability.
v4.0: /explain endpoint uses KernelSHAP (model-agnostic, works with any black-box endpoint).

KernelSHAP vs LIME:
  - LIME: masks one word at a time, estimates linear model locally
  - KernelSHAP: uses Shapley values from game theory — averages contribution
    across ALL subsets of words, giving more accurate, consistent attribution.
  - Both work as black-box wrappers around the SageMaker endpoint.
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
import re
import numpy as np
import logging
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME             = os.getenv("SAGEMAKER_ENDPOINT", "prayas-ml-crp-project-name-endpoint")
AWS_REGION                = os.getenv("AWS_REGION", "ap-south-1")
REAL_CONFIDENCE_THRESHOLD = 0.85

# SHAP settings
# Max unique words to compute SHAP for (keeps runtime reasonable)
SHAP_MAX_WORDS  = 15
# Number of random coalition samples per word (higher = more accurate, slower)
SHAP_SAMPLES    = 20
EXPLANATION_TOP_N = 10

logger.info(f"API v4.0 — endpoint: {ENDPOINT_NAME}  region: {AWS_REGION}")

config            = Config(read_timeout=120, retries={"max_attempts": 2})
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=config)


class ArticleInput(BaseModel):
    text:  str
    title: Optional[str] = ""


# ── Core inference ────────────────────────────────────────────────────────────

def invoke_endpoint(text: str) -> dict:
    """Invoke SageMaker endpoint. Returns flat dict."""
    payload = {"text": text}
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        raw    = response["Body"].read().decode("utf-8")
        result = json.loads(raw)
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


def _apply_heuristic(label: str, fake_prob: float, real_prob: float):
    if label == "REAL" and real_prob < REAL_CONFIDENCE_THRESHOLD:
        return (
            "UNVERIFIED", 0,
            f"The model leans toward real ({real_prob*100:.1f}% real probability) but confidence "
            f"is below the verification threshold ({REAL_CONFIDENCE_THRESHOLD*100:.0f}%). "
            f"Treat with caution and verify independently.",
            True
        )
    return (label, 1 if label == "FAKE" else 0, None, False)


def _interpretation(label: str, fake_prob: float, real_prob: float) -> str:
    if label == "FAKE":
        pct = fake_prob * 100
        if pct >= 95:
            return f"Strong indicators of fabricated content ({pct:.1f}% fake probability). Do not share without verification."
        if pct >= 80:
            return f"Likely misinformation ({pct:.1f}% fake probability). Cross-check with trusted sources."
        return f"Possible misinformation ({pct:.1f}% fake probability). Investigate before sharing."
    else:
        pct = real_prob * 100
        if pct >= 95:
            return f"Strong signals of authentic content ({pct:.1f}% real probability). Still verify key claims."
        return f"Leans toward authentic ({pct:.1f}% real probability). Cross-check with primary sources."


def _build_response(r: dict, title: str = "") -> dict:
    raw_label  = r.get("label", "REAL").upper()
    fake_prob  = float(r.get("fake_prob", 0.0))
    real_prob  = float(r.get("real_prob", 1 - fake_prob))
    conf_raw   = float(r.get("confidence", max(fake_prob, real_prob)))

    final_label, prediction, heuristic_interp, triggered = _apply_heuristic(raw_label, fake_prob, real_prob)
    interpretation = heuristic_interp if triggered else _interpretation(raw_label, fake_prob, real_prob)
    conf_label   = "Medium" if triggered else _confidence_label(conf_raw)
    display_prob = real_prob if triggered else (fake_prob if final_label == "FAKE" else real_prob)

    return {
        "title":          title,
        "prediction":     prediction,
        "label":          final_label,
        "probability":    round(display_prob, 4),
        "fake_prob":      round(fake_prob, 4),
        "real_prob":      round(real_prob, 4),
        "confidence":     conf_label,
        "confidence_raw": round(conf_raw, 4),
        "interpretation": interpretation,
        "heuristic_flag": triggered,
    }


# ── KernelSHAP explainability ─────────────────────────────────────────────────

STOPWORDS = {
    "the","and","or","is","in","at","of","to","a","an","it","as","by",
    "be","was","are","for","on","that","this","with","from","not","but",
    "have","had","has","its","he","she","they","we","our","their","been",
    "will","would","could","should","also","said","says","about","after",
    "when","which","who","were","more","than","into","over","then",
    "them","these","those","such","even","both","each","some","just","very"
}


def _tokenize(text: str) -> list:
    """Extract meaningful words, preserving original casing."""
    return re.findall(r'\b[a-zA-Z]\w*\b', text)


def _build_masked_text(tokens: list, present_mask: list) -> str:
    """Reconstruct text keeping only tokens where present_mask[i] = True."""
    return " ".join(t for t, keep in zip(tokens, present_mask) if keep)


def _get_fake_prob(text: str) -> float:
    """Get fake probability for any text string, with neutral fallback."""
    if not text.strip():
        return 0.5
    try:
        r = invoke_endpoint(text)
        return float(r.get("fake_prob", 0.0))
    except Exception:
        return 0.5


def _shapley_kernel_weight(subset_size: int, total: int) -> float:
    """
    KernelSHAP weight for a coalition of given size.
    w(S) = (M-1) / (C(M,|S|) * |S| * (M - |S|))
    where M = total number of features (words).
    Coalitions of size 0 or M get weight infinity (handled separately).
    """
    from math import comb
    if subset_size == 0 or subset_size == total:
        return 1e10  # very large weight for boundary conditions
    return (total - 1) / (comb(total, subset_size) * subset_size * (total - subset_size))


def shap_explain(text: str, top_n: int = EXPLANATION_TOP_N) -> list:
    """
    KernelSHAP word-level attribution for a black-box text classifier.

    Algorithm:
      1. Tokenize text, deduplicate meaningful words → feature set F of size M
      2. For each word i in F:
         a. Sample `SHAP_SAMPLES` random coalitions S NOT containing i
         b. Compute f(S ∪ {i}) - f(S) for each coalition (marginal contribution)
         c. Weight each difference by the KernelSHAP weight w(|S|)
         d. SHAP value φᵢ = weighted average of marginal contributions
      3. Return top_n words by |φᵢ|

    Positive φ  → word increases fake probability → pushes toward FAKE
    Negative φ  → word decreases fake probability → pushes toward REAL

    This satisfies the Shapley axioms (efficiency, symmetry, dummy, additivity)
    giving theoretically grounded, consistent feature attributions.
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    # Deduplicate: pick first occurrence of each meaningful word
    seen = {}
    for i, token in enumerate(tokens):
        word_lower = token.lower()
        if len(word_lower) <= 2 or word_lower in STOPWORDS:
            continue
        if word_lower not in seen:
            seen[word_lower] = i

    word_indices = list(seen.values())    # indices into tokens list
    word_keys    = list(seen.keys())      # lowercase keys
    M = len(word_indices)

    if M == 0:
        return []

    # Limit to top SHAP_MAX_WORDS most central tokens (by position spread)
    if M > SHAP_MAX_WORDS:
        # Take every Nth word to cover the article evenly
        step = M // SHAP_MAX_WORDS
        word_indices = word_indices[::step][:SHAP_MAX_WORDS]
        word_keys    = word_keys[::step][:SHAP_MAX_WORDS]
        M = len(word_indices)

    logger.info("KernelSHAP: M=%d words, %d samples each", M, SHAP_SAMPLES)

    # Baseline: empty text (all words masked) and full text
    baseline_prob = _get_fake_prob("")   # f(∅)
    full_prob     = _get_fake_prob(text) # f(F)
    logger.info("KernelSHAP baseline=%.4f  full=%.4f", baseline_prob, full_prob)

    shap_values = np.zeros(M)
    shap_weights_sum = np.zeros(M)

    rng = np.random.default_rng(seed=42)

    for i in range(M):
        marginals = []
        weights   = []

        for _ in range(SHAP_SAMPLES):
            # Random coalition S of the OTHER M-1 words (not including word i)
            other_indices = [j for j in range(M) if j != i]
            # Random subset size for the coalition
            s_size = int(rng.integers(0, M))  # 0 to M-1
            if s_size > 0 and len(other_indices) > 0:
                chosen = rng.choice(other_indices, size=min(s_size, len(other_indices)), replace=False).tolist()
            else:
                chosen = []

            # Build masks: S without word i, and S with word i
            mask_without = [False] * len(tokens)
            mask_with    = [False] * len(tokens)

            for j in chosen:
                mask_without[word_indices[j]] = True
                mask_with[word_indices[j]]    = True
            mask_with[word_indices[i]] = True  # add word i

            text_without = _build_masked_text(tokens, mask_without)
            text_with    = _build_masked_text(tokens, mask_with)

            prob_without = _get_fake_prob(text_without)
            prob_with    = _get_fake_prob(text_with)

            marginal = prob_with - prob_without
            weight   = _shapley_kernel_weight(len(chosen), M)

            marginals.append(marginal)
            weights.append(weight)

        # Weighted average of marginal contributions
        total_w = sum(weights)
        if total_w > 0:
            shap_values[i] = sum(m * w for m, w in zip(marginals, weights)) / total_w
        else:
            shap_values[i] = 0.0

        logger.info("  word='%s' shap=%.4f", word_keys[i], shap_values[i])

    # Build result list
    results = []
    for i in range(M):
        results.append({
            "word":      tokens[word_indices[i]],   # original casing
            "score":     round(float(shap_values[i]), 4),
            "direction": "fake" if shap_values[i] > 0 else "real",
        })

    results.sort(key=lambda x: abs(x["score"]), reverse=True)
    return results[:top_n]


# ── Endpoints ─────────────────────────────────────────────────────────────────

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


@app.post("/explain")
async def explain_single(article: ArticleInput):
    """
    Returns prediction + KernelSHAP word-level explanation.
    Slower than /predict (~30–60s) — calls endpoint multiple times per word.
    """
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")

    combined = f"{article.title} {article.text}".strip() if article.title else article.text.strip()

    # Base prediction
    r        = invoke_endpoint(combined)
    response = _build_response(r, title=article.title or "")

    # KernelSHAP explanation
    logger.info("Running KernelSHAP explanation...")
    explanation = shap_explain(combined, top_n=EXPLANATION_TOP_N)

    response["explanation"] = explanation
    response["explanation_method"] = "KernelSHAP"
    response["explanation_note"] = (
        "SHAP values show each word's average marginal contribution to the fake probability, "
        "computed across random coalitions of other words. "
        "Positive values push toward FAKE, negative values push toward REAL."
    )
    return response


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
    unverified = sum(1 for r in results if r["label"] == "UNVERIFIED")
    high_conf  = sum(1 for r in results if r["confidence"] in ("High", "Very High"))

    return {
        "total_articles": len(results),
        "predictions":    results,
        "summary": {
            "fake_detected":   fake_count,
            "real_detected":   len(results) - fake_count - unverified,
            "unverified":      unverified,
            "high_confidence": high_conf,
        },
    }
