"""
inference.py — SageMaker Inference Handler
Deployed inside model.tar.gz at: code/inference.py

ROOT CAUSE FIXES:
  1. Timeout: model_fn loaded model EVERY call. Now loaded once at startup.
  2. Payload mismatch: CLI used {"inputs":...} but this handler expects {"text":...}
  3. Output mismatch: api.py expected result["predictions"][0] and result["probabilities"][0]
     but this script returns {"prediction":..., "confidence":..., "probabilities":{...}}
     → Now returns BOTH old format + new format keys so either consumer works.
  4. 3-class model vs 2-class: model was trained for fake/real (2 classes) but
     inference.py used label_map with 3 classes. Fixed to binary.
  5. GPU image deployed but ml.m5.large has no GPU → model.to(device) hangs on 
     first call. Added explicit CPU fallback.
"""

import json
import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
)

# ── Globals: populated once by model_fn, reused across all invocations ────────
_MODEL = None
_TOKENIZER = None
_DEVICE = None
_LABEL_MAP = {0: "REAL", 1: "FAKE"}


def model_fn(model_dir: str):
    """
    Called ONCE at container startup. Load model + tokenizer into memory.
    SageMaker caches the return value and passes it to predict_fn.
    """
    global _MODEL, _TOKENIZER, _DEVICE

    # Always use CPU — endpoint uses ml.m5.large (no GPU)
    _DEVICE = torch.device("cpu")

    # Try DistilBERT first (what model_deploy.py packs), then fall back to BERT
    try:
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(model_dir)
        _MODEL = DistilBertForSequenceClassification.from_pretrained(model_dir)
    except Exception:
        _TOKENIZER = BertTokenizer.from_pretrained(model_dir)
        _MODEL = BertForSequenceClassification.from_pretrained(model_dir)

    _MODEL.to(_DEVICE)
    _MODEL.eval()

    return {"model": _MODEL, "tokenizer": _TOKENIZER, "device": _DEVICE}


def input_fn(request_body: str, content_type: str = "application/json") -> str:
    """
    Parse the incoming request body.
    Accepts BOTH {"text": "..."} AND {"inputs": "..."} for CLI/SDK compatibility.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    data = json.loads(request_body)

    # Support both payload shapes
    text = data.get("text") or data.get("inputs") or ""
    if not text:
        raise ValueError("Request JSON must contain a non-empty 'text' or 'inputs' field.")

    return text


def predict_fn(input_text: str, model_dict: dict) -> dict:
    """
    Run inference. Returns a dict that output_fn will serialise.
    """
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    device = model_dict["device"]

    inputs = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = outputs.logits.argmax(-1).item()

    num_classes = probs.shape[-1]

    # ── Build probability list (always length 2 for fake/real) ───────────────
    if num_classes == 2:
        prob_list = probs[0].tolist()          # [real_prob, fake_prob]
    else:
        # Model has unexpected number of classes — treat index 1 as fake
        prob_list = [probs[0][0].item(), probs[0][1].item()]

    fake_prob = prob_list[1] if len(prob_list) > 1 else prob_list[0]
    real_prob = prob_list[0]
    label = _LABEL_MAP.get(pred_idx, "REAL")

    return {
        # ── Keys expected by api.py (legacy format) ───────────────────────
        "predictions":   [pred_idx],
        "probabilities": [prob_list],

        # ── Clean flat keys (nicer for direct consumers / curl) ───────────
        "prediction":  pred_idx,
        "label":       label,
        "fake_prob":   round(fake_prob, 6),
        "real_prob":   round(real_prob, 6),
        "confidence":  float(max(prob_list)),

        # ── Raw probabilities dict ────────────────────────────────────────
        "probabilities_named": {
            "REAL": round(real_prob, 6),
            "FAKE": round(fake_prob, 6),
        },
    }


def output_fn(prediction: dict, accept: str = "application/json"):
    """Serialise the prediction dict as JSON."""
    return json.dumps(prediction), "application/json"
