"""
src/deployment/model_deploy.py
────────────────────────────────
Creates (or updates) a SageMaker real-time inference endpoint for the
trained fake-news detection model.

Supports NB (SKLearn), LSTM (TensorFlow), and BERT (HuggingFace) model types.

Usage:
    python src/deployment/model_deploy.py

Env vars:
    MODEL_DATA              — s3://… artifact path
    MODEL_TYPE              — nb | lstm | bert
    SAGEMAKER_ENDPOINT_NAME — name of the endpoint to create/update
"""

import os
import sys
import json
import logging
import textwrap
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.huggingface import HuggingFaceModel
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dep import (
    AWS_REGION,
    S3_BUCKET,
    PROJECT_NAME,
    SAGEMAKER_ROLE_ARN,
    ENDPOINT_NAME,
    ENDPOINT_INSTANCE,
    MODEL_TYPE,
)

S3_ARTIFACTS_PREFIX = os.environ.get("S3_ARTIFACTS_PREFIX", f"{PROJECT_NAME}/artifacts/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Inline inference handlers (one per model type)
# These are uploaded as part of the SageMaker model package.
# ─────────────────────────────────────────────────────────────────────────────

NB_INFERENCE_SCRIPT = textwrap.dedent("""
import os, re, pickle, json
import numpy as np, pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for r in ("punkt","punkt_tab","stopwords","wordnet"):
    nltk.download(r, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def basic_clean(text):
    import re
    text=str(text)
    text=re.sub(r"http\\S+|www\\.\\S+"," ",text)
    text=re.sub(r"\\S+@\\S+"," ",text)
    text=re.sub(r"<.*?>"," ",text)
    text=re.sub(r"[\\r\\n]+"," ",text)
    text=re.sub(r"\\s+"," ",text).strip()
    return text

def clean_for_tfidf(text):
    text=basic_clean(text).lower()
    text=re.sub(r"[^a-z\\s]"," ",text)
    tokens=nltk.word_tokenize(text)
    tokens=[t for t in tokens if t not in stop_words and len(t)>1]
    tokens=[lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def model_fn(model_dir):
    with open(os.path.join(model_dir,"tfidf_vectorizer.pkl"),"rb") as f: tfidf=pickle.load(f)
    with open(os.path.join(model_dir,"nb_model.pkl"),         "rb") as f: nb=pickle.load(f)
    return {"tfidf": tfidf, "nb": nb}

def input_fn(request_body, content_type):
    data=json.loads(request_body)
    return data.get("text", data.get("inputs",""))

def predict_fn(text, model):
    clean=clean_for_tfidf(text)
    X=model["tfidf"].transform([clean])
    proba=model["nb"].predict_proba(X)[0]
    return {"label":"fake" if proba[1]>=0.5 else "real","fake_prob":float(proba[1]),"real_prob":float(proba[0])}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
""")

LSTM_INFERENCE_SCRIPT = textwrap.dedent("""
import os, re, pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 300

def basic_clean(text):
    import re
    text=str(text)
    text=re.sub(r"http\\S+|www\\.\\S+"," ",text)
    text=re.sub(r"<.*?>"," ",text)
    text=re.sub(r"[\\r\\n]+"," ",text)
    text=re.sub(r"\\s+"," ",text).strip()
    return text.lower()

def model_fn(model_dir):
    model=tf.keras.models.load_model(os.path.join(model_dir,"lstm_model"))
    with open(os.path.join(model_dir,"tokenizer.pkl"),"rb") as f: tokenizer=pickle.load(f)
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, content_type):
    return json.loads(request_body).get("text","")

def predict_fn(text, bundle):
    clean=basic_clean(text)
    seq=bundle["tokenizer"].texts_to_sequences([clean])
    pad=pad_sequences(seq,maxlen=MAX_LEN,padding="post",truncating="post")
    fake_prob=float(bundle["model"].predict(pad,verbose=0)[0][0])
    return {"label":"fake" if fake_prob>=0.5 else "real","fake_prob":fake_prob,"real_prob":1-fake_prob}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
""")

BERT_INFERENCE_SCRIPT = textwrap.dedent("""
import os, json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def model_fn(model_dir):
    tok   = BertTokenizer.from_pretrained(os.path.join(model_dir,"bert_tokenizer"))
    model = BertForSequenceClassification.from_pretrained(os.path.join(model_dir,"bert_model"))
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"model": model, "tokenizer": tok, "device": device}

def input_fn(request_body, content_type):
    return json.loads(request_body).get("text","")

def predict_fn(text, bundle):
    enc=bundle["tokenizer"](text,return_tensors="pt",truncation=True,padding=True,max_length=256)
    enc={k:v.to(bundle["device"]) for k,v in enc.items()}
    with torch.no_grad():
        out=bundle["model"](**enc)
        probs=torch.softmax(out.logits,dim=1)[0].cpu().tolist()
    fake_prob=probs[1]
    return {"label":"fake" if fake_prob>=0.5 else "real","fake_prob":fake_prob,"real_prob":probs[0]}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
""")


# ─────────────────────────────────────────────────────────────────────────────
# Deployment helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_sagemaker_session():
    return sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))


def endpoint_exists(sm_client, endpoint_name: str) -> bool:
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError:
        return False


def deploy_model(model_data: str, model_type: str, session: sagemaker.Session) -> str:
    sm_client  = boto3.client("sagemaker", region_name=AWS_REGION)
    exists     = endpoint_exists(sm_client, ENDPOINT_NAME)
    action_str = "Updating" if exists else "Creating"
    log.info("%s endpoint: %s", action_str, ENDPOINT_NAME)

    SCRIPTS = {"nb": NB_INFERENCE_SCRIPT, "lstm": LSTM_INFERENCE_SCRIPT, "bert": BERT_INFERENCE_SCRIPT}

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(SCRIPTS[model_type])
        script_path = tmp.name

    kwargs = dict(
        model_data=model_data,
        role=SAGEMAKER_ROLE_ARN,
        entry_point=script_path,
        sagemaker_session=session,
    )

    if model_type == "nb":
        sm_model = SKLearnModel(framework_version="1.2-1", **kwargs)
    elif model_type == "lstm":
        sm_model = TensorFlowModel(framework_version="2.13", **kwargs)
    else:
        sm_model = HuggingFaceModel(transformers_version="4.26", pytorch_version="1.13", py_version="py39", **kwargs)

    predictor = sm_model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        endpoint_name=ENDPOINT_NAME,
        update_endpoint=exists,
        wait=True,
    )

    log.info("Endpoint live: %s", ENDPOINT_NAME)
    return predictor.endpoint_name


def smoke_test(endpoint_name: str) -> None:
    """Send a quick prediction to verify the endpoint is healthy."""
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    test_payload = json.dumps({"text": "Breaking: Scientists confirm miraculous cure with zero side effects found!"})
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=test_payload,
    )
    result = json.loads(response["Body"].read().decode())
    log.info("Smoke test result: %s", result)
    assert "label" in result, "Smoke test failed — response missing 'label' key"
    log.info("Smoke test passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    model_data = os.environ.get("MODEL_DATA", "")
    model_type = os.environ.get("MODEL_TYPE", MODEL_TYPE).lower()

    if not model_data:
        log.error("MODEL_DATA env var is not set.")
        sys.exit(1)

    session = get_sagemaker_session()
    endpoint_name = deploy_model(model_data, model_type, session)
    smoke_test(endpoint_name)

    # Write endpoint name for monitoring job
    Path("deployment.env").write_text(
        f"ENDPOINT_NAME={endpoint_name}\nMODEL_TYPE={model_type}\n"
    )
    log.info("Deployment complete.")


if __name__ == "__main__":
    main()
