"""
src/deployment/model_deploy.py
────────────────────────────────
Creates (or updates) a SageMaker real-time inference endpoint for the
trained fake-news detection model.

Supports BERT (HuggingFace) model type.

Usage:
    python src/deployment/model_deploy.py

Env vars:
    MODEL_DATA              — s3://… artifact path
    MODEL_TYPE              — bert
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

    SCRIPTS = {"bert": BERT_INFERENCE_SCRIPT}

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(SCRIPTS[model_type])
        script_path = tmp.name

    kwargs = dict(
        model_data=model_data,
        role=SAGEMAKER_ROLE_ARN,
        entry_point=script_path,
        sagemaker_session=session,
    )

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