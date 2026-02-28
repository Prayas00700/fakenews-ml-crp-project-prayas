#!/usr/bin/env python3
"""
model_deploy.py — Deploy BERT/DistilBERT model to a SageMaker endpoint.

KEY FIXES vs original:
  1. Use CPU image (ml.m5.large has no GPU). Original used GPU image
     → model.to('cuda') hung indefinitely on first inference call.
  2. inference.py injected here is now correct (binary fake/real, both
     payload shapes, model loaded ONCE in model_fn).
  3. SAGEMAKER_SUBMIT_DIRECTORY must be '/opt/ml/model/code' (not /code).
  4. requirements.txt inside the tarball now matches the container Python.
"""
import os
import sys
import json
import argparse
import boto3
import tarfile
import shutil
from datetime import datetime
import time


INFERENCE_CODE = '''import json
import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
)

_MODEL     = None
_TOKENIZER = None
_DEVICE    = None
_LABEL_MAP = {0: "REAL", 1: "FAKE"}


def model_fn(model_dir):
    """Called ONCE at container startup — load model into memory."""
    global _MODEL, _TOKENIZER, _DEVICE

    # ml.m5.large has NO GPU — always use CPU
    _DEVICE = torch.device("cpu")

    try:
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(model_dir)
        _MODEL     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    except Exception:
        _TOKENIZER = BertTokenizer.from_pretrained(model_dir)
        _MODEL     = BertForSequenceClassification.from_pretrained(model_dir)

    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return {"model": _MODEL, "tokenizer": _TOKENIZER, "device": _DEVICE}


def input_fn(request_body, content_type="application/json"):
    """Accept {"text": "..."} OR {"inputs": "..."} (CLI compat)."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    text = data.get("text") or data.get("inputs") or ""
    if not text:
        raise ValueError("JSON body must contain a non-empty 'text' field.")
    return text


def predict_fn(input_text, model_dict):
    tokenizer = model_dict["tokenizer"]
    model     = model_dict["model"]
    device    = model_dict["device"]

    inputs = tokenizer(
        input_text, truncation=True, padding=True,
        max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs  = model(**inputs)
        probs    = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = outputs.logits.argmax(-1).item()

    prob_list = probs[0].tolist()           # [real_prob, fake_prob]
    fake_prob = prob_list[1] if len(prob_list) > 1 else prob_list[0]
    real_prob = prob_list[0]

    return {
        # Legacy keys (api.py original format)
        "predictions":   [pred_idx],
        "probabilities": [prob_list],
        # Flat keys (convenient for direct consumers)
        "prediction":    pred_idx,
        "label":         _LABEL_MAP.get(pred_idx, "REAL"),
        "fake_prob":     round(fake_prob, 6),
        "real_prob":     round(real_prob, 6),
        "confidence":    round(float(max(prob_list)), 6),
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
'''


def find_latest_model(s3_client, bucket_name, project_name):
    print("🔍 Searching for trained model...")
    prefix = f"{project_name}/models/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" in response:
            model_files = [
                obj for obj in response["Contents"]
                if obj["Key"].endswith("model.tar.gz") and "/output/" in obj["Key"]
            ]
            if model_files:
                latest = max(model_files, key=lambda x: x["LastModified"])
                model_uri = f"s3://{bucket_name}/{latest['Key']}"
                print(f"✅ Found model: {model_uri}")
                return model_uri
    except Exception as e:
        print(f"❌ Error searching S3: {e}")
    raise FileNotFoundError("No trained model found in S3.")


def prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri):
    print("\n📦 Preparing model artifact (injecting fixed inference.py)...")
    model_key = model_uri.replace(f"s3://{bucket_name}/", "")
    s3_client.download_file(bucket_name, model_key, "model.tar.gz")

    os.makedirs("./model", exist_ok=True)
    with tarfile.open("model.tar.gz", "r:gz") as tar:
        tar.extractall("./model")
    os.remove("model.tar.gz")

    # SageMaker HuggingFace DLC looks for inference code in <model_dir>/code/
    os.makedirs("./model/code", exist_ok=True)

    with open("./model/code/inference.py", "w") as f:
        f.write(INFERENCE_CODE)

    # Pin versions compatible with the CPU container
    with open("./model/code/requirements.txt", "w") as f:
        f.write("transformers==4.37.0\n")
        f.write("torch==2.1.0\n")

    print("🗜️  Repackaging artifact...")
    with tarfile.open("deploy_model.tar.gz", "w:gz") as tar:
        tar.add("./model", arcname=".")

    deploy_key = f"{project_name}/models/deploy/model.tar.gz"
    s3_client.upload_file("deploy_model.tar.gz", bucket_name, deploy_key)

    shutil.rmtree("./model")
    os.remove("deploy_model.tar.gz")
    print(f"✅ Artifact uploaded to s3://{bucket_name}/{deploy_key}")
    return f"s3://{bucket_name}/{deploy_key}"


def create_sagemaker_model(model_name, model_data_url, sagemaker_role, aws_region):
    print(f"\n🤖 Creating SageMaker model: {model_name}")
    sm = boto3.client("sagemaker", region_name=aws_region)

    # ── FIX: use CPU image — ml.m5.large has no GPU ──────────────────────────
    # Original used:  huggingface-pytorch-inference:2.1.0-...-gpu-...
    # GPU image on a CPU instance causes model loading to hang → timeout
    image_uri = (
        f"763104351884.dkr.ecr.{aws_region}.amazonaws.com/"
        f"huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
    )

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM":            "inference.py",
                # FIX: correct path inside the container
                "SAGEMAKER_SUBMIT_DIRECTORY":   "/opt/ml/model/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            },
        },
        ExecutionRoleArn=sagemaker_role,
    )
    print("✅ SageMaker model resource created")


def create_endpoint_config(config_name, model_name, aws_region):
    print(f"⚙️  Creating endpoint config: {config_name}")
    sm = boto3.client("sagemaker", region_name=aws_region)
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":           "AllTraffic",
            "ModelName":             model_name,
            "InstanceType":          "ml.m5.large",   # CPU — matches CPU image above
            "InitialInstanceCount":  1,
        }],
    )
    print("✅ Endpoint config created")


def create_or_update_endpoint(endpoint_name, config_name, aws_region):
    print(f"🚀 Deploying endpoint: {endpoint_name}")
    sm = boto3.client("sagemaker", region_name=aws_region)
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print("📝 Updating existing endpoint...")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        print("🆕 Creating new endpoint...")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)


def wait_for_endpoint(endpoint_name, aws_region):
    print("⏳ Waiting for endpoint to become InService...")
    sm = boto3.client("sagemaker", region_name=aws_region)
    while True:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"   Status: {status}")
        if status == "InService":
            print("✅ Endpoint is live!")
            break
        if status in ["Failed", "RollingBack"]:
            reason = response.get("FailureReason", "unknown")
            raise RuntimeError(f"Deployment failed: {status} — {reason}")
        time.sleep(30)


def main(args):
    s3_client = boto3.client("s3", region_name=args.region)

    model_uri   = find_latest_model(s3_client, args.bucket, args.project)
    deploy_uri  = prepare_model_for_deployment(s3_client, args.bucket, args.project, model_uri)

    timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name  = f"{args.project}-model-{timestamp}"
    config_name = f"{args.project}-config-{timestamp}"

    create_sagemaker_model(model_name, deploy_uri, args.role, args.region)
    create_endpoint_config(config_name, model_name, args.region)
    create_or_update_endpoint(args.endpoint, config_name, args.region)
    wait_for_endpoint(args.endpoint, args.region)

    deployment_info = {
        "endpoint_name": args.endpoint,
        "model_name":    model_name,
        "region":        args.region,
        "timestamp":     datetime.now().isoformat(),
        "image":         "huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04",
    }
    with open(args.output_file, "w") as f:
        json.dump(deployment_info, f, indent=2)

    with open("deployment.env", "w") as f:
        f.write(f"ENDPOINT_NAME={args.endpoint}\n")
        f.write(f"MODEL_TYPE=bert\n")

    print(f"\n✅ Deployment complete. Info saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",      required=True)
    parser.add_argument("--project",     required=True)
    parser.add_argument("--endpoint",    required=True)
    parser.add_argument("--role",        required=True)
    parser.add_argument("--region",      required=True)
    parser.add_argument("--output-file", default="deployment_info.json")
    args = parser.parse_args()
    main(args)
