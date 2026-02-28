#!/usr/bin/env python3
"""
model_deploy.py — Deploy BERT/DistilBERT model to a SageMaker endpoint.
FIXED: Corrected model_fn signature and subfolder loading logic.
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

# ── FIXED INFERENCE CODE ─────────────────────────────────────────────────────
INFERENCE_CODE = '''import json
import os
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)

_MODEL     = None
_TOKENIZER = None
_DEVICE    = None
_LABEL_MAP = {0: "REAL", 1: "FAKE"}

def model_fn(model_dir, context=None):
    """
    Called ONCE at container startup. 
    Accepts 'context' to match HuggingFace container signature.
    """
    global _MODEL, _TOKENIZER, _DEVICE

    _DEVICE = torch.device("cpu")

    # Match the subfolder structure confirmed in your artifact list
    tokenizer_path = os.path.join(model_dir, "bert_tokenizer")
    model_path     = os.path.join(model_dir, "bert_model")

    print(f"model_dir contents : {os.listdir(model_dir)}")
    print(f"Loading tokenizer  : {tokenizer_path}")
    print(f"Loading model      : {model_path}")

    # local_files_only=True ensures we don't try to download from HF Hub
    _TOKENIZER = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    _MODEL     = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    
    _MODEL.to(_DEVICE)
    _MODEL.eval()

    print("✅ Model and tokenizer loaded successfully from subfolders.")
    return {"model": _MODEL, "tokenizer": _TOKENIZER, "device": _DEVICE}

def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    # Support both standard HF 'inputs' and custom 'text' keys
    text = data.get("text") or data.get("inputs") or ""
    if not text:
        raise ValueError("JSON body must contain a non-empty 'text' or 'inputs' field.")
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

    prob_list = probs[0].tolist()
    fake_prob = prob_list[1] if len(prob_list) > 1 else prob_list[0]
    real_prob = prob_list[0]

    return {
        "predictions":   [pred_idx],
        "probabilities": [prob_list],
        "prediction":    pred_idx,
        "label":         _LABEL_MAP.get(pred_idx, "REAL"),
        "fake_prob":     round(fake_prob, 6),
        "real_prob":     round(real_prob, 6),
        "confidence":    round(float(max(prob_list)), 6),
    }

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
'''

# ── DEPLOYMENT LOGIC ─────────────────────────────────────────────────────────

def find_latest_model(s3_client, bucket_name, project_name):
    print("🔍 Searching for trained model...")
    prefix = f"{project_name}/models/"
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" in response:
        model_files = [obj for obj in response["Contents"] if obj["Key"].endswith("model.tar.gz") and "/output/" in obj["Key"]]
        if model_files:
            latest = max(model_files, key=lambda x: x["LastModified"])
            print(f"✅ Found model: s3://{bucket_name}/{latest['Key']}")
            return f"s3://{bucket_name}/{latest['Key']}"
    raise FileNotFoundError("No trained model found in S3.")

def find_model_root(base_dir: str) -> str:
    marker_files = {"config.json", "pytorch_model.bin", "model.safetensors", "vocab.txt"}
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != "code"]
        if set(files) & marker_files or any(d in ["bert_model", "bert_tokenizer"] for d in dirs):
            return root
    return base_dir

def prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri):
    print("\n📦 Preparing model artifact...")
    model_key = model_uri.replace(f"s3://{bucket_name}/", "")
    s3_client.download_file(bucket_name, model_key, "model.tar.gz")

    os.makedirs("./model_extracted", exist_ok=True)
    with tarfile.open("model.tar.gz", "r:gz") as tar:
        tar.extractall("./model_extracted")
    
    actual_root = find_model_root("./model_extracted")
    staging = "./model_staging"
    if os.path.exists(staging): shutil.rmtree(staging)
    shutil.copytree(actual_root, staging)

    os.makedirs(f"{staging}/code", exist_ok=True)
    with open(f"{staging}/code/inference.py", "w") as f:
        f.write(INFERENCE_CODE)
    with open(f"{staging}/code/requirements.txt", "w") as f:
        f.write("transformers==4.37.0\ntorch==2.1.0\n")

    with tarfile.open("deploy_model.tar.gz", "w:gz") as tar:
        tar.add(staging, arcname=".")

    deploy_key = f"{project_name}/models/deploy/model.tar.gz"
    s3_client.upload_file("deploy_model.tar.gz", bucket_name, deploy_key)

    shutil.rmtree("./model_extracted")
    shutil.rmtree(staging)
    os.remove("model.tar.gz")
    os.remove("deploy_model.tar.gz")
    return f"s3://{bucket_name}/{deploy_key}"

def create_sagemaker_model(model_name, model_data_url, sagemaker_role, aws_region):
    sm = boto3.client("sagemaker", region_name=aws_region)
    image_uri = f"763104351884.dkr.ecr.{aws_region}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=sagemaker_role,
    )
    print("✅ SageMaker model resource created")

def create_endpoint_config(config_name, model_name, aws_region):
    sm = boto3.client("sagemaker", region_name=aws_region)
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
        }],
    )

def create_or_update_endpoint(endpoint_name, config_name, aws_region):
    sm = boto3.client("sagemaker", region_name=aws_region)
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

def wait_for_endpoint(endpoint_name, aws_region):
    sm = boto3.client("sagemaker", region_name=aws_region)
    while True:
        status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status == "InService": break
        time.sleep(30)

def main(args):
    s3_client = boto3.client("s3", region_name=args.region)
    model_uri = find_latest_model(s3_client, args.bucket, args.project)
    deploy_uri = prepare_model_for_deployment(s3_client, args.bucket, args.project, model_uri)
    
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name, config_name = f"{args.project}-m-{ts}", f"{args.project}-c-{ts}"

    create_sagemaker_model(model_name, deploy_uri, args.role, args.region)
    create_endpoint_config(config_name, model_name, args.region)
    create_or_update_endpoint(args.endpoint, config_name, args.region)
    wait_for_endpoint(args.endpoint, args.region)

    with open(args.output_file, "w") as f:
        json.dump({"endpoint_name": args.endpoint}, f)
    
    with open("deployment.env", "w") as f:
        f.write(f"ENDPOINT_NAME={args.endpoint}\nMODEL_TYPE=bert\n")
    print("✅ Deployment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output-file", default="deployment_info.json")
    main(parser.parse_args())
