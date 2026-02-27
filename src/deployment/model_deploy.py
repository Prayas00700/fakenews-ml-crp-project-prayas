#!/usr/bin/env python3
"""
Model Deployment Script (Boto3 Version)
Deploys BERT models to SageMaker using the low-level SDK to avoid 
dependency issues in CI/CD environments.
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

def find_latest_model(s3_client, bucket_name, project_name):
    print("🔍 Searching for trained model...")
    prefix = f"{project_name}/models/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            # Look for model artifacts in the output folder
            model_files = [obj for obj in response['Contents'] if obj['Key'].endswith('model.tar.gz') and '/output/' in obj['Key']]
            if model_files:
                latest = max(model_files, key=lambda x: x['LastModified'])
                model_uri = f"s3://{bucket_name}/{latest['Key']}"
                print(f"✅ Found model: {model_uri}")
                return model_uri
    except Exception as e:
        print(f"❌ Error: {e}")
    raise FileNotFoundError("No trained model found.")

def prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri):
    print("\n📦 Preparing model for deployment (injecting inference logic)...")
    model_key = model_uri.replace(f"s3://{bucket_name}/", "")
    s3_client.download_file(bucket_name, model_key, 'model.tar.gz')
    
    os.makedirs('./model', exist_ok=True)
    with tarfile.open('model.tar.gz', 'r:gz') as tar:
        tar.extractall('./model')
    os.remove('model.tar.gz')
    
    # Create the code directory for SageMaker Inference Toolkit
    os.makedirs('./model/code', exist_ok=True)
    
    print("📝 Creating inference script...")
    inference_code = '''import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def model_fn(model_dir):
    # Hugging Face containers look for artifacts in the root model_dir
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return {'model': model, 'tokenizer': tokenizer, 'device': device}

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data['text']
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    device = model_dict['device']
    
    inputs = tokenizer(input_data, truncation=True, padding=True, 
                      max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = outputs.logits.argmax(-1).item()
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return {
        'prediction': label_map[pred],
        'confidence': float(probs[0][pred]),
        'probabilities': {label_map[i]: float(probs[0][i]) for i in range(3)}
    }

def output_fn(prediction, accept):
    return json.dumps(prediction), 'application/json'
'''
    with open('./model/code/inference.py', 'w') as f:
        f.write(inference_code)
    
    with open('./model/code/requirements.txt', 'w') as f:
        f.write('transformers==4.44.2\n')
    
    print("🗜️  Repackaging artifact...")
    with tarfile.open('deploy_model.tar.gz', 'w:gz') as tar:
        tar.add('./model', arcname='.')
    
    deploy_key = f"{project_name}/models/deploy/model.tar.gz"
    s3_client.upload_file('deploy_model.tar.gz', bucket_name, deploy_key)
    
    shutil.rmtree('./model')
    os.remove('deploy_model.tar.gz')
    return f"s3://{bucket_name}/{deploy_key}"

def create_sagemaker_model(model_name, model_data_url, sagemaker_role, aws_region):
    print(f"\n🤖 Creating SageMaker model: {model_name}")
    sm = boto3.client('sagemaker', region_name=aws_region)
    
    # Use the official Hugging Face Inference DLC URI for BERT/Transformers
    # Account 763104351884 is the AWS Deep Learning Container account
    image_uri = f"763104351884.dkr.ecr.{aws_region}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
    
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data_url,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                'HF_TASK': 'text-classification',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
            }
        },
        ExecutionRoleArn=sagemaker_role
    )
    print("✅ Model resource created")

def create_endpoint_config(config_name, model_name, aws_region):
    print(f"⚙️  Creating config: {config_name}")
    sm = boto3.client('sagemaker', region_name=aws_region)
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InstanceType': 'ml.m5.large',
            'InitialInstanceCount': 1
        }]
    )
    print("✅ Endpoint config created")

def create_or_update_endpoint(endpoint_name, config_name, aws_region):
    print(f"🚀 Deploying/Updating endpoint: {endpoint_name}")
    sm = boto3.client('sagemaker', region_name=aws_region)
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print("📝 Updating existing endpoint...")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        print("🆕 Creating new endpoint...")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

def wait_for_endpoint(endpoint_name, aws_region):
    print(f"⏳ Waiting for endpoint to become 'InService'...")
    sm = boto3.client('sagemaker', region_name=aws_region)
    while True:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        if status == 'InService':
            print("✅ Endpoint is live!")
            break
        elif status in ['Failed', 'RollingBack']:
            raise Exception(f"Deployment failed with status: {status}")
        time.sleep(30)

def main(args):
    s3_client = boto3.client('s3', region_name=args.region)
    
    # 1. Find the latest artifact from the training job
    model_uri = find_latest_model(s3_client, args.bucket, args.project)
    
    # 2. Package the model with our custom BERT inference logic
    deploy_uri = prepare_model_for_deployment(s3_client, args.bucket, args.project, model_uri)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{args.project}-model-{timestamp}"
    config_name = f"{args.project}-config-{timestamp}"
    
    # 3. Create SageMaker resources using Boto3
    create_sagemaker_model(model_name, deploy_uri, args.role, args.region)
    create_endpoint_config(config_name, model_name, args.region)
    create_or_update_endpoint(args.endpoint, config_name, args.region)
    
    # 4. Wait for completion
    wait_for_endpoint(args.endpoint, args.region)
    
    # 5. Output results
    deployment_info = {
        'endpoint_name': args.endpoint,
        'model_name': model_name,
        'region': args.region,
        'timestamp': datetime.now().isoformat()
    }
    with open(args.output_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--output-file', default='deployment_info.json')
    args = parser.parse_args()
    main(args)
