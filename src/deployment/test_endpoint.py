#!/usr/bin/env python3
"""
test_endpoint.py — Diagnose your SageMaker endpoint from the CLI.

Usage:
    pip install boto3
    python test_endpoint.py \
        --endpoint gunjan-ml-crp-project-name-endpoint \
        --region   ap-south-1

This tells you EXACTLY which response shape the endpoint returns
so you know whether you need to redeploy or just fix api.py.
"""

import argparse
import boto3
import json
import sys


def test(endpoint_name: str, region: str):
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
    )

    test_cases = [
        {"text": "Breaking news: Scientists discover water on Mars."},
        {"inputs": "Breaking news: Scientists discover water on Mars."},   # HF default
    ]

    for payload in test_cases:
        body = json.dumps(payload)
        print(f"\n── Payload: {body}")
        try:
            resp = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=body,
            )
            raw = json.loads(resp["Body"].read().decode())
            print(f"   ✅ Response keys : {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}")
            print(f"   ✅ Full response : {json.dumps(raw, indent=4)}")
        except Exception as exc:
            print(f"   ❌ Error: {exc}")

    print("\nDone. Use the response keys above to confirm which api.py branch handles your endpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--region",   default="ap-south-1")
    args = parser.parse_args()
    test(args.endpoint, args.region)
