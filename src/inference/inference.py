"""
src/inference/inference.py
──────────────────────────
Utility functions for calling the deployed SageMaker endpoint from
any Python application.
"""

import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)


def get_runtime_client(region: str | None = None):
    return boto3.client(
        "sagemaker-runtime",
        region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def predict(
    text: str,
    endpoint_name: str | None = None,
    threshold: float = 0.5,
    region: str | None = None,
) -> dict:
    """
    Call the SageMaker endpoint with a news article text.

    Returns a dict like:
        {
            "label":     "fake" | "real",
            "fake_prob": 0.92,
            "real_prob": 0.08,
            "threshold": 0.5,
        }
    """
    endpoint_name = endpoint_name or os.environ.get(
        "SAGEMAKER_ENDPOINT_NAME", "fake-news-detector-endpoint"
    )
    client = get_runtime_client(region)

    payload = json.dumps({"text": text})
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read().decode())
        result["threshold"] = threshold
        # Re-apply threshold in case it was changed by caller
        result["label"] = "fake" if result.get("fake_prob", 0) >= threshold else "real"
        return result
    except ClientError as exc:
        log.error("Endpoint invocation failed: %s", exc)
        raise


def batch_predict(texts: list[str], **kwargs) -> list[dict]:
    """Run predict() on a list of texts."""
    return [predict(t, **kwargs) for t in texts]
