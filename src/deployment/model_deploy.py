"""
src/deployment/model_deploy.py
────────────────────────────────
Creates (or updates) a SageMaker real-time inference endpoint for the
trained fake-news detection model.

Supports NB (SKLearn), LSTM (PyTorch), and BERT (HuggingFace) model types.
"""

import os
import sys
import json
import logging
import textwrap
import tempfile
from pathlib import Path

# Add project root to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.huggingface import HuggingFaceModel
# --- FIX: Added PyTorchModel to support your .pt artifact ---
from sagemaker.pytorch.model import PyTorchModel
from botocore.exceptions import ClientError

# Import your constants from dep.py

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

def get_sagemaker_session():
    boto_session = boto3.Session(region_name=AWS_REGION)
    return sagemaker.Session(boto_session=boto_session)

def deploy_model(model_data: str, model_type: str, session: sagemaker.Session) -> str:
    log.info("Deploying %s model from %s", model_type, model_data)

    if model_type == "nb":
        model = SKLearnModel(
            model_data=model_data,
            role=SAGEMAKER_ROLE_ARN,
            entry_point="src/deployment/inference_nb.py",
            framework_version="1.2-1",
            sagemaker_session=session,
        )
    elif model_type == "lstm":
        # --- FIX: Use PyTorchModel for .pt files ---
        model = PyTorchModel(
            model_data=model_data,
            role=SAGEMAKER_ROLE_ARN,
            entry_point="src/deployment/inference_lstm.py",
            framework_version="2.0.1",
            py_version="py310",
            sagemaker_session=session,
        )
    elif model_type == "bert":
        model = HuggingFaceModel(
            model_data=model_data,
            role=SAGEMAKER_ROLE_ARN,
            transformers_version="4.26",
            pytorch_version="1.13",
            py_version="py39",
            sagemaker_session=session,
        )
    else:
        log.error("Unknown MODEL_TYPE: %s", model_type)
        sys.exit(1)

    log.info("Creating/Updating endpoint: %s", ENDPOINT_NAME)
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        endpoint_name=ENDPOINT_NAME,
        update_endpoint=True,
    )
    log.info("Model deployed successfully to endpoint: %s", ENDPOINT_NAME)
    return predictor.endpoint_name

def smoke_test(endpoint_name: str) -> None:
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

def main() -> None:
    model_data = os.environ.get("MODEL_DATA", "")
    model_type = os.environ.get("MODEL_TYPE", MODEL_TYPE).lower()

    if not model_data:
        log.error("MODEL_DATA env var is not set.")
        sys.exit(1)

    session = get_sagemaker_session()
    endpoint_name = deploy_model(model_data, model_type, session)
    smoke_test(endpoint_name)

    Path("deployment.env").write_text(
        f"ENDPOINT_NAME={endpoint_name}\nMODEL_TYPE={model_type}\n"
    )
    log.info("Wrote deployment.env")

if __name__ == "__main__":
    main()
