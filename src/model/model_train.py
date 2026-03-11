"""
src/model/model_train.py
────────────────────────
Fake News Detection — SageMaker Training Job Launcher

Mirrors the heart-disease pipeline pattern:
  - Accepts --bucket, --prefix, --model-type, --epochs, --batch-size CLI args
    (all passed from GitHub Actions env vars / inputs)
  - Uploads the appropriate training script to S3 code/ prefix
  - Creates a SageMaker Training Job via boto3 directly (no SDK estimator magic)
  - Waits for completion, streams status updates
  - Downloads + logs final metrics
  - Writes model_data.env for downstream GitHub Actions jobs

Supported MODEL_TYPE values:
  nb   → TF-IDF + Complement Naive Bayes  (ml.m5.xlarge, CPU)
  bert → BERT fine-tuning                 (ml.g4dn.xlarge, GPU)

Usage:
    python src/model/model_train.py \
        --bucket   my-bucket \
        --prefix   fake-news \
        --model-type bert \
        --epochs 2 \
        --batch-size 8
"""

import os
import sys
import json
import time
import logging
import argparse
import textwrap
import tempfile
from pathlib import Path
from datetime import datetime

import boto3

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dep import (
    AWS_REGION,
    SAGEMAKER_ROLE_ARN,
    SAGEMAKER_INSTANCE,
    SAGEMAKER_INSTANCE_TRAIN,
    # Hyperparameter defaults
    TFIDF_MAX_FEATURES,
    BERT_MODEL_NAME, BERT_MAX_LEN, BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Docker images  (public SageMaker managed containers)
# ─────────────────────────────────────────────────────────────────────────────

SKLEARN_IMAGE   = f"683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
PT_GPU_IMAGE    = f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker-v1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Inline training scripts (uploaded to S3 code/ prefix and run in SageMaker)
# ─────────────────────────────────────────────────────────────────────────────

NB_TRAIN_SCRIPT = textwrap.dedent("""\
    \"\"\"Naive Bayes training script — runs inside SageMaker Training container.\"\"\"
    import os, re, pickle, json, argparse
    import pandas as pd, numpy as np
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    for r in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(r, quiet=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features",    type=int,   default=20000)
    parser.add_argument("--bucket",          type=str,   required=True)
    parser.add_argument("--prefix",          type=str,   default="fake-news")
    parser.add_argument("--model-dir",       type=str,   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir",        type=str,   default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    args = parser.parse_args()

    # Load processed data
    import glob
    parquet_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    csv_files     = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
    elif csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(f"No parquet or CSV found in {args.data_dir}")

    X = df["clean_tfidf"].values
    y = df["label_id"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    tfidf = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    Xtr_v = tfidf.fit_transform(Xtr)
    Xte_v = tfidf.transform(Xte)

    nb = ComplementNB()
    nb.fit(Xtr_v, ytr)

    y_pred  = nb.predict(Xte_v)
    y_prob  = nb.predict_proba(Xte_v)[:, 1]
    report  = classification_report(yte, y_pred, output_dict=True)
    auc     = roc_auc_score(yte, y_prob)

    metrics = {
        "model_type":      "nb",
        "auc":             float(auc),
        "accuracy":        float(report["accuracy"]),
        "f1_fake":         float(report["1"]["f1-score"]),
        "precision_fake":  float(report["1"]["precision"]),
        "recall_fake":     float(report["1"]["recall"]),
    }
    # SageMaker metric regex format
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1-Fake: {metrics['f1_fake']:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(args.model_dir, "nb_model.pkl"), "wb") as f:
        pickle.dump(nb, f)
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saved to {args.model_dir}")
""")

BERT_TRAIN_SCRIPT = textwrap.dedent("""\
    \"\"\"BERT fine-tuning script — runs inside SageMaker Training container.\"\"\"
    import os, json, pickle, argparse, glob
    import pandas as pd, numpy as np, torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import BertTokenizer, BertForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-model",  type=str,   default="bert-base-uncased")
    parser.add_argument("--max-len",     type=int,   default=256)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--bucket",      type=str,   required=True)
    parser.add_argument("--prefix",      type=str,   default="fake-news")
    parser.add_argument("--model-dir",   type=str,   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir",    type=str,   default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    args = parser.parse_args()

    class FakeNewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts; self.labels = labels
            self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts.iloc[idx], truncation=True,
                padding="max_length", max_length=self.max_len, return_tensors="pt"
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(self.labels.iloc[idx], dtype=torch.long),
            }

    # Load processed data
    parquet_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    csv_files     = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
    elif csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(f"No parquet or CSV found in {args.data_dir}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text_raw"], df["label_id"],
        test_size=0.2, stratify=df["label_id"], random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_dl  = DataLoader(FakeNewsDataset(X_train, y_train, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True)
    test_dl   = DataLoader(FakeNewsDataset(X_test,  y_test,  tokenizer, args.max_len), batch_size=args.batch_size)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train(); total_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward(); optimizer.step()
            total_loss += out.loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={total_loss:.4f}")

    model.eval(); preds = []; probs = []; true_labels = []
    with torch.no_grad():
        for batch in test_dl:
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            p   = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p); preds.extend((p >= 0.5).astype(int))
            true_labels.extend(batch["labels"].cpu().numpy())

    report  = classification_report(true_labels, preds, output_dict=True)
    metrics = {
        "model_type":      "bert",
        "auc":             float(roc_auc_score(true_labels, probs)),
        "accuracy":        float(report["accuracy"]),
        "f1_fake":         float(report["1"]["f1-score"]),
        "precision_fake":  float(report["1"]["precision"]),
        "recall_fake":     float(report["1"]["recall"]),
    }
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1-Fake: {metrics['f1_fake']:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.model_dir, "bert_model"))
    tokenizer.save_pretrained(os.path.join(args.model_dir, "bert_tokenizer"))
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"BERT model saved to {args.model_dir}")
""")

SCRIPT_MAP = {
    "nb":   ("train_nb.py",   NB_TRAIN_SCRIPT),
    "bert": ("train_bert.py", BERT_TRAIN_SCRIPT),
}

# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper scripts — used as sagemaker_program entry point.
#
# The SageMaker DLC bootstrap calls sagemaker_program first, which lets it
# run `pip install -r requirements.txt` beforehand. The wrapper then
# subprocess-calls the real training script so that all freshly installed
# packages (transformers, etc.) are importable in a clean Python process.
# Without this two-process pattern, packages installed by the bootstrap are
# sometimes not visible to the already-running Python interpreter.
# ─────────────────────────────────────────────────────────────────────────────

NB_WRAPPER_SCRIPT = """import subprocess, sys
subprocess.check_call([sys.executable, "train_nb.py"] + sys.argv[1:])
"""

BERT_WRAPPER_SCRIPT = """import subprocess, sys
subprocess.check_call([sys.executable, "train_bert.py"] + sys.argv[1:])
"""

WRAPPER_MAP = {
    "nb":   ("train_nb_wrapper.py",   NB_WRAPPER_SCRIPT),
    "bert": ("train_bert_wrapper.py", BERT_WRAPPER_SCRIPT),
}

# Per-model requirements bundled into sourcedir.tar.gz alongside the training script.
#
# HOW IT WORKS (critical):
#   The SageMaker DLC bootstrap runner reads requirements.txt from the extracted
#   sourcedir and runs `pip install -r requirements.txt` BEFORE calling
#   sagemaker_program. This only works because we set sagemaker_submit_directory
#   AND sagemaker_program as hyperparameters — the DLC bootstrap handles it.
#   Raw boto3 create_training_job alone does NOT process requirements.txt.
#
# TORCH RULE:
#   torch MUST be pinned here to match the container version (2.3.1 for the
#   2.3.0-gpu-py311 DLC). This tells pip "already satisfied" and ensures
#   transformers can resolve torch correctly without reinstalling it.
REQUIREMENTS_MAP = {
    "nb": (
        "nltk\n"
        "pandas\n"
        "numpy\n"
        "scikit-learn\n"
        "pyarrow\n"
    ),
    "bert": (
        # Pin torch to the version pre-installed in the 2.3.0-gpu-py311 DLC.
        # pip will see "already satisfied" and skip reinstalling, but the pin
        # ensures transformers resolves its torch dependency correctly.
        "torch==2.3.1\n"
        "transformers==4.44.2\n"
        "accelerate\n"
        "pandas\n"
        "numpy\n"
        "scikit-learn\n"
        "pyarrow\n"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# S3 / SageMaker helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_clients():
    """Return boto3 s3 and sagemaker clients."""
    s3  = boto3.client("s3",        region_name=AWS_REGION)
    sm  = boto3.client("sagemaker", region_name=AWS_REGION)
    return s3, sm


def upload_training_script(s3: boto3.client, script_content: str, script_name: str,
                            bucket: str, prefix: str, model_type: str = "bert") -> str:
    """
    Package wrapper + real training script + requirements.txt into sourcedir.tar.gz
    and upload to S3.

    Layout inside the tar:
        train_<type>_wrapper.py   ← sagemaker_program entry point
        train_<type>.py           ← real training logic
        requirements.txt          ← pip-installed by DLC bootstrap before entry point

    The DLC bootstrap (triggered by sagemaker_submit_directory + sagemaker_program
    hyperparameters) runs:
        1. pip install -r requirements.txt
        2. python train_<type>_wrapper.py <hyperparams>

    The wrapper then subprocess-calls train_<type>.py in a fresh Python process,
    ensuring all newly installed packages are importable.
    """
    import tarfile, io

    requirements_content = REQUIREMENTS_MAP.get(model_type, "")
    wrapper_name, wrapper_content = WRAPPER_MAP[model_type]

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # ── Wrapper script (sagemaker_program) ───────────────────────────────
        wrapper_bytes = wrapper_content.encode("utf-8")
        wrap_info     = tarfile.TarInfo(name=wrapper_name)
        wrap_info.size = len(wrapper_bytes)
        tar.addfile(wrap_info, io.BytesIO(wrapper_bytes))

        # ── Real training script ─────────────────────────────────────────────
        script_bytes = script_content.encode("utf-8")
        info         = tarfile.TarInfo(name=script_name)
        info.size    = len(script_bytes)
        tar.addfile(info, io.BytesIO(script_bytes))

        # ── requirements.txt ─────────────────────────────────────────────────
        if requirements_content:
            req_bytes = requirements_content.encode("utf-8")
            req_info  = tarfile.TarInfo(name="requirements.txt")
            req_info.size = len(req_bytes)
            tar.addfile(req_info, io.BytesIO(req_bytes))
            logger.info("Bundled requirements.txt into sourcedir for model_type=%s", model_type)

    buf.seek(0)
    s3_key = f"{prefix}/code/sourcedir_{script_name.replace('.py','')}.tar.gz"
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=buf.read(),
        ContentType="application/gzip",
    )
    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info("Training script uploaded: %s", s3_uri)
    logger.info("  Wrapper   : %s", wrapper_name)
    logger.info("  Script    : %s", script_name)
    logger.info("  sourcedir : %s", s3_uri)
    return s3_uri, wrapper_name   # return wrapper_name — this is sagemaker_program


def build_hyperparameters(model_type: str, bucket: str, prefix: str, epochs: int, batch_size: int) -> dict:
    """Build the HyperParameters dict for the SageMaker training job."""
    base = {
        "bucket":      bucket,
        "prefix":      prefix,
        "epochs":      str(epochs),
        "batch-size":  str(batch_size),
    }
    if model_type == "nb":
        base["max-features"] = str(TFIDF_MAX_FEATURES)
        base["sagemaker_program"] = "train_nb_wrapper.py"   # wrapper → DLC installs reqs first
    elif model_type == "bert":
        base.update({
            "bert-model":  BERT_MODEL_NAME,
            "max-len":     str(BERT_MAX_LEN),
            "batch-size":  str(BERT_BATCH_SIZE),
            "epochs":      str(BERT_EPOCHS),
            "lr":          str(BERT_LR),
            "sagemaker_program": "train_bert_wrapper.py",   # wrapper → DLC installs reqs first
        })
    return base


def start_training_job(
    sm: boto3.client,
    job_name: str,
    model_type: str,
    bucket: str,
    prefix: str,
    epochs: int,
    batch_size: int,
    sourcedir_uri: str,
    script_name: str,
) -> str:
    """Create and start a SageMaker Training Job, return its ARN."""
    # Choose image and instance
    if model_type == "nb":
        image    = SKLEARN_IMAGE
        instance = SAGEMAKER_INSTANCE     # CPU is fine for NB
    else:   # bert
        image    = PT_GPU_IMAGE
        instance = SAGEMAKER_INSTANCE_TRAIN

    hyperparams = build_hyperparameters(model_type, bucket, prefix, epochs, batch_size)
    hyperparams["sagemaker_submit_directory"] = sourcedir_uri

    logger.info("Using container image: %s", image)
    logger.info("Using instance type  : %s", instance)

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=SAGEMAKER_ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage":                    image,
            "TrainingInputMode":                "File",
            "EnableSageMakerMetricsTimeSeries": True,
            "MetricDefinitions": [
                {"Name": "Accuracy", "Regex": r"Accuracy: ([0-9.]+)"},
                {"Name": "AUC",      "Regex": r"AUC: ([0-9.]+)"},
                {"Name": "F1-Fake",  "Regex": r"F1-Fake: ([0-9.]+)"},
            ],
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType":             "S3Prefix",
                        "S3Uri":                  f"s3://{bucket}/{prefix}/data/processed/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{bucket}/{prefix}/models/output/",
        },
        ResourceConfig={
            "InstanceType":    instance,
            "InstanceCount":   1,
            "VolumeSizeInGB":  30,
        },
        EnableManagedSpotTraining=True,
        StoppingCondition={
            "MaxRuntimeInSeconds": 36000,   # 10 hours max (sufficient for 10 epochs)
            "MaxWaitTimeInSeconds": 43200,  # 12 hours max wait for spot
        },
        HyperParameters=hyperparams,
        Tags=[
            {"Key": "Project",    "Value": prefix},
            {"Key": "ModelType",  "Value": model_type},
            {"Key": "Stage",      "Value": "Training"},
        ],
    )

    logger.info("SageMaker Training Job started: %s", job_name)
    logger.info("  ARN       : %s", response["TrainingJobArn"])
    logger.info("  Image     : %s", image)
    logger.info("  Instance  : %s", instance)
    logger.info("  Epochs    : %d", epochs)
    logger.info("  Batch size: %d", batch_size)
    return response["TrainingJobArn"]


def wait_for_training_job(sm: boto3.client, job_name: str) -> dict:
    """Poll the training job until completion; return the describe response."""
    logger.info("Waiting for training job: %s", job_name)
    logger.info("This may take 10–60 minutes depending on model type...")

    last_status     = None
    last_secondary  = None
    start_time      = time.time()

    while True:
        response       = sm.describe_training_job(TrainingJobName=job_name)
        status         = response["TrainingJobStatus"]
        secondary      = response.get("SecondaryStatus", "")

        if status != last_status or secondary != last_secondary:
            elapsed = int(time.time() - start_time)
            logger.info("[%ds] Status: %s  →  %s", elapsed, status, secondary)
            last_status    = status
            last_secondary = secondary

        if status == "Completed":
            elapsed = int(time.time() - start_time)
            logger.info("Training completed in %ds (%.1f minutes)", elapsed, elapsed / 60)

            if "FinalMetricDataList" in response:
                logger.info("Final Metrics:")
                for m in response["FinalMetricDataList"]:
                    logger.info("  %s: %.4f", m["MetricName"], m["Value"])
            break

        elif status in ("Failed", "Stopped"):
            reason = response.get("FailureReason", "Unknown")
            logger.error("Training job failed: %s", reason)
            sys.exit(1)

        time.sleep(30)

    return response


def download_and_log_metrics(sm: boto3.client, s3: boto3.client, job_name: str, bucket: str, prefix: str) -> None:
    """Download model.tar.gz, extract metrics.json, and log it."""
    response  = sm.describe_training_job(TrainingJobName=job_name)
    model_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
    logger.info("Model artifact: %s", model_uri)

    # Parse bucket / key from the S3 URI
    parts      = model_uri.replace("s3://", "").split("/", 1)
    art_bucket = parts[0]
    art_key    = parts[1]

    import tarfile, io
    obj    = s3.get_object(Bucket=art_bucket, Key=art_key)
    buf    = io.BytesIO(obj["Body"].read())

    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        try:
            metrics_member = tar.getmember("metrics.json")
            metrics_file   = tar.extractfile(metrics_member)
            metrics        = json.loads(metrics_file.read())
            logger.info("\nTraining Metrics:")
            for k, v in metrics.items():
                logger.info("  %s: %s", k, v)
        except KeyError:
            logger.warning("metrics.json not found in model artifact")

        try:
            history_member = tar.getmember("training_history.json")
            history_file   = tar.extractfile(history_member)
            history        = json.loads(history_file.read())
            logger.info("\nTraining History (last 5 epochs):")
            if "loss" in history:
                losses = history["loss"]
                accs   = history.get("accuracy", [None] * len(losses))
                for i in range(max(0, len(losses) - 5), len(losses)):
                    logger.info("  Epoch %d: Loss=%.4f, Acc=%.4f", i + 1, losses[i], accs[i] or 0)
        except KeyError:
            pass   # NB and BERT don't produce training_history.json


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SageMaker Training Job for Fake News Detection")
    parser.add_argument("--bucket",      required=True,          help="S3 bucket name")
    parser.add_argument("--prefix",      default="fake-news",    help="Project prefix in S3")
    parser.add_argument("--model-type",  default="bert",
                        choices=["nb", "bert"],                  help="Model type to train")
    parser.add_argument("--epochs",      type=int, default=2,    help="Training epochs")
    parser.add_argument("--batch-size",  type=int, default=8,    help="Batch size")
    parser.add_argument("--no-wait",     action="store_true",    help="Submit job and exit immediately without waiting")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FAKE NEWS — MODEL TRAINING (SageMaker)")
    logger.info("=" * 70)
    logger.info("  Bucket     : %s", args.bucket)
    logger.info("  Prefix     : %s", args.prefix)
    logger.info("  Model type : %s", args.model_type)
    logger.info("  Epochs     : %d", args.epochs)
    logger.info("  Batch size : %d", args.batch_size)
    logger.info("=" * 70)

    if not SAGEMAKER_ROLE_ARN:
        logger.error("SAGEMAKER_ROLE_ARN env var is not set. Cannot launch training job.")
        sys.exit(1)

    s3, sm    = get_clients()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name  = f"fake-news-{args.model_type}-{timestamp}"

    # ── 1. Upload training script to S3 ──────────────────────────────────────
    script_name, script_content = SCRIPT_MAP[args.model_type][0], SCRIPT_MAP[args.model_type][1]
    sourcedir_uri, script_name  = upload_training_script(
        s3, script_content, script_name, args.bucket, args.prefix,
        model_type=args.model_type,
    )

    # ── 2. Start training job ─────────────────────────────────────────────────
    start_training_job(
        sm, job_name, args.model_type, args.bucket, args.prefix,
        args.epochs, args.batch_size, sourcedir_uri, script_name,
    )

    # ── 3. Write job name immediately so it's available even in --no-wait mode ─
    Path("model_data.env").write_text(
        f"MODEL_DATA=pending\n"
        f"MODEL_TYPE={args.model_type}\n"
        f"TRAINING_JOB_NAME={job_name}\n"
    )
    logger.info("Wrote model_data.env (job submitted, model URI pending)")

    if args.no_wait:
        logger.info("=" * 70)
        logger.info("TRAINING JOB SUBMITTED (--no-wait mode)")
        logger.info("=" * 70)
        logger.info("  Job name : %s", job_name)
        logger.info("  Region   : %s", AWS_REGION)
        logger.info("  Monitor  : SageMaker → Training → Training jobs → %s", job_name)
        logger.info("=" * 70)
        return

    # ── 4. Wait for completion ────────────────────────────────────────────────
    response  = wait_for_training_job(sm, job_name)
    model_uri = response["ModelArtifacts"]["S3ModelArtifacts"]

    # ── 5. Download + log metrics ─────────────────────────────────────────────
    download_and_log_metrics(sm, s3, job_name, args.bucket, args.prefix)

    # ── 6. Verify model artifacts ─────────────────────────────────────────────
    logger.info("Verifying model artifacts in S3...")
    objects = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"{args.prefix}/models/output/")
    for obj in objects.get("Contents", []):
        logger.info("  %s", obj["Key"])

    # ── 7. Update env file with real model URI ────────────────────────────────
    Path("model_data.env").write_text(
        f"MODEL_DATA={model_uri}\n"
        f"MODEL_TYPE={args.model_type}\n"
        f"TRAINING_JOB_NAME={job_name}\n"
    )
    logger.info("Updated model_data.env with final model URI")

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("  Job name   : %s", job_name)
    logger.info("  Model URI  : %s", model_uri)
    logger.info("  Region     : %s", AWS_REGION)
    logger.info("=" * 70)
    logger.info("View in AWS Console:")
    logger.info("  SageMaker → Training → Training jobs → %s", job_name)


if __name__ == "__main__":
    main()
