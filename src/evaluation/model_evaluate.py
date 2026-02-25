"""
src/evaluation/model_evaluate.py
─────────────────────────────────
Downloads the trained model artifact from S3, runs a held-out evaluation,
writes metrics to S3, and exits with code 1 if quality gates are not met
(which will fail the GitHub Actions workflow and block deployment).

Usage:
    python src/evaluation/model_evaluate.py
    
Env vars expected:
    MODEL_DATA  — s3://bucket/prefix/model.tar.gz  (written by model_train.py)
    MODEL_TYPE  — nb | lstm | bert
"""

import os
import sys
import json
import logging
import pickle
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dep import (
    AWS_REGION,
    S3_BUCKET,
    PROJECT_NAME,
    MIN_ACCURACY,
    MIN_F1_FAKE,
    MODEL_TYPE,
    MAX_LEN,
)

# S3 prefixes derived from PROJECT_NAME (can be overridden by env at runtime)
S3_PROCESSED_PREFIX = os.environ.get("S3_PROCESSED_PREFIX", f"{PROJECT_NAME}/data/processed/")
S3_METRICS_PREFIX   = os.environ.get("S3_METRICS_PREFIX",   f"{PROJECT_NAME}/metrics/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def parse_s3_uri(uri: str):
    """Split  s3://bucket/key  into  (bucket, key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


def download_and_extract_model(s3, model_data_uri: str, dest: Path) -> Path:
    bucket, key = parse_s3_uri(model_data_uri)
    tarball = dest / "model.tar.gz"
    log.info("Downloading model artifact: %s", model_data_uri)
    s3.download_file(Bucket=bucket, Key=key, Filename=str(tarball))
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest)
    log.info("Extracted model to %s", dest)
    return dest


def load_test_data(s3, tmp: Path) -> pd.DataFrame:
    proc_path = tmp / "processed_dataset.parquet"
    log.info("Downloading processed dataset for evaluation…")
    s3.download_file(
        Bucket=S3_BUCKET,
        Key=f"{S3_PROCESSED_PREFIX}processed_dataset.parquet",
        Filename=str(proc_path),
    )
    df = pd.read_parquet(proc_path)
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
    return test_df.reset_index(drop=True)


def evaluate_nb(model_dir: Path, test_df: pd.DataFrame) -> dict:
    from sklearn.metrics import classification_report, roc_auc_score
    with open(model_dir / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(model_dir / "nb_model.pkl", "rb") as f:
        nb = pickle.load(f)

    X = tfidf.transform(test_df["clean_tfidf"].values)
    y = test_df["label_id"].values
    y_pred = nb.predict(X)
    y_prob = nb.predict_proba(X)[:, 1]
    report = classification_report(y, y_pred, output_dict=True)
    return {
        "model_type": "nb",
        "accuracy":       report["accuracy"],
        "auc":            float(roc_auc_score(y, y_prob)),
        "f1_fake":        report["1"]["f1-score"],
        "precision_fake": report["1"]["precision"],
        "recall_fake":    report["1"]["recall"],
    }


def evaluate_lstm(model_dir: Path, test_df: pd.DataFrame) -> dict:
    """
    Evaluate the PyTorch BiLSTM model saved by train_lstm.py.
    Loads lstm_model.pt + vocab.pkl + model_config.json.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, roc_auc_score

    # ── Reconstruct model architecture (must match train_lstm.py) ────────────
    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, lstm_units, dense_units, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm      = nn.LSTM(embed_dim, lstm_units, batch_first=True, bidirectional=True)
            self.dropout   = nn.Dropout(dropout)
            self.fc1       = nn.Linear(lstm_units * 2, dense_units)
            self.fc2       = nn.Linear(dense_units, 1)
            self.relu      = nn.ReLU()
        def forward(self, x):
            emb = self.embedding(x)
            _, (h, _) = self.lstm(emb)
            h = torch.cat([h[-2], h[-1]], dim=1)
            return self.fc2(self.dropout(self.relu(self.fc1(h)))).squeeze(1)

    # ── Load vocab and config ─────────────────────────────────────────────────
    with open(model_dir / "vocab.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open(model_dir / "model_config.json") as f:
        cfg = json.load(f)

    max_len = cfg["max_len"]

    # ── Encode and pad test texts ─────────────────────────────────────────────
    def encode_and_pad(texts):
        seqs = [[word2idx.get(w, 1) for w in str(t).split()[:max_len]] for t in texts]
        padded = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        return padded

    X = encode_and_pad(test_df["clean_lstm"].fillna("").values)
    y = test_df["label_id"].values

    # ── Load model weights ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        lstm_units=cfg["lstm_units"], dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
    ).to(device)
    state = torch.load(model_dir / "lstm_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── Run inference in batches ──────────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.tensor(X, dtype=torch.long))
    loader  = DataLoader(dataset, batch_size=64)
    probs_list = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs_list.extend(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.array(probs_list)
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y, y_pred, output_dict=True)
    return {
        "model_type": "lstm",
        "accuracy":       report["accuracy"],
        "auc":            float(roc_auc_score(y, y_prob)),
        "f1_fake":        report["1"]["f1-score"],
        "precision_fake": report["1"]["precision"],
        "recall_fake":    report["1"]["recall"],
    }


def evaluate_bert(model_dir: Path, test_df: pd.DataFrame) -> dict:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import classification_report, roc_auc_score

    class SimpleDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len):
            self.texts=texts; self.labels=labels; self.tok=tok; self.max_len=max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc=self.tok(self.texts.iloc[idx],truncation=True,padding="max_length",
                         max_length=self.max_len,return_tensors="pt")
            return {"input_ids":enc["input_ids"].squeeze(0),
                    "attention_mask":enc["attention_mask"].squeeze(0),
                    "labels":torch.tensor(self.labels.iloc[idx],dtype=torch.long)}

    tokenizer = BertTokenizer.from_pretrained(str(model_dir / "bert_tokenizer"))
    model     = BertForSequenceClassification.from_pretrained(str(model_dir / "bert_model"))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dl = DataLoader(SimpleDataset(test_df["clean_text_raw"], test_df["label_id"], tokenizer, 256), batch_size=16)
    preds=[]; probs=[]; true=[]
    with torch.no_grad():
        for b in dl:
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            p   = torch.softmax(out.logits,dim=1)[:,1].cpu().numpy()
            probs.extend(p); preds.extend((p>=0.5).astype(int))
            true.extend(b["labels"].cpu().numpy())
    report = classification_report(true, preds, output_dict=True)
    return {
        "model_type": "bert",
        "accuracy":       report["accuracy"],
        "auc":            float(roc_auc_score(true, probs)),
        "f1_fake":        report["1"]["f1-score"],
        "precision_fake": report["1"]["precision"],
        "recall_fake":    report["1"]["recall"],
    }


def main() -> None:
    model_data = os.environ.get("MODEL_DATA", "")
    model_type = os.environ.get("MODEL_TYPE", MODEL_TYPE).lower()

    if not model_data:
        log.error("MODEL_DATA env var is not set. Cannot evaluate.")
        sys.exit(1)

    s3 = get_s3_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp        = Path(tmpdir)
        model_dir  = tmp / "model"
        model_dir.mkdir()

        download_and_extract_model(s3, model_data, model_dir)
        test_df = load_test_data(s3, tmp)

        evaluators = {"nb": evaluate_nb, "lstm": evaluate_lstm, "bert": evaluate_bert}
        if model_type not in evaluators:
            log.error("Unknown MODEL_TYPE: %s", model_type)
            sys.exit(1)

        metrics = evaluators[model_type](model_dir, test_df)
        log.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))

        # ── Quality gates ──────────────────────────────────────────────────────
        passed = True
        if metrics["accuracy"] < MIN_ACCURACY:
            log.error("GATE FAILED: accuracy=%.4f < threshold=%.4f", metrics["accuracy"], MIN_ACCURACY)
            passed = False
        if metrics["f1_fake"] < MIN_F1_FAKE:
            log.error("GATE FAILED: f1_fake=%.4f < threshold=%.4f", metrics["f1_fake"], MIN_F1_FAKE)
            passed = False

        metrics["quality_gate_passed"] = passed

        # ── Upload metrics to S3 ───────────────────────────────────────────────
        metrics_path = tmp / "eval_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        s3.upload_file(
            Filename=str(metrics_path),
            Bucket=S3_BUCKET,
            Key=f"{S3_METRICS_PREFIX}eval_metrics.json",
        )
        log.info("Metrics uploaded to s3://%s/%seval_metrics.json", S3_BUCKET, S3_METRICS_PREFIX)

        # Also expose for the next GitHub Actions job
        Path("eval_metrics.env").write_text(
            f"QUALITY_GATE_PASSED={str(passed).lower()}\n"
            f"MODEL_ACCURACY={metrics['accuracy']:.4f}\n"
            f"MODEL_F1_FAKE={metrics['f1_fake']:.4f}\n"
        )

    if not passed:
        log.error("Quality gate(s) failed — aborting pipeline.")
        sys.exit(1)

    log.info("All quality gates passed. Proceeding to deployment.")


if __name__ == "__main__":
    main()
    