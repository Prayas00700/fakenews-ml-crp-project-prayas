"""
src/data/data_preprocess.py
───────────────────────────
Fake News Detection — Data Preprocessing Script
Designed to run as a SageMaker Processing Job (mirrors heart-disease pipeline).

When running on SageMaker:
  - Raw CSVs are downloaded from S3 input channel  →  /opt/ml/processing/input/data/
  - Output is written to /opt/ml/processing/output/ which SageMaker auto-uploads to S3
  - S3 output lands at:  s3://<bucket>/<project>/data/processed/

When running locally / on the GitHub Actions runner directly:
  - Downloads raw CSVs from S3 to a temp directory
  - Writes processed files to /tmp and uploads to S3 manually

Pipeline steps:
  1. Load raw CSVs (fake_or_real_news.csv + FakeNewsNet.csv)
  2. Normalise labels  →  fake=1 / real=0
  3. Merge both datasets
  4. Clean text for TF-IDF (heavy: lowercase, remove stopwords, lemmatise)
  5. Clean text for LSTM/BERT (light: basic clean + lowercase)
  6. Save processed_dataset.parquet + processed_dataset.csv + preprocessing_stats.json
  7. Write SUCCESS marker

Usage (SageMaker container entry-point):
    python data_preprocess.py \
        --bucket   my-bucket \
        --prefix   fake-news \
        --output-path /opt/ml/processing/output

Usage (standalone / GitHub Actions runner):
    python src/data/data_preprocess.py \
        --bucket   my-bucket \
        --prefix   fake-news
"""

import os
import re
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from io import StringIO

import boto3
import pandas as pd
import numpy as np
import nltk
from botocore.exceptions import BotoCoreError, ClientError
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dep import (
    AWS_REGION,
    DATA1_FILENAME,
    DATA2_FILENAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# NLTK setup
# ──────────────────────────────────────────────────────────────────────────────

def download_nltk_resources() -> None:
    """Download all required NLTK corpora (quiet, no-op if already present)."""
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            logger.warning("Could not download NLTK resource %s: %s", resource, exc)


# ──────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_s3_client() -> boto3.client:
    """Return a boto3 S3 client.  Credentials injected by GitHub Actions OIDC."""
    return boto3.client("s3", region_name=AWS_REGION)


def load_latest_csv_from_s3(s3: boto3.client, bucket: str, prefix: str, filename_pattern: str) -> pd.DataFrame:
    """
    Find the most recently uploaded CSV whose key contains filename_pattern
    under  s3://<bucket>/<prefix>/data/raw/  and return it as a DataFrame.
    """
    raw_prefix = f"{prefix}/data/raw/"
    logger.info("Searching for '%s' in s3://%s/%s", filename_pattern, bucket, raw_prefix)

    response = s3.list_objects_v2(Bucket=bucket, Prefix=raw_prefix)
    if "Contents" not in response:
        raise FileNotFoundError(f"No files found in s3://{bucket}/{raw_prefix}")

    # Filter to files that match the pattern
    matches = [
        obj for obj in response["Contents"]
        if filename_pattern.replace(".csv", "").lower() in obj["Key"].lower()
        and obj["Key"].endswith(".csv")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No CSV matching '{filename_pattern}' found in s3://{bucket}/{raw_prefix}"
        )

    # Pick the most recently modified
    latest = max(matches, key=lambda x: x["LastModified"])
    s3_key = latest["Key"]
    logger.info("  Loading: s3://%s/%s", bucket, s3_key)

    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    df  = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")), engine="python", on_bad_lines="skip")
    logger.info("  Loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def upload_file_to_s3(s3: boto3.client, local_path: Path, bucket: str, s3_key: str) -> None:
    """Upload a single file to S3 with server-side encryption."""
    logger.info("Uploading %s  →  s3://%s/%s", local_path, bucket, s3_key)
    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("Upload failed: %s", exc)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Label normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_label(v) -> str | None:
    """
    Map any variant of fake/real labels to canonical 'fake' or 'real'.
    Returns None for unrecognised values so those rows can be dropped.
    """
    if pd.isna(v):
        return None
    # Handle numeric 0 / 1
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            return "real" if int(v) == 1 else "fake"
        except Exception:
            pass
    s = str(v).strip().lower()
    if s in ("fake", "f", "0", "false", "hoax", "fake news"):
        return "fake"
    if s in ("real", "r", "1", "true", "reliable"):
        return "real"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────────────

def basic_clean(text: str) -> str:
    """Light cleaning applied to ALL text variants."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                 # remove emails
    text = re.sub(r"<.*?>", " ", text)                   # remove HTML tags
    text = re.sub(r"[\r\n]+", " ", text)                 # remove newlines
    text = re.sub(r"[_\[\]\(\)\{\};:<>@#\$%\^&\+=~`|\\\/]", " ", text)  # symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_tfidf(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Heavy cleaning for TF-IDF / Naive Bayes:
    lowercase → remove non-alpha → remove stopwords → lemmatise
    """
    text   = basic_clean(text).lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_for_lstm(text: str) -> str:
    """
    Light cleaning for LSTM / BERT:
    Keep stopwords and sentence structure — the model learns those patterns.
    """
    return basic_clean(text).lower()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading and normalisation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_df1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare fake_or_real_news.csv
    Expected columns: id, title, text, label
    """
    logger.info("Preparing df1 (fake_or_real_news)...")

    # Drop unnamed index columns
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        logger.info("  Dropping junk columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    df["title"] = df.get("title", pd.Series(dtype=str)).fillna("")
    df["text"]  = df.get("text",  pd.Series(dtype=str)).fillna("")

    df["clean_text_raw"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    df["label_norm"]     = df["label"].apply(normalize_label)

    before = len(df)
    df = df.dropna(subset=["label_norm"]).reset_index(drop=True)
    logger.info("  Dropped %d rows with unrecognised labels", before - len(df))
    logger.info("  Shape: %s  |  Labels: %s", df.shape, df["label_norm"].value_counts().to_dict())

    return df[["clean_text_raw", "label_norm"]].rename(columns={"label_norm": "label"})


def prepare_df2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare FakeNewsNet.csv
    Expected columns: title, news_url, source_domain, tweet_num, label (or 'real')
    """
    logger.info("Preparing df2 (FakeNewsNet)...")

    # --- FIX 1: Robust Column Mapping ---
    # We check for 'real' or other variants and map them to 'label' immediately
    if "real" in df.columns:
        df = df.rename(columns={"real": "label"})
        logger.info("  Renamed column 'real' -> 'label'")
    
    # Ensure a 'label' column exists now, otherwise normalize_label will fail
    if "label" not in df.columns:
        # Fallback for datasets that use 'class' or similar, or error out clearly
        for col in ["class", "target"]:
            if col in df.columns:
                df = df.rename(columns={col: "label"})
                break

    df["title"]         = df.get("title",         pd.Series(dtype=str)).fillna("")
    df["source_domain"] = df.get("source_domain",  pd.Series(dtype=str)).fillna("")
    
    # --- FIX 2: Text Column Safety ---
    # FakeNewsNet often has a 'title' but sometimes missing 'text' body
    df["text"]          = df.get("text",           pd.Series(dtype=str)).fillna("")

    df["clean_text_raw"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    
    # This now safely finds 'label' because of the rename above
    df["label_norm"]     = df["label"].apply(normalize_label)

    before = len(df)
    df = df.dropna(subset=["label_norm"]).reset_index(drop=True)
    logger.info("  Dropped %d rows with unrecognised labels", before - len(df))
    logger.info("  Shape: %s  |  Labels: %s", df.shape, df["label_norm"].value_counts().to_dict())

    return df[["clean_text_raw", "label_norm"]].rename(columns={"label_norm": "label"})

# ──────────────────────────────────────────────────────────────────────────────
# Save processed outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, stats: dict, bucket: str, prefix: str, output_path: str | None) -> None:
    """
    Save processed_dataset.parquet, processed_dataset.csv, and
    preprocessing_stats.json.

    If output_path is given (SageMaker Processing Job local output mount):
      → files are written there and SageMaker auto-uploads to S3.

    If output_path is None (standalone / GH Actions runner):
      → files are written to /tmp and uploaded to S3 manually.
    """
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path   = Path(output_path) if output_path else Path("/tmp")
    base_path.mkdir(parents=True, exist_ok=True)

    # ── Write files ────────────────────────────────────────────────────────────
    parquet_path = base_path / "processed_dataset.parquet"
    csv_path     = base_path / "processed_dataset.csv"
    stats_path   = base_path / "preprocessing_stats.json"

    df.to_parquet(parquet_path, index=False)
    logger.info("  Saved parquet: %s", parquet_path)

    df.to_csv(csv_path, index=False)
    logger.info("  Saved CSV    : %s", csv_path)

    stats["timestamp"] = timestamp
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("  Saved stats  : %s", stats_path)

    # ── Write SUCCESS marker ──────────────────────────────────────────────────
    (base_path / "SUCCESS").write_text("Preprocessing completed successfully\n")

    # ── If standalone mode, upload to S3 manually ─────────────────────────────
    if output_path is None:
        s3 = get_s3_client()
        processed_prefix = f"{prefix}/data/processed/"
        for local_file, s3_filename in [
            (parquet_path, f"processed_dataset_{timestamp}.parquet"),
            (csv_path,     f"processed_dataset_{timestamp}.csv"),
            (stats_path,   f"preprocessing_stats_{timestamp}.json"),
        ]:
            s3_key = f"{processed_prefix}{s3_filename}"
            upload_file_to_s3(s3, local_file, bucket, s3_key)

        logger.info("All processed files uploaded to s3://%s/%s", bucket, processed_prefix)

    return timestamp, str(parquet_path), str(csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Fake News datasets")
    parser.add_argument("--bucket",      required=True,                  help="S3 bucket name")
    parser.add_argument("--prefix",      default="fake-news",            help="Project prefix in S3")
    parser.add_argument("--output-path", default=None,                   help="Local output path (for SageMaker Processing)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FAKE NEWS — DATA PREPROCESSING")
    logger.info("=" * 70)
    logger.info("  Bucket      : %s", args.bucket)
    logger.info("  Prefix      : %s", args.prefix)
    logger.info("  Output path : %s", args.output_path or "Direct upload to S3")
    logger.info("=" * 70)

    try:
        # ── Step 1: NLTK resources ─────────────────────────────────────────────
        logger.info("Downloading NLTK resources...")
        download_nltk_resources()
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        logger.info("  stop_words loaded (%d words)", len(stop_words))

        # ── Step 2: Load raw CSVs from S3 ─────────────────────────────────────
        s3 = get_s3_client()
        raw_df1 = load_latest_csv_from_s3(s3, args.bucket, args.prefix, DATA1_FILENAME)
        raw_df2 = load_latest_csv_from_s3(s3, args.bucket, args.prefix, DATA2_FILENAME)

        # ── Step 3: Prepare each dataset ──────────────────────────────────────
        df1_proc = prepare_df1(raw_df1)
        df2_proc = prepare_df2(raw_df2)
        logger.info("df1 shape: %s  |  df2 shape: %s", df1_proc.shape, df2_proc.shape)

        # ── Step 4: Merge ──────────────────────────────────────────────────────
        df = pd.concat([df1_proc, df2_proc], ignore_index=True)
        logger.info("Merged shape: %s", df.shape)
        logger.info("Merged label distribution:\n%s", df["label"].value_counts())

        # ── Step 5: Clean text for TF-IDF and LSTM ────────────────────────────
        logger.info("Cleaning text for TF-IDF... (this may take 20–60 seconds)")
        df["clean_tfidf"] = df["clean_text_raw"].apply(
            lambda t: clean_for_tfidf(t, stop_words, lemmatizer)
        )
        logger.info("Cleaning text for LSTM...")
        df["clean_lstm"] = df["clean_text_raw"].apply(clean_for_lstm)
        df["label_id"]   = df["label"].map({"fake": 1, "real": 0})
        logger.info("Text cleaning complete")

        # ── Step 6: Build stats ────────────────────────────────────────────────
        stats = {
            "total_rows":    int(len(df)),
            "fake_count":    int((df["label"] == "fake").sum()),
            "real_count":    int((df["label"] == "real").sum()),
            "df1_rows":      int(len(df1_proc)),
            "df2_rows":      int(len(df2_proc)),
            "num_features":  int(len(df.columns)),
            "columns":       list(df.columns),
            "bucket":        args.bucket,
            "prefix":        args.prefix,
        }
        logger.info("Preprocessing stats:")
        logger.info("  Total rows : %d", stats["total_rows"])
        logger.info("  Fake       : %d", stats["fake_count"])
        logger.info("  Real       : %d", stats["real_count"])
        logger.info("  Features   : %d", stats["num_features"])

        # ── Step 7: Save outputs ───────────────────────────────────────────────
        timestamp, parquet_path, csv_path = save_outputs(
            df, stats, args.bucket, args.prefix, args.output_path
        )

        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info("  Timestamp  : %s", timestamp)
        logger.info("  Rows       : %d", stats["total_rows"])
        logger.info("  Fake       : %d", stats["fake_count"])
        logger.info("  Real       : %d", stats["real_count"])
        if args.output_path:
            logger.info("  Output dir : %s (SageMaker will upload to S3)", args.output_path)
        else:
            logger.info("  Uploaded to: s3://%s/%s/data/processed/", args.bucket, args.prefix)
        logger.info("=" * 70)

        return 0

    except Exception as exc:
        logger.error("PREPROCESSING FAILED: %s", exc)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
