"""
src/data/data_download.py
─────────────────────────
Uploads the two local Fake News CSV datasets to S3 under:

    s3://<bucket>/<project>/data/raw/

Pattern mirrors the heart-disease pipeline:
  - Accepts --bucket and --project CLI args (passed by GitHub Actions)
  - Creates S3 folder structure if it doesn't exist
  - Uploads each CSV with a timestamp suffix so versions are preserved
  - Writes a metadata JSON alongside each upload
  - Writes s3_paths.env for downstream workflow jobs

Usage (local):
    python src/data/data_download.py \
        --bucket my-bucket \
        --project fake-news \
        --data-dir data/local

GitHub Actions passes --bucket and --project from ${{ env.S3_BUCKET }}
and ${{ env.PROJECT_NAME }} secrets/variables.
"""

import os
import sys
import json
import logging
import argparse
from io import StringIO
from pathlib import Path
from datetime import datetime

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

# Allow running from repo root
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
# S3 helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_s3_client() -> boto3.client:
    """Return a boto3 S3 client using the environment-provided credentials."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        # Credentials are injected by GitHub Actions via aws-actions/configure-aws-credentials
        # (OIDC role assumption) — boto3 picks them up from the environment automatically.
    )


def setup_s3_bucket(s3: boto3.client, bucket: str, region: str) -> None:
    """Create S3 bucket with versioning if it doesn't exist, and scaffold folder structure."""
    # Check / create bucket
    try:
        s3.head_bucket(Bucket=bucket)
        logger.info("Bucket already exists: s3://%s", bucket)
    except ClientError as exc:
        error_code = int(exc.response["Error"]["Code"])
        if error_code == 404:
            logger.info("Creating S3 bucket: s3://%s  (region=%s)", bucket, region)
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            # Enable versioning
            s3.put_bucket_versioning(
                Bucket=bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )
            logger.info("Versioning enabled on bucket s3://%s", bucket)
        else:
            raise


def create_folder_structure(s3: boto3.client, bucket: str, project: str) -> None:
    """Scaffold the S3 folder hierarchy for the project."""
    folders = [
        f"{project}/data/raw/",
        f"{project}/data/processed/",
        f"{project}/models/output/",
        f"{project}/metrics/",
        f"{project}/code/",
    ]
    for folder in folders:
        s3.put_object(Bucket=bucket, Key=folder, Body=b"")
    logger.info("S3 folder structure created under s3://%s/%s/", bucket, project)


def upload_csv_to_s3(
    s3: boto3.client,
    local_path: Path,
    bucket: str,
    project: str,
    timestamp: str,
) -> str:
    """
    Upload a single CSV file to s3://<bucket>/<project>/data/raw/<stem>_<timestamp>.csv
    Returns the full S3 URI.
    """
    if not local_path.exists():
        logger.error("Local file not found: %s", local_path)
        sys.exit(1)

    stem    = local_path.stem          # e.g.  fake_or_real_news
    ext     = local_path.suffix        # e.g.  .csv
    s3_key  = f"{project}/data/raw/{stem}_{timestamp}{ext}"
    s3_uri  = f"s3://{bucket}/{s3_key}"

    file_size_mb = local_path.stat().st_size / 1_048_576
    logger.info(
        "Uploading  %s  (%.2f MB)  →  %s",
        local_path, file_size_mb, s3_uri,
    )

    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        logger.info("Upload complete: %s", s3_uri)
    except (BotoCoreError, ClientError) as exc:
        logger.error("Upload failed: %s", exc)
        sys.exit(1)

    return s3_uri


def upload_metadata(
    s3: boto3.client,
    bucket: str,
    project: str,
    timestamp: str,
    metadata: dict,
) -> None:
    """Upload a metadata JSON alongside the raw data files."""
    metadata_key = f"{project}/data/raw/metadata_{timestamp}.json"
    s3.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Metadata uploaded to s3://%s/%s", bucket, metadata_key)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Fake News CSV datasets to S3")
    parser.add_argument("--bucket",   required=True,          help="S3 bucket name")
    parser.add_argument("--project",  default="fake-news",    help="Project name (S3 prefix)")
    parser.add_argument("--data-dir", default="data/local",   help="Local directory containing CSVs")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FAKE NEWS — DATA UPLOAD TO S3")
    logger.info("=" * 70)
    logger.info("  Bucket  : %s", args.bucket)
    logger.info("  Project : %s", args.project)
    logger.info("  Data dir: %s", args.data_dir)
    logger.info("=" * 70)

    s3        = get_s3_client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir  = Path(args.data_dir)

    # ── 1. Setup S3 bucket & folder structure ─────────────────────────────────
    setup_s3_bucket(s3, args.bucket, AWS_REGION)
    create_folder_structure(s3, args.bucket, args.project)

    # ── 2. Upload each CSV dataset ────────────────────────────────────────────
    datasets = [DATA1_FILENAME, DATA2_FILENAME]
    uploaded_uris = {}

    for filename in datasets:
        local_path = data_dir / filename
        s3_uri = upload_csv_to_s3(s3, local_path, args.bucket, args.project, timestamp)
        uploaded_uris[filename] = s3_uri

    # ── 3. Upload metadata JSON ───────────────────────────────────────────────
    metadata = {
        "timestamp":      timestamp,
        "project":        args.project,
        "bucket":         args.bucket,
        "datasets":       {
            DATA1_FILENAME: {
                "s3_uri":    uploaded_uris[DATA1_FILENAME],
                "rows":      None,   # populated below
                "columns":   None,
            },
            DATA2_FILENAME: {
                "s3_uri":    uploaded_uris[DATA2_FILENAME],
                "rows":      None,
                "columns":   None,
            },
        },
    }

    # Optionally read row counts for the metadata
    for filename in datasets:
        local_path = data_dir / filename
        try:
            df = pd.read_csv(local_path, engine="python", on_bad_lines="skip", nrows=0)
            full_df = pd.read_csv(local_path, engine="python", on_bad_lines="skip")
            metadata["datasets"][filename]["rows"]    = len(full_df)
            metadata["datasets"][filename]["columns"] = len(full_df.columns)
            logger.info("  %s: %d rows, %d columns", filename, len(full_df), len(full_df.columns))
        except Exception as exc:
            logger.warning("Could not read row count for %s: %s", filename, exc)

    upload_metadata(s3, args.bucket, args.project, timestamp, metadata)

    # ── 4. Write s3_paths.env for GitHub Actions downstream jobs ─────────────
    env_path = Path("s3_paths.env")
    with env_path.open("w") as fh:
        fh.write(f"S3_BUCKET={args.bucket}\n")
        fh.write(f"PROJECT_NAME={args.project}\n")
        fh.write(f"UPLOAD_TIMESTAMP={timestamp}\n")
        for filename, uri in uploaded_uris.items():
            env_key = filename.replace(".", "_").replace("-", "_").upper()
            fh.write(f"S3_{env_key}={uri}\n")
    logger.info("Wrote S3 paths to %s", env_path)

    logger.info("=" * 70)
    logger.info("SUCCESS — All datasets uploaded")
    logger.info("  Raw data at: s3://%s/%s/data/raw/", args.bucket, args.project)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
