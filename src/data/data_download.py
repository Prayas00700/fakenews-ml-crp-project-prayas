"""
src/data/data_download.py
─────────────────────────
Downloads the two Fake News CSV datasets from Kaggle and uploads them to S3 under:

    s3://<bucket>/<project>/data/raw/

Datasets:
  1. vimlendusharma/fake-or-real-news  →  fake_or_real_news.csv
  2. algord/fakenewsnet                →  FakeNewsNet.csv

Pattern mirrors the heart-disease pipeline:
  - Accepts --bucket and --project CLI args (passed by GitHub Actions)
  - Downloads CSVs from Kaggle using the Kaggle API
  - Creates S3 folder structure if it doesn't exist
  - Uploads each CSV with a timestamp suffix so versions are preserved
  - Writes a metadata JSON alongside each upload
  - Writes s3_paths.env for downstream workflow jobs

Prerequisites:
  - KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set
    (passed by GitHub Actions via repository secrets)
  - AWS credentials configured (OIDC role assumption via GitHub Actions)

Usage (local):
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    python src/data/data_download.py \
        --bucket my-bucket \
        --project fake-news

GitHub Actions passes --bucket and --project from ${{ env.S3_BUCKET }}
and ${{ env.PROJECT_NAME }} secrets/variables.
"""

import os
import sys
import json
import logging
import argparse
import zipfile
import shutil
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
# Kaggle dataset configuration
# ──────────────────────────────────────────────────────────────────────────────

KAGGLE_DATASETS = {
    DATA1_FILENAME: {
        "slug":     "vimlendusharma/fake-or-real-news",
        "zip_name": "fake-or-real-news.zip",      # zip produced by `kaggle datasets download`
        "csv_name": "fake_or_real_news.csv",       # CSV inside the zip
    },
    DATA2_FILENAME: {
        "slug":     "algord/fake-news",
        "zip_name": "FakeNewsNet.zip",
        "csv_name": "FakeNewsNet.csv",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Kaggle helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_kaggle_credentials() -> None:
    """Ensure KAGGLE_USERNAME and KAGGLE_KEY are available."""
    missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(v)]
    if missing:
        logger.error(
            "Missing Kaggle credentials: %s\n"
            "Set them as GitHub Actions secrets and expose them as env vars.",
            ", ".join(missing),
        )
        sys.exit(1)
    logger.info("Kaggle credentials found (KAGGLE_USERNAME=%s)", os.environ["KAGGLE_USERNAME"])


def download_kaggle_dataset(slug: str, dest_dir: Path) -> None:
    """
    Download a Kaggle dataset using the kaggle CLI.
    Equivalent to:  kaggle datasets download -d <slug> -p <dest_dir> --unzip
    """
    import subprocess
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", slug,
        "-p", str(dest_dir),
        "--unzip",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("kaggle CLI failed:\n%s", result.stderr)
        sys.exit(1)
    logger.info("Kaggle download output:\n%s", result.stdout.strip())


def fetch_datasets(download_dir: Path) -> dict[str, Path]:
    """
    Download both Kaggle datasets and return a mapping of
    {filename_key: local_csv_path}.
    """
    local_paths = {}
    for filename, cfg in KAGGLE_DATASETS.items():
        dataset_dir = download_dir / filename.replace(".csv", "")
        download_kaggle_dataset(cfg["slug"], dataset_dir)

        csv_path = dataset_dir / cfg["csv_name"]
        if not csv_path.exists():
            # Sometimes Kaggle extracts into a nested folder — search recursively
            matches = list(dataset_dir.rglob(cfg["csv_name"]))
            if not matches:
                logger.error(
                    "Expected CSV '%s' not found after download in %s",
                    cfg["csv_name"], dataset_dir,
                )
                sys.exit(1)
            csv_path = matches[0]
            logger.info("Found CSV at: %s", csv_path)

        local_paths[filename] = csv_path
        logger.info("Dataset ready: %s  →  %s", filename, csv_path)

    return local_paths


# ──────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_s3_client() -> boto3.client:
    """Return a boto3 S3 client using the environment-provided credentials."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        # Credentials injected by GitHub Actions via aws-actions/configure-aws-credentials
    )


def setup_s3_bucket(s3: boto3.client, bucket: str, region: str) -> None:
    """Create S3 bucket with versioning if it doesn't exist, and scaffold folder structure."""
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
    s3_filename: str,
    timestamp: str,
) -> str:
    """
    Upload a single CSV file to s3://<bucket>/<project>/data/raw/<stem>_<timestamp>.csv
    Returns the full S3 URI.
    """
    stem    = Path(s3_filename).stem
    ext     = Path(s3_filename).suffix
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
    parser = argparse.ArgumentParser(description="Download Fake News datasets from Kaggle and upload to S3")
    parser.add_argument("--bucket",       required=True,           help="S3 bucket name")
    parser.add_argument("--project",      default="fake-news",     help="Project name (S3 prefix)")
    parser.add_argument("--download-dir", default="/tmp/kaggle_data", help="Temp directory for Kaggle downloads")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FAKE NEWS — KAGGLE DOWNLOAD + S3 UPLOAD")
    logger.info("=" * 70)
    logger.info("  Bucket      : %s", args.bucket)
    logger.info("  Project     : %s", args.project)
    logger.info("  Download dir: %s", args.download_dir)
    logger.info("=" * 70)

    # ── 0. Validate Kaggle credentials ────────────────────────────────────────
    validate_kaggle_credentials()

    # ── 1. Download datasets from Kaggle ──────────────────────────────────────
    download_dir = Path(args.download_dir)
    local_paths  = fetch_datasets(download_dir)

    s3        = get_s3_client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 2. Setup S3 bucket & folder structure ─────────────────────────────────
    setup_s3_bucket(s3, args.bucket, AWS_REGION)
    create_folder_structure(s3, args.bucket, args.project)

    # ── 3. Upload each CSV dataset ────────────────────────────────────────────
    datasets = [DATA1_FILENAME, DATA2_FILENAME]
    uploaded_uris = {}

    for filename in datasets:
        local_path = local_paths[filename]
        s3_uri = upload_csv_to_s3(
            s3, local_path, args.bucket, args.project, filename, timestamp
        )
        uploaded_uris[filename] = s3_uri

    # ── 4. Upload metadata JSON ───────────────────────────────────────────────
    metadata = {
        "timestamp": timestamp,
        "project":   args.project,
        "bucket":    args.bucket,
        "source":    "kaggle",
        "datasets":  {
            DATA1_FILENAME: {
                "kaggle_slug": KAGGLE_DATASETS[DATA1_FILENAME]["slug"],
                "s3_uri":      uploaded_uris[DATA1_FILENAME],
                "rows":        None,
                "columns":     None,
            },
            DATA2_FILENAME: {
                "kaggle_slug": KAGGLE_DATASETS[DATA2_FILENAME]["slug"],
                "s3_uri":      uploaded_uris[DATA2_FILENAME],
                "rows":        None,
                "columns":     None,
            },
        },
    }

    for filename in datasets:
        local_path = local_paths[filename]
        try:
            full_df = pd.read_csv(local_path, engine="python", on_bad_lines="skip")
            metadata["datasets"][filename]["rows"]    = len(full_df)
            metadata["datasets"][filename]["columns"] = len(full_df.columns)
            logger.info("  %s: %d rows, %d columns", filename, len(full_df), len(full_df.columns))
        except Exception as exc:
            logger.warning("Could not read row count for %s: %s", filename, exc)

    upload_metadata(s3, args.bucket, args.project, timestamp, metadata)

    # ── 5. Write s3_paths.env for GitHub Actions downstream jobs ──────────────
    env_path = Path("s3_paths.env")
    with env_path.open("w") as fh:
        fh.write(f"S3_BUCKET={args.bucket}\n")
        fh.write(f"PROJECT_NAME={args.project}\n")
        fh.write(f"UPLOAD_TIMESTAMP={timestamp}\n")
        for filename, uri in uploaded_uris.items():
            env_key = filename.replace(".", "_").replace("-", "_").upper()
            fh.write(f"S3_{env_key}={uri}\n")
    logger.info("Wrote S3 paths to %s", env_path)

    # ── 6. Cleanup temp download dir ─────────────────────────────────────────
    shutil.rmtree(download_dir, ignore_errors=True)
    logger.info("Cleaned up temp download directory: %s", download_dir)

    logger.info("=" * 70)
    logger.info("SUCCESS — All datasets downloaded from Kaggle and uploaded to S3")
    logger.info("  Raw data at: s3://%s/%s/data/raw/", args.bucket, args.project)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
