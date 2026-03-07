"""
data_download.py
Downloads Kaggle datasets and uploads them to S3.

Usage:
    python src/data/data_download.py \
        --bucket <s3-bucket> \
        --project <project-name> \
        --download-dir /tmp/kaggle_data
"""

import argparse
import os
import boto3
import zipfile
from pathlib import Path
import kaggle  # requires kaggle package + KAGGLE_USERNAME/KAGGLE_KEY env vars


DATASETS = [
    {
        "slug": "vimlendusharma/fake-or-real-news",
        "files": None,  # download all files
    },
    {
        "slug": "clmentbisaillon/fake-and-real-news-dataset",
        "files": ["Fake.csv", "True.csv"],
    },
]


def download_datasets(download_dir: Path):
    """Download all configured Kaggle datasets into download_dir."""
    download_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        slug = dataset["slug"]
        dest = download_dir / slug.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {slug} → {dest}")
        kaggle.api.dataset_download_files(slug, path=str(dest), unzip=True)
        print(f"  ✓ Done: {slug}")

    return download_dir


def merge_fake_real(download_dir: Path) -> Path:
    """
    Merge Fake.csv and True.csv from the clmentbisaillon dataset,
    adding a 'label' column (0 = fake, 1 = real), and save as merged_news.csv.
    """
    import pandas as pd

    base = download_dir / "clmentbisaillon_fake-and-real-news-dataset"
    fake_path = base / "Fake.csv"
    true_path = base / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        print("  ⚠ Fake.csv or True.csv not found, skipping merge.")
        return None

    print("  Merging Fake.csv + True.csv with labels...")
    fake_df = __import__("pandas").read_csv(fake_path)
    true_df = __import__("pandas").read_csv(true_path)

    fake_df["label"] = 0  # fake
    true_df["label"] = 1  # real

    merged = __import__("pandas").concat([fake_df, true_df], ignore_index=True)
    merged_path = base / "merged_news.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  ✓ Merged dataset saved to {merged_path} ({len(merged)} rows)")
    return merged_path


def upload_to_s3(download_dir: Path, bucket: str, project: str):
    """Recursively upload all files from download_dir to S3."""
    s3 = boto3.client("s3")
    prefix = f"{project}/data/raw"

    files = list(download_dir.rglob("*"))
    data_files = [f for f in files if f.is_file()]

    print(f"  Uploading {len(data_files)} file(s) to s3://{bucket}/{prefix}/")

    for local_path in data_files:
        relative = local_path.relative_to(download_dir)
        s3_key = f"{prefix}/{relative}"
        print(f"    → s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_path), bucket, s3_key)

    print(f"  ✓ Upload complete.")


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle data and upload to S3.")
    parser.add_argument("--bucket",       required=True, help="S3 bucket name")
    parser.add_argument("--project",      required=True, help="Project name (used as S3 prefix)")
    parser.add_argument("--download-dir", required=True, help="Local directory to store downloads")
    parser.add_argument("--skip-upload",  action="store_true", help="Skip S3 upload (for local testing)")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)

    # 1. Download datasets
    print("\n[1/3] Downloading Kaggle datasets...")
    download_datasets(download_dir)

    # 2. Merge fake/real news dataset
    print("\n[2/3] Merging clmentbisaillon datasets...")
    merge_fake_real(download_dir)

    # 3. Upload to S3
    if not args.skip_upload:
        print(f"\n[3/3] Uploading to s3://{args.bucket}/{args.project}/data/raw/")
        upload_to_s3(download_dir, args.bucket, args.project)
    else:
        print("\n[3/3] Skipping S3 upload (--skip-upload flag set).")

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
