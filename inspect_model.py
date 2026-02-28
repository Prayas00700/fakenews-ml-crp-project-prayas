"""
inspect_model.py — Shows exactly what's inside your model.tar.gz on S3.

Usage:
    python inspect_model.py \
        --bucket your-bucket-name \
        --project fake-news \
        --region ap-south-1

This tells us the exact directory structure so we can fix the packaging.
"""

import argparse
import boto3
import tarfile
import os
import shutil

def main(args):
    s3 = boto3.client("s3", region_name=args.region)

    # ── Find the latest model.tar.gz ─────────────────────────────────────────
    print(f"\n🔍 Searching s3://{args.bucket}/{args.project}/models/ ...\n")
    response = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"{args.project}/models/")

    model_files = [
        obj for obj in response.get("Contents", [])
        if obj["Key"].endswith("model.tar.gz")
    ]

    if not model_files:
        print("❌ No model.tar.gz found in S3!")
        return

    # Show all found
    print("Found model artifacts:")
    for f in model_files:
        print(f"  s3://{args.bucket}/{f['Key']}  ({f['Size'] / 1e6:.1f} MB)")

    # Pick latest
    latest = max(model_files, key=lambda x: x["LastModified"])
    print(f"\n📦 Inspecting latest: s3://{args.bucket}/{latest['Key']}\n")

    # ── Download ──────────────────────────────────────────────────────────────
    s3.download_file(args.bucket, latest["Key"], "/tmp/inspect_model.tar.gz")

    # ── List contents ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("TAR CONTENTS (all files):")
    print("=" * 60)
    with tarfile.open("/tmp/inspect_model.tar.gz", "r:gz") as tar:
        members = tar.getnames()
        for m in sorted(members):
            print(f"  {m}")

    # ── Extract and show directory tree ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIRECTORY TREE after extraction:")
    print("=" * 60)
    extract_dir = "/tmp/model_inspect"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    with tarfile.open("/tmp/inspect_model.tar.gz", "r:gz") as tar:
        tar.extractall(extract_dir)

    for root, dirs, files in os.walk(extract_dir):
        level = root.replace(extract_dir, "").count(os.sep)
        indent = "  " * level
        rel_root = root.replace(extract_dir, "") or "/ (ROOT)"
        print(f"{indent}{rel_root}/")
        for f in sorted(files):
            print(f"{indent}  📄 {f}")

    # ── Check for tokenizer files specifically ────────────────────────────────
    print("\n" + "=" * 60)
    print("TOKENIZER FILE LOCATIONS:")
    print("=" * 60)
    tokenizer_files = [
        "tokenizer_config.json", "vocab.txt", "config.json",
        "special_tokens_map.json", "pytorch_model.bin", "model.safetensors",
    ]
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f in tokenizer_files:
                rel_path = os.path.join(root, f).replace(extract_dir, "")
                print(f"  ✅ {rel_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    shutil.rmtree(extract_dir)
    os.remove("/tmp/inspect_model.tar.gz")
    print("\n✅ Inspection complete. Share the output above to diagnose the issue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",  required=True)
    parser.add_argument("--project", default="fake-news")
    parser.add_argument("--region",  default="ap-south-1")
    args = parser.parse_args()
    main(args)
