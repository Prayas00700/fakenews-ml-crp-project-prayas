"""
Data Download/Upload Script for Fake News Project
Reads local datasets, performs initial cleanup, and uploads to S3.
"""

import pandas as pd
import boto3
import os
import argparse
import logging
import json
from io import StringIO
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_clean_data():
    """
    Loads the Fake News datasets, drops 'Unnamed' columns, 
    and returns the cleaned dataframes.
    """
    # Define paths (Assuming files are in the 'data/' folder relative to root)
    # You must ensure these files exist in your repo or are downloaded first
    DATA1_PATH = "data/fake_or_real_news.csv" 
    DATA2_PATH = "data/FakeNewsNet.csv"
    
    logger.info(f"Looking for datasets at: {DATA1_PATH} and {DATA2_PATH}")

    if not os.path.exists(DATA1_PATH) or not os.path.exists(DATA2_PATH):
        logger.error("❌ Dataset files not found! Please ensure CSVs are in 'data/' folder.")
        raise FileNotFoundError("One or more dataset files are missing.")

    # --- Load Data 1 ---
    logger.info("Loading Data 1...")
    df1 = pd.read_csv(DATA1_PATH, engine='python', on_bad_lines='skip')
    
    # Drop junk columns (Your Logic)
    for col in df1.columns:
        if col.lower().startswith('unnamed'):
            logger.info(f"Dropping junk column in df1: {col}")
            df1 = df1.drop(columns=[col])

    # --- Load Data 2 ---
    logger.info("Loading Data 2...")
    df2 = pd.read_csv(DATA2_PATH)
    
    # Drop junk columns (Your Logic)
    for col in df2.columns:
        if col.lower().startswith('unnamed'):
            logger.info(f"Dropping junk column in df2: {col}")
            df2 = df2.drop(columns=[col])

    logger.info(f"Data Loaded: df1 shape={df1.shape}, df2 shape={df2.shape}")
    return df1, df2

def upload_to_s3(df, filename, bucket_name, project_name):
    """
    Uploads a dataframe to S3 as a CSV
    """
    logger.info(f"Uploading {filename} to S3 bucket: {bucket_name}")

    s3_client = boto3.client('s3')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define S3 Key (Path)
    s3_key = f"{project_name}/data/raw/{filename}_{timestamp}.csv"
    
    # Create Buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=csv_buffer.getvalue().encode('utf-8'),
        ContentType='text/csv'
    )

    s3_uri = f"s3://{bucket_name}/{s3_key}"
    logger.info(f"✅ Data uploaded to {s3_uri}")

    # Upload metadata
    metadata = {
        'filename': filename,
        'timestamp': timestamp,
        'rows': len(df),
        'columns': len(df.columns),
        's3_uri': s3_uri
    }

    metadata_key = f"{project_name}/data/raw/metadata_{filename}_{timestamp}.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8')
    )

    return s3_uri

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', default='fakenews-ml-crp-project', help='Project name')
    args = parser.parse_args()

    # 1. Load and Clean (Your existing logic)
    try:
        df1, df2 = load_and_clean_data()
        
        # 2. Upload to S3 (New S3 logic)
        uri1 = upload_to_s3(df1, "fake_or_real_news", args.bucket, args.project)
        uri2 = upload_to_s3(df2, "FakeNewsNet", args.bucket, args.project)
        
        logger.info("SUCCESS - All datasets uploaded to S3.")
        
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        exit(1)

if __name__ == "__main__":
    main()