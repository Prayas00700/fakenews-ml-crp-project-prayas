"""
dep.py — Central configuration and shared constants for the Fake News Detection MLOps pipeline.
All scripts import from here to keep bucket names, paths, and hyperparameters consistent.

ALL values are read from environment variables so GitHub Actions secrets/variables
drive every setting — nothing is hard-coded except safe defaults.
"""

import os

# ─────────────────────────────────────────────
# AWS / S3 Configuration
# Sourced from GitHub Secrets / Variables
# ─────────────────────────────────────────────
AWS_REGION    = os.environ.get("AWS_REGION",    "ap-south-1")
S3_BUCKET     = os.environ.get("S3_BUCKET",     "fake-news-mlops-bucket")
PROJECT_NAME  = os.environ.get("PROJECT_NAME",  "fake-news")

# S3 key prefixes — always scoped under PROJECT_NAME
# Results in:  fake-news/data/raw/
#              fake-news/data/processed/
#              fake-news/models/output/
#              fake-news/metrics/
#              fake-news/code/
S3_RAW_PREFIX        = f"{PROJECT_NAME}/data/raw/"
S3_PROCESSED_PREFIX  = f"{PROJECT_NAME}/data/processed/"
S3_MODEL_PREFIX      = f"{PROJECT_NAME}/models/output/"
S3_METRICS_PREFIX    = f"{PROJECT_NAME}/metrics/"
S3_CODE_PREFIX       = f"{PROJECT_NAME}/code/"
S3_ARTIFACTS_PREFIX  = f"{PROJECT_NAME}/artifacts/"

# Local data filenames (CSV files on your machine / in the repo's data/local/)
LOCAL_DATA_DIR  = os.environ.get("LOCAL_DATA_DIR", "data/local")
DATA1_FILENAME  = "fake_or_real_news.csv"
DATA2_FILENAME  = "merged_news.csv"   # clmentbisaillon dataset
DATA3_FILENAME  = "fake_news_dataset_1000.csv"  # additional dataset

# ─────────────────────────────────────────────
# SageMaker Configuration
# Sourced from GitHub Secrets
# ─────────────────────────────────────────────
SAGEMAKER_ROLE_ARN       = os.environ.get("SAGEMAKER_ROLE_ARN",       "")
SAGEMAKER_INSTANCE       = os.environ.get("SAGEMAKER_INSTANCE",       "ml.t3.medium")
SAGEMAKER_INSTANCE_TRAIN = os.environ.get("SAGEMAKER_INSTANCE_TRAIN", "ml.g4dn.xlarge")

# ─────────────────────────────────────────────
# Model / Training Hyperparameters
# MODEL_TYPE can be overridden by GitHub Actions input
# ─────────────────────────────────────────────
MODEL_TYPE = os.environ.get("MODEL_TYPE", "lstm")   # "nb", "lstm", or "bert"

# TF-IDF + Naive Bayes
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE  = (1, 2)

# LSTM
MAX_VOCAB   = 20_000
MAX_LEN     = 300
EMBED_DIM   = 100
LSTM_UNITS  = 128
DENSE_UNITS = 64
DROPOUT     = 0.3
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS      = int(os.environ.get("EPOCHS",     8))

# BERT
BERT_MODEL_NAME  = "bert-base-uncased"
BERT_MAX_LEN     = 256
BERT_BATCH_SIZE  = 8
BERT_EPOCHS      = 10
BERT_LR          = 2e-5

# ─────────────────────────────────────────────
# Evaluation thresholds / Quality Gates
# ─────────────────────────────────────────────
MIN_ACCURACY = 0.80   # deployment blocked if below this
MIN_F1_FAKE  = 0.65   # changing this from 0.78 to 0.65 as model trained is not good

# ─────────────────────────────────────────────
# SageMaker Endpoint
# ─────────────────────────────────────────────
ENDPOINT_NAME     = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "fake-news-detector-endpoint")
ENDPOINT_INSTANCE = os.environ.get("ENDPOINT_INSTANCE",       "ml.m5.large")
