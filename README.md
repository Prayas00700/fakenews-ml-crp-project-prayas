# Fake News Detection — MLOps Pipeline

End-to-end ML pipeline for fake news detection using **GitHub Actions** + **Amazon SageMaker**.
Mirrors the heart-disease pipeline pattern.

---

## Pipeline Flow

```
[1] Upload CSVs → S3 raw
        ↓
[2] SageMaker Processing Job → S3 processed
        ↓
[3] SageMaker Training Job   → S3 models/output
        ↓
[4] Model Evaluation + Quality Gates
        ↓ (only if accuracy ≥ 0.80 and F1 ≥ 0.78)
[5] SageMaker Endpoint (production approval required)
        ↓
[6] Model Monitor — hourly drift detection
```

### S3 Folder Structure

All data is scoped under `$PROJECT_NAME`:

```
s3://<S3_BUCKET>/
└── <PROJECT_NAME>/           e.g. fake-news
    ├── data/
    │   ├── raw/              fake_or_real_news_<ts>.csv, FakeNewsNet_<ts>.csv
    │   └── processed/        processed_dataset_<ts>.parquet, *.csv, stats.json
    ├── models/output/        model.tar.gz from SageMaker training
    ├── metrics/              eval_metrics.json, monitor_report_latest.json
    ├── code/                 preprocessing + training scripts uploaded by GH Actions
    └── monitoring/           data-capture, baseline-results, monitor-output
```

---

## GitHub Secrets & Variables

Go to **Settings → Secrets and variables → Actions**.

### Secrets (sensitive)
| Secret | Example |
|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::123456789:role/GitHubActionsRole` |
| `S3_BUCKET` | `fake-news-mlops-bucket` |
| `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::123456789:role/SageMakerExecutionRole` |
| `SAGEMAKER_ENDPOINT_NAME` | `fake-news-detector-endpoint` |

### Variables (non-sensitive, visible in logs)
| Variable | Example |
|---|---|
| `AWS_REGION` | `ap-south-1` |
| `PROJECT_NAME` | `fake-news` |
| `SAGEMAKER_INSTANCE_TRAIN` | `ml.g4dn.xlarge` |
| `SAGEMAKER_INSTANCE` | `ml.m5.xlarge` |
| `ENDPOINT_INSTANCE` | `ml.m5.large` |

---

## Supported Models

| MODEL_TYPE | Algorithm | SageMaker Instance |
|---|---|---|
| `nb` | TF-IDF + Complement Naive Bayes | ml.m5.xlarge (CPU) |
| `lstm` | Bidirectional LSTM (TensorFlow) | ml.g4dn.xlarge (GPU) |
| `bert` | BERT fine-tuning (HuggingFace) | ml.g4dn.xlarge (GPU) |

---

## Quick Start

1. Add all GitHub Secrets & Variables listed above
2. Place your CSV files:
   ```
   data/local/
   ├── fake_or_real_news.csv
   └── FakeNewsNet.csv
   ```
3. Commit and push to `main` — workflow 1 fires automatically
4. Or trigger any workflow manually via **Actions → Run workflow**
