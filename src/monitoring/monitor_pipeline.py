"""
src/monitoring/monitor_pipeline.py
────────────────────────────────────
Sets up Amazon SageMaker Model Monitor on the deployed endpoint:
 1. Enables data capture on the endpoint.
 2. Creates a baseline from the training dataset.
 3. Schedules a monitoring job to detect data drift hourly.
 4. Fetches the latest monitoring report and logs any violations.

Run after deployment:
    python src/monitoring/monitor_pipeline.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import boto3
import sagemaker
from sagemaker.model_monitor import (
    DefaultModelMonitor,
    DataCaptureConfig,
    CronExpressionGenerator,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dep import (
    AWS_REGION,
    S3_BUCKET,
    PROJECT_NAME,
    SAGEMAKER_ROLE_ARN,
    ENDPOINT_NAME,
    ENDPOINT_INSTANCE,
)

# S3 prefixes derived from PROJECT_NAME
S3_PROCESSED_PREFIX = os.environ.get("S3_PROCESSED_PREFIX", f"{PROJECT_NAME}/data/processed/")
S3_METRICS_PREFIX   = os.environ.get("S3_METRICS_PREFIX",   f"{PROJECT_NAME}/metrics/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# S3 paths for monitoring artefacts — all scoped under PROJECT_NAME
DATA_CAPTURE_URI    = f"s3://{S3_BUCKET}/{PROJECT_NAME}/monitoring/data-capture/"
BASELINE_DATA_URI   = f"s3://{S3_BUCKET}/{S3_PROCESSED_PREFIX}processed_dataset.csv"
BASELINE_RESULT_URI = f"s3://{S3_BUCKET}/{PROJECT_NAME}/monitoring/baseline-results/"
MONITOR_OUTPUT_URI  = f"s3://{S3_BUCKET}/{PROJECT_NAME}/monitoring/monitor-output/"


def get_session() -> sagemaker.Session:
    return sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Enable data capture on the endpoint
# ─────────────────────────────────────────────────────────────────────────────

def enable_data_capture(session: sagemaker.Session) -> None:
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)

    log.info("Enabling data capture on endpoint: %s", ENDPOINT_NAME)
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=DATA_CAPTURE_URI,
        capture_options=["Input", "Output"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"],
    )

    # Re-deploy with data capture enabled (update endpoint config)
    sm_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=_get_or_create_capture_config(sm_client, data_capture_config, session),
    )
    log.info("Data capture enabled. Capturing to %s", DATA_CAPTURE_URI)


def _get_or_create_capture_config(sm_client, data_capture_config, session) -> str:
    """Returns an EndpointConfigName that has data capture enabled."""
    timestamp  = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    config_name = f"fake-news-endpoint-config-{timestamp}"

    # Get current endpoint config to copy instance details
    ep_desc    = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    old_config = sm_client.describe_endpoint_config(
        EndpointConfigName=ep_desc["EndpointConfigName"]
    )
    prod_variants = old_config["ProductionVariants"]

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=prod_variants,
        DataCaptureConfig={
            "EnableCapture": data_capture_config.enable_capture,
            "InitialSamplingPercentage": data_capture_config.sampling_percentage,
            "DestinationS3Uri": data_capture_config.destination_s3_uri,
            "CaptureOptions": [{"CaptureMode": m} for m in ["Input", "Output"]],
            "CaptureContentTypeHeader": {
                "CsvContentTypes":  ["text/csv"],
                "JsonContentTypes": ["application/json"],
            },
        },
    )
    return config_name


# ─────────────────────────────────────────────────────────────────────────────
# 2. Create a monitoring baseline from the training data
# ─────────────────────────────────────────────────────────────────────────────

def create_baseline(session: sagemaker.Session) -> DefaultModelMonitor:
    monitor = DefaultModelMonitor(
        role=SAGEMAKER_ROLE_ARN,
        instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )

    log.info("Suggesting baseline constraints from: %s", BASELINE_DATA_URI)
    monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATA_URI,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=BASELINE_RESULT_URI,
        wait=True,
        logs=False,
    )
    log.info("Baseline created at: %s", BASELINE_RESULT_URI)
    return monitor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Schedule a monitoring job
# ─────────────────────────────────────────────────────────────────────────────

def schedule_monitoring(monitor: DefaultModelMonitor) -> None:
    monitor_name = f"fake-news-monitor-{datetime.utcnow().strftime('%Y%m%d')}"
    log.info("Creating monitoring schedule: %s", monitor_name)

    monitor.create_monitoring_schedule(
        monitor_schedule_name=monitor_name,
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=MONITOR_OUTPUT_URI,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    log.info("Monitoring schedule created (runs hourly).")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fetch and log the latest monitoring report
# ─────────────────────────────────────────────────────────────────────────────

def fetch_monitoring_report(session: sagemaker.Session) -> None:
    s3 = boto3.client("s3", region_name=AWS_REGION)

    prefix = "monitoring/monitor-output/"
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    objects  = response.get("Contents", [])

    if not objects:
        log.info("No monitoring reports found yet.")
        return

    # Latest report
    latest = sorted(objects, key=lambda o: o["LastModified"], reverse=True)[0]
    log.info("Latest monitoring report: s3://%s/%s", S3_BUCKET, latest["Key"])

    obj  = s3.get_object(Bucket=S3_BUCKET, Key=latest["Key"])
    body = obj["Body"].read().decode()

    try:
        report = json.loads(body)
        violations = report.get("violations", [])
        if violations:
            log.warning("⚠ Data drift violations detected (%d):", len(violations))
            for v in violations:
                log.warning("   feature=%s  type=%s  description=%s",
                            v.get("feature_name"), v.get("constraint_check_type"), v.get("description"))
        else:
            log.info("No violations found in latest monitoring report.")

        # Save report summary to S3 metrics prefix
        summary_key = f"{S3_METRICS_PREFIX}monitor_report_latest.json"
        s3.put_object(Bucket=S3_BUCKET, Key=summary_key, Body=json.dumps(report, indent=2))
        log.info("Report summary saved to s3://%s/%s", S3_BUCKET, summary_key)

    except json.JSONDecodeError:
        log.info("Report content: %s", body[:500])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    action = os.environ.get("MONITOR_ACTION", "full").lower()
    # MONITOR_ACTION can be:
    #   "full"    — enable capture + baseline + schedule + report
    #   "report"  — only fetch and log the latest report (for scheduled runs)

    session = get_session()

    if action == "report":
        log.info("Fetching latest monitoring report…")
        fetch_monitoring_report(session)
        return

    # Full setup
    enable_data_capture(session)
    monitor = create_baseline(session)
    schedule_monitoring(monitor)
    fetch_monitoring_report(session)
    log.info("Monitoring pipeline setup complete.")


if __name__ == "__main__":
    main()
