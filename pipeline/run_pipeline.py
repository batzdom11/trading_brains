"""
Vertex AI Pipeline for TFT model retraining.

Submits a Custom Training Job using a pre-built container image that:
  1. Fetches SPY 1-min data from Polygon.io
  2. Engineers 76 features
  3. Trains a TemporalFusionTransformer
  4. Uploads the best checkpoint to GCS

Usage:
  # First time: build and push the training container
  python run_pipeline.py --build

  # Then submit a training run
  python run_pipeline.py --run

  # Or do both
  python run_pipeline.py --build --run

  # Override training parameters
  python run_pipeline.py --run --max_epochs 20 --lookback_days 180
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# Clear stale GOOGLE_APPLICATION_CREDENTIALS if the file doesn't exist
_cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if _cred_path and not os.path.exists(_cred_path):
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

PROJECT_ID = "trading-brains"
REGION = "europe-west6"
GCS_BUCKET = "tft-for-trading-brains"
GCS_MODEL_PATH = "tft_checkpoint_latest.ckpt"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/tft-training"
STAGING_BUCKET = f"gs://{GCS_BUCKET}"


def build_container():
    """Build and push the training container image."""
    print("Building training container...")
    pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        "gcloud", "builds", "submit",
        "--tag", IMAGE_URI,
        "--project", PROJECT_ID,
        pipeline_dir,
    ]
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)
    print("Container built and pushed successfully.")


def run_pipeline(args):
    """Submit a Vertex AI Custom Training Job."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"tft-training-{timestamp}"

    # Build the training arguments
    train_args = [
        "--polygon_api_key", args.polygon_api_key,
        "--gcs_bucket", GCS_BUCKET,
        "--gcs_model_path", GCS_MODEL_PATH,
        "--symbol", args.symbol,
        "--lookback_days", str(args.lookback_days),
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--hidden_size", str(args.hidden_size),
        "--attention_head_size", str(args.attention_head_size),
        "--dropout", str(args.dropout),
        "--hidden_continuous_size", str(args.hidden_continuous_size),
        "--patience", str(args.patience),
    ]

    # Choose machine type based on whether GPU is requested
    if args.gpu:
        machine_type = "n1-standard-8"
        accelerator_type = "NVIDIA_TESLA_T4"
        accelerator_count = 1
    else:
        machine_type = args.machine_type
        accelerator_type = None
        accelerator_count = None

    print(f"\nSubmitting Vertex AI Custom Training Job:")
    print(f"  Display name:  {display_name}")
    print(f"  Image:         {IMAGE_URI}")
    print(f"  Machine type:  {machine_type}")
    print(f"  GPU:           {accelerator_type or 'None'}")
    print(f"  Max epochs:    {args.max_epochs}")
    print(f"  Lookback days: {args.lookback_days}")
    print(f"  Batch size:    {args.batch_size}")
    print()

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=IMAGE_URI,
        staging_bucket=STAGING_BUCKET,
    )

    # Build run kwargs
    run_kwargs = dict(
        args=train_args,
        replica_count=1,
        machine_type=machine_type,
        sync=False,
    )
    if accelerator_type:
        run_kwargs["accelerator_type"] = accelerator_type
        run_kwargs["accelerator_count"] = accelerator_count

    job.run(**run_kwargs)

    # Wait for job to be registered in Vertex AI before exiting
    job.wait_for_resource_creation()
    print(f"\nTraining job submitted: {display_name}")
    print(f"Job resource name: {job.resource_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}&region={REGION}")


def main():
    parser = argparse.ArgumentParser(description="TFT Training Pipeline for Vertex AI")

    # Actions
    parser.add_argument("--build", action="store_true", help="Build and push the training container")
    parser.add_argument("--run", action="store_true", help="Submit a training job to Vertex AI")
    parser.add_argument("--run_all", action="store_true", help="Submit training jobs for all 5 tickers with stagger delay")
    parser.add_argument("--stagger_minutes", type=int, default=5, help="Minutes to wait between job submissions (for --run_all)")

    # API key (required for --run)
    parser.add_argument("--polygon_api_key", default=None, help="Polygon.io API key")

    # Training parameters
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--lookback_days", type=int, default=420)
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--attention_head_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_continuous_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)

    # Infrastructure
    parser.add_argument("--machine_type", default="n1-standard-4", help="Vertex AI machine type")
    parser.add_argument("--gpu", action="store_true", help="Use a T4 GPU")

    args = parser.parse_args()

    if not args.build and not args.run and not args.run_all:
        parser.print_help()
        sys.exit(1)

    if args.build:
        build_container()

    if args.run:
        if not args.polygon_api_key:
            # Try environment variable
            import os
            args.polygon_api_key = os.environ.get("POLYGON_API_KEY")
            if not args.polygon_api_key:
                print("ERROR: --polygon_api_key is required (or set POLYGON_API_KEY env var)")
                sys.exit(1)
        run_pipeline(args)

    if args.run_all:
        if not args.polygon_api_key:
            import os
            args.polygon_api_key = os.environ.get("POLYGON_API_KEY")
            if not args.polygon_api_key:
                print("ERROR: --polygon_api_key is required (or set POLYGON_API_KEY env var)")
                sys.exit(1)
        tickers = ["SPY", "GOOG", "QQQ", "TSLA", "AAPL"]
        for i, ticker in enumerate(tickers):
            args.symbol = ticker
            print(f"\n{'='*60}")
            print(f"Submitting job {i+1}/{len(tickers)}: {ticker}")
            print(f"{'='*60}")
            run_pipeline(args)
            if i < len(tickers) - 1:
                wait = args.stagger_minutes * 60
                print(f"\nWaiting {args.stagger_minutes} minutes before next submission...")
                time.sleep(wait)


if __name__ == "__main__":
    main()
