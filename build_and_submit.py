"""Build container and submit training jobs."""
import subprocess
import sys
import time

GCLOUD = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"


def run(args, timeout=600):
    """Run a gcloud command and return output."""
    print(f"Running: {' '.join(args[1:])}")
    result = subprocess.run(
        [GCLOUD] + args,
        capture_output=True, text=True, timeout=timeout
    )
    if result.stdout:
        print(result.stdout[-3000:])
    if result.stderr:
        # Filter out update notices
        lines = [l for l in result.stderr.split('\n') 
                 if not any(x in l for x in ['Updates are available', '$ gcloud', 'survey', 'endpoint'])]
        err = '\n'.join(lines).strip()
        if err:
            print(f"STDERR: {err}")
    return result.returncode


def build():
    """Build the training container."""
    print("=" * 60)
    print("BUILDING CONTAINER")
    print("=" * 60)
    rc = run([
        "builds", "submit",
        "--tag", "gcr.io/trading-brains/tft-training",
        "--project", "trading-brains",
        "pipeline/"
    ], timeout=900)
    if rc != 0:
        print("BUILD FAILED!")
        sys.exit(1)
    print("BUILD SUCCEEDED!")
    return True


def submit_job(ticker):
    """Submit a training job for a ticker."""
    import json
    import tempfile
    
    print(f"\nSubmitting job for {ticker}...")
    
    # Write a config JSON to avoid gcloud --args comma parsing issues
    config = {
        "workerPoolSpecs": [{
            "machineSpec": {
                "machineType": "n1-highmem-8",
            },
            "replicaCount": 1,
            "containerSpec": {
                "imageUri": "gcr.io/trading-brains/tft-training",
                "args": [
                    "--polygon_api_key", "IYHqABfrpYN6yfa7bFS9LLNxwJzpn0YE",
                    "--gcs_bucket", "tft-for-trading-brains",
                    "--gcs_model_path", f"tft_checkpoint_{ticker}_latest.ckpt",
                    "--symbol", ticker,
                    "--lookback_days", "420",
                    "--max_epochs", "25",
                    "--batch_size", "128",
                    "--learning_rate", "0.001",
                    "--hidden_size", "64",
                    "--attention_head_size", "4",
                    "--dropout", "0.1",
                    "--hidden_continuous_size", "32",
                    "--patience", "10",
                    "--num_workers", "4",
                ],
            },
        }]
    }
    
    config_path = os.path.join(tempfile.gettempdir(), f"tft_job_{ticker.lower()}.yaml")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    rc = run([
        "ai", "custom-jobs", "create",
        "--project=trading-brains",
        "--region=europe-west4",
        f"--display-name=tft-training-{ticker.lower()}",
        f"--config={config_path}",
    ])
    return rc == 0


if __name__ == "__main__":
    import os
    os.chdir(r"c:\Users\batzi\OneDrive\Dokumente\trading_brains")
    
    if "--build" in sys.argv or "--all" in sys.argv:
        build()
    
    if "--submit" in sys.argv or "--all" in sys.argv:
        default_tickers = ["SPY", "GOOG", "QQQ", "TSLA", "AAPL"]
        # Allow --tickers SPY,QQQ,TSLA to override
        tickers = default_tickers
        for arg in sys.argv:
            if arg.startswith("--tickers="):
                tickers = arg.split("=")[1].split(",")
        stagger = 60  # 1 minute between jobs (GPU jobs start fast)
        
        for i, ticker in enumerate(tickers):
            if i > 0:
                print(f"Waiting {stagger}s before next job...")
                time.sleep(stagger)
            submit_job(ticker)
    
    if "--status" in sys.argv:
        run([
            "ai", "custom-jobs", "list",
            "--project=trading-brains",
            "--region=europe-west4",
            "--sort-by=~createTime",
            "--limit=10",
            "--format=table(displayName,state,createTime)"
        ])
