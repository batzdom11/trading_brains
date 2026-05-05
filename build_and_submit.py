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
    print(f"\nSubmitting job for {ticker}...")
    rc = run([
        "ai", "custom-jobs", "create",
        "--project=trading-brains",
        "--region=europe-west6",
        f"--display-name=tft-training-{ticker.lower()}",
        "--worker-pool-spec="
        "machine-type=n1-standard-4,"
        "replica-count=1,"
        "container-image-uri=gcr.io/trading-brains/tft-training",
        f"--args=--polygon_api_key,IYHqABfrpYN6yfa7bFS9LLNxwJzpn0YE,"
        f"--gcs_bucket,tft-for-trading-brains,"
        f"--gcs_model_path,tft_checkpoint_{ticker}_latest.ckpt,"
        f"--symbol,{ticker},"
        f"--lookback_days,420,"
        f"--max_epochs,25,"
        f"--batch_size,64,"
        f"--learning_rate,0.001,"
        f"--hidden_size,64,"
        f"--attention_head_size,4,"
        f"--dropout,0.1,"
        f"--hidden_continuous_size,32,"
        f"--patience,10",
    ])
    return rc == 0


if __name__ == "__main__":
    import os
    os.chdir(r"c:\Users\batzi\OneDrive\Dokumente\trading_brains")
    
    if "--build" in sys.argv or "--all" in sys.argv:
        build()
    
    if "--submit" in sys.argv or "--all" in sys.argv:
        tickers = ["SPY", "GOOG", "QQQ", "TSLA", "AAPL"]
        stagger = 300  # 5 minutes between jobs
        
        for i, ticker in enumerate(tickers):
            if i > 0:
                print(f"Waiting {stagger}s before next job...")
                time.sleep(stagger)
            submit_job(ticker)
    
    if "--status" in sys.argv:
        run([
            "ai", "custom-jobs", "list",
            "--project=trading-brains",
            "--region=europe-west6",
            "--sort-by=~createTime",
            "--limit=10",
            "--format=table(displayName,state,createTime)"
        ])
