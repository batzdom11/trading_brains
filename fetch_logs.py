"""Fetch logs for a failed Vertex AI training job."""
import subprocess
import sys

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "7842206211972792320"

# Try with quotes inside the filter
filter_str = 'resource.type="ml_job"\nresource.labels.job_id="' + JOB_ID + '"'

cmd = [
    r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
    "logging", "read",
    filter_str,
    "--project=trading-brains",
    "--limit=40",
    "--freshness=168h",
    "--format=value(textPayload)",
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.stdout:
    print(result.stdout[:5000])
else:
    print("No stdout output")
if result.stderr:
    print("STDERR:", result.stderr[:500])
