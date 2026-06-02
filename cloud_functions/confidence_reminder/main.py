import functions_framework
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@functions_framework.http
def send_reminder(request):
    """Send confidence threshold calibration reminder email."""
    
    subject = "🔔 TFT Trading Brains: Calibrate Confidence Thresholds"
    
    body = """Hi,

It's been 2 weeks since you deployed quantile-based confidence scoring to your TFT prediction pipeline.

You now have enough data to calibrate the signal_strength thresholds in v_signals.

Run this query in BigQuery Console to check the distribution:

---

SELECT
  ticker,
  COUNT(*) AS num_predictions,
  MIN(confidence_60m) AS min_conf,
  APPROX_QUANTILES(confidence_60m, 10)[OFFSET(1)] AS p10,
  APPROX_QUANTILES(confidence_60m, 10)[OFFSET(2)] AS p20,
  APPROX_QUANTILES(confidence_60m, 10)[OFFSET(5)] AS median,
  APPROX_QUANTILES(confidence_60m, 10)[OFFSET(7)] AS p70,
  APPROX_QUANTILES(confidence_60m, 10)[OFFSET(9)] AS p90,
  MAX(confidence_60m) AS max_conf,
  AVG(ABS(confidence_60m)) AS avg_abs_conf
FROM `trading-brains.tft_predictions.tft_predictions_logs`
WHERE confidence_60m IS NOT NULL
  AND TIMESTAMP(timestamp) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
GROUP BY ticker
ORDER BY ticker;

---

Then update the thresholds in v_signals:

  STRONG_BUY threshold  → set to ~p75 of positive confidence values
  STRONG_SELL threshold → set to ~p25 of negative confidence values (mirror)
  BUY threshold         → set to ~p50 (median) of positive confidence
  SELL threshold        → set to ~p50 of negative confidence

Current thresholds: STRONG = |8|, BUY/SELL = |3|

Also check signal accuracy with:

SELECT
  signal_strength,
  COUNT(*) AS count,
  AVG(signal_correct) AS accuracy,
  AVG(signal_pnl_pct) AS avg_pnl
FROM `trading-brains.tft_predictions.v_signal_performance`
WHERE prediction_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
GROUP BY signal_strength
ORDER BY signal_strength;

If STRONG_BUY accuracy < 55%, tighten thresholds.
If it fires < 5% of the time, consider loosening them.

---

This was an automated reminder from your trading_brains pipeline.
"""

    # Use SendGrid-style or direct SMTP
    # For Cloud Functions, we'll use the Mailgun/SendGrid approach
    # But simplest: use Google Cloud's built-in email via App Engine mail or a simple SMTP relay
    
    # Using Gmail SMTP with App Password (set as env var)
    import os
    sender_email = os.environ.get("SENDER_EMAIL", "batz.iam@gmail.com")
    sender_password = os.environ.get("SENDER_APP_PASSWORD")
    recipient = "batz.iam@gmail.com"
    
    if not sender_password:
        # Fallback: just log the reminder (useful if SMTP not configured)
        print(f"REMINDER (no SMTP configured):\nTo: {recipient}\nSubject: {subject}\n\n{body}")
        return f"Reminder logged (no SMTP password set). Would have sent to {recipient}", 200
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()
        return f"Reminder email sent to {recipient}", 200
    except Exception as e:
        print(f"Email send failed: {e}")
        return f"Failed to send: {e}", 500
