"""
Create BigQuery views for the Looker Studio dashboard.
Run once to set up the views, then connect Looker Studio to them.
"""
from google.cloud import bigquery

client = bigquery.Client(project="trading-brains")
dataset = "trading-brains.tft_predictions"


def create_view(view_id, sql):
    """Create or replace a BQ view."""
    full_id = f"{dataset}.{view_id}"
    view = bigquery.Table(full_id)
    view.view_query = sql
    try:
        client.delete_table(full_id, not_found_ok=True)
    except Exception:
        pass
    client.create_table(view)
    print(f"Created view: {full_id}")


# ============================================================
# VIEW 1: Latest predictions — predicted % change per ticker
# Shows the most recent prediction for each ticker with
# predicted percentage rise/fall at 15/30/45/60 minutes.
# ============================================================
create_view("vw_latest_predictions", """
SELECT
  ticker,
  TIMESTAMP(timestamp) AS prediction_time,
  last_price,
  ROUND(return_15m, 3) AS pct_change_15m,
  ROUND(return_30m, 3) AS pct_change_30m,
  ROUND(return_45m, 3) AS pct_change_45m,
  ROUND(return_60m, 3) AS pct_change_60m,
  ROUND(pred_15m, 2) AS pred_15m,
  ROUND(pred_30m, 2) AS pred_30m,
  ROUND(pred_45m, 2) AS pred_45m,
  ROUND(pred_60m, 2) AS pred_60m,
  CASE WHEN return_60m >= 0 THEN 'UP' ELSE 'DOWN' END AS direction_60m
FROM `trading-brains.tft_predictions.tft_predictions_logs`
QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) = 1
""")


# ============================================================
# VIEW 2: Full prediction history — for time-series charts
# All predictions with timestamps parsed, for line charts.
# ============================================================
create_view("vw_prediction_history", """
SELECT
  ticker,
  TIMESTAMP(timestamp) AS prediction_time,
  TIMESTAMP(last_price_time) AS base_time,
  last_price,
  pred_15m,
  pred_30m,
  pred_45m,
  pred_60m,
  return_15m AS pct_change_15m,
  return_30m AS pct_change_30m,
  return_45m AS pct_change_45m,
  return_60m AS pct_change_60m
FROM `trading-brains.tft_predictions.tft_predictions_logs`
WHERE ticker IS NOT NULL
""")


# ============================================================
# VIEW 3: Actual vs Predicted — joined for line chart overlay
# Each row has both the predicted and actual price at each
# horizon, plus the error. X-axis = prediction_time.
# ============================================================
create_view("vw_actual_vs_predicted", """
SELECT
  p.ticker,
  TIMESTAMP(p.timestamp) AS prediction_time,
  TIMESTAMP(p.last_price_time) AS base_time,
  p.last_price AS base_price,

  -- 15-minute horizon
  ROUND(p.pred_15m, 2) AS pred_15m,
  a.actual_15m_price,
  ROUND(p.pred_15m - a.actual_15m_price, 2) AS error_15m,
  ROUND(ABS(p.pred_15m - a.actual_15m_price) / a.actual_15m_price * 100, 3) AS pct_error_15m,

  -- 30-minute horizon
  ROUND(p.pred_30m, 2) AS pred_30m,
  a.actual_30m_price,
  ROUND(p.pred_30m - a.actual_30m_price, 2) AS error_30m,
  ROUND(ABS(p.pred_30m - a.actual_30m_price) / a.actual_30m_price * 100, 3) AS pct_error_30m,

  -- 45-minute horizon
  ROUND(p.pred_45m, 2) AS pred_45m,
  a.actual_45m_price,
  ROUND(p.pred_45m - a.actual_45m_price, 2) AS error_45m,
  ROUND(ABS(p.pred_45m - a.actual_45m_price) / a.actual_45m_price * 100, 3) AS pct_error_45m,

  -- 60-minute horizon
  ROUND(p.pred_60m, 2) AS pred_60m,
  a.actual_60m_price,
  ROUND(p.pred_60m - a.actual_60m_price, 2) AS error_60m,
  ROUND(ABS(p.pred_60m - a.actual_60m_price) / a.actual_60m_price * 100, 3) AS pct_error_60m,

  -- Direction accuracy
  CASE WHEN (p.pred_60m - p.last_price) * (a.actual_60m_price - p.last_price) > 0
       THEN 'Correct' ELSE 'Wrong' END AS direction_60m

FROM `trading-brains.tft_predictions.tft_predictions_logs` p
INNER JOIN `trading-brains.tft_predictions.tft_actuals` a
  ON p.timestamp = a.prediction_timestamp
  AND p.ticker = a.ticker
WHERE p.ticker IS NOT NULL
""")


# ============================================================
# VIEW 4: Model health — rolling accuracy metrics over time
# Aggregates error metrics into daily/hourly windows per ticker.
# ============================================================
create_view("vw_model_health", """
WITH paired AS (
  SELECT
    p.ticker,
    TIMESTAMP(p.timestamp) AS prediction_time,
    DATE(TIMESTAMP(p.timestamp)) AS prediction_date,
    p.last_price,
    p.pred_15m, a.actual_15m_price,
    p.pred_30m, a.actual_30m_price,
    p.pred_45m, a.actual_45m_price,
    p.pred_60m, a.actual_60m_price,
    ABS(p.pred_15m - a.actual_15m_price) AS ae_15m,
    ABS(p.pred_30m - a.actual_30m_price) AS ae_30m,
    ABS(p.pred_45m - a.actual_45m_price) AS ae_45m,
    ABS(p.pred_60m - a.actual_60m_price) AS ae_60m,
    ABS(p.pred_60m - a.actual_60m_price) / a.actual_60m_price * 100 AS mape_60m,
    CASE WHEN (p.pred_60m - p.last_price) * (a.actual_60m_price - p.last_price) > 0
         THEN 1 ELSE 0 END AS direction_correct_60m
  FROM `trading-brains.tft_predictions.tft_predictions_logs` p
  INNER JOIN `trading-brains.tft_predictions.tft_actuals` a
    ON p.timestamp = a.prediction_timestamp
    AND p.ticker = a.ticker
  WHERE p.ticker IS NOT NULL
)
SELECT
  ticker,
  prediction_date,
  COUNT(*) AS n_predictions,

  -- Mean Absolute Error by horizon
  ROUND(AVG(ae_15m), 4) AS mae_15m,
  ROUND(AVG(ae_30m), 4) AS mae_30m,
  ROUND(AVG(ae_45m), 4) AS mae_45m,
  ROUND(AVG(ae_60m), 4) AS mae_60m,

  -- Average % error at 60m
  ROUND(AVG(mape_60m), 4) AS avg_pct_error_60m,

  -- Direction accuracy at 60m
  ROUND(AVG(direction_correct_60m) * 100, 1) AS direction_accuracy_pct,

  -- Best and worst absolute error at 60m
  ROUND(MIN(ae_60m), 4) AS best_ae_60m,
  ROUND(MAX(ae_60m), 4) AS worst_ae_60m

FROM paired
GROUP BY ticker, prediction_date
ORDER BY prediction_date DESC, ticker
""")


# ============================================================
# VIEW 5: Per-prediction detail for health scorecards
# Individual prediction-level metrics for drill-down.
# ============================================================
create_view("vw_prediction_detail", """
SELECT
  p.ticker,
  TIMESTAMP(p.timestamp) AS prediction_time,
  p.last_price AS base_price,

  p.pred_15m, a.actual_15m_price,
  p.pred_30m, a.actual_30m_price,
  p.pred_45m, a.actual_45m_price,
  p.pred_60m, a.actual_60m_price,

  ROUND(ABS(p.pred_15m - a.actual_15m_price), 4) AS ae_15m,
  ROUND(ABS(p.pred_30m - a.actual_30m_price), 4) AS ae_30m,
  ROUND(ABS(p.pred_45m - a.actual_45m_price), 4) AS ae_45m,
  ROUND(ABS(p.pred_60m - a.actual_60m_price), 4) AS ae_60m,

  ROUND(ABS(p.pred_60m - a.actual_60m_price) / a.actual_60m_price * 100, 4) AS pct_error_60m,

  CASE WHEN (p.pred_60m - p.last_price) * (a.actual_60m_price - p.last_price) > 0
       THEN 1 ELSE 0 END AS direction_correct_60m,

  p.return_15m AS predicted_pct_15m,
  p.return_30m AS predicted_pct_30m,
  p.return_45m AS predicted_pct_45m,
  p.return_60m AS predicted_pct_60m,

  ROUND((a.actual_15m_price - p.last_price) / p.last_price * 100, 4) AS actual_pct_15m,
  ROUND((a.actual_30m_price - p.last_price) / p.last_price * 100, 4) AS actual_pct_30m,
  ROUND((a.actual_45m_price - p.last_price) / p.last_price * 100, 4) AS actual_pct_45m,
  ROUND((a.actual_60m_price - p.last_price) / p.last_price * 100, 4) AS actual_pct_60m

FROM `trading-brains.tft_predictions.tft_predictions_logs` p
INNER JOIN `trading-brains.tft_predictions.tft_actuals` a
  ON p.timestamp = a.prediction_timestamp
  AND p.ticker = a.ticker
WHERE p.ticker IS NOT NULL
""")


print("\nAll views created successfully!")
print("Connect Looker Studio to these views as data sources:")
print("  1. vw_latest_predictions   → Scorecard / Table section")
print("  2. vw_prediction_history   → Prediction % change line chart")
print("  3. vw_actual_vs_predicted  → Actual vs Predicted line charts")
print("  4. vw_model_health         → Daily model health metrics")
print("  5. vw_prediction_detail    → Per-prediction drill-down")
