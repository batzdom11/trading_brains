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


# ============================================================
# VIEW 6: Market status — open/closed indicator & countdown
# Returns a single row with market state, next prediction time,
# and human-readable countdown. Use as Looker Studio scorecards.
# Predictions fire at :00 from 10am-4pm ET, Mon-Fri.
# ============================================================
create_view("vw_market_status", """
WITH params AS (
  SELECT
    CURRENT_TIMESTAMP() AS utc_now,
    TIMESTAMP(DATETIME(CURRENT_TIMESTAMP(), 'America/New_York')) AS et_ts,
    EXTRACT(HOUR FROM DATETIME(CURRENT_TIMESTAMP(), 'America/New_York')) AS et_hour,
    EXTRACT(MINUTE FROM DATETIME(CURRENT_TIMESTAMP(), 'America/New_York')) AS et_minute,
    EXTRACT(DAYOFWEEK FROM DATETIME(CURRENT_TIMESTAMP(), 'America/New_York')) AS et_dow,
    DATE(DATETIME(CURRENT_TIMESTAMP(), 'America/New_York')) AS et_date,
    DATETIME(CURRENT_TIMESTAMP(), 'America/New_York') AS et_datetime
),
market_times AS (
  SELECT *,
    -- Market open: 9:30 ET, close: 4:00 ET
    -- Predictions run 10:00-16:00 ET
    CASE
      WHEN et_dow IN (1, 7) THEN FALSE  -- Sunday=1, Saturday=7
      WHEN et_hour < 9 THEN FALSE
      WHEN et_hour = 9 AND et_minute < 30 THEN FALSE
      WHEN et_hour >= 16 THEN FALSE
      ELSE TRUE
    END AS is_market_open,
    CASE
      WHEN et_dow IN (1, 7) THEN FALSE
      WHEN et_hour < 10 THEN FALSE
      WHEN et_hour >= 17 THEN FALSE
      ELSE TRUE
    END AS is_prediction_window,
    -- Next prediction hour (predictions at :00 from 10-16)
    CASE
      WHEN et_dow IN (1, 7) THEN NULL  -- handled below
      WHEN et_hour < 10 THEN 10
      WHEN et_hour >= 16 THEN NULL     -- after last prediction
      ELSE et_hour + 1                 -- next full hour
    END AS next_pred_hour_today
  FROM params
),
next_trading_day AS (
  SELECT *,
    CASE
      WHEN et_dow = 7 THEN 2  -- Saturday → Monday (+2)
      WHEN et_dow = 1 THEN 1  -- Sunday → Monday (+1)
      WHEN et_dow = 6 THEN 3  -- Friday after hours → Monday (+3)
      WHEN next_pred_hour_today IS NULL THEN
        CASE WHEN et_dow = 6 THEN 3 ELSE 1 END  -- weekday after hours → next day (or Monday)
      ELSE 0
    END AS days_until_next
  FROM market_times
),
result AS (
  SELECT
    utc_now,
    et_datetime AS current_time_et,
    is_market_open,
    is_prediction_window,

    -- Status label
    CASE
      WHEN is_market_open AND is_prediction_window THEN 'MARKET OPEN'
      WHEN is_market_open AND NOT is_prediction_window THEN 'MARKET OPEN'
      ELSE 'MARKET CLOSED'
    END AS market_status,

    -- Predictions status
    CASE
      WHEN is_prediction_window AND et_hour >= 10 AND et_hour <= 16 THEN 'PREDICTIONS LIVE'
      ELSE 'PREDICTIONS PAUSED'
    END AS prediction_status,

    -- Status emoji/indicator
    CASE
      WHEN is_market_open THEN '🟢'
      ELSE '🔴'
    END AS status_indicator,

    -- Next prediction timestamp (ET)
    CASE
      WHEN days_until_next = 0 AND next_pred_hour_today IS NOT NULL THEN
        DATETIME(
          TIMESTAMP(CONCAT(CAST(et_date AS STRING), ' ', LPAD(CAST(next_pred_hour_today AS STRING), 2, '0'), ':00:00')),
          'America/New_York'
        )
      ELSE
        DATETIME(
          TIMESTAMP(CONCAT(CAST(DATE_ADD(et_date, INTERVAL days_until_next DAY) AS STRING), ' 10:00:00')),
          'America/New_York'
        )
    END AS next_prediction_et,

    -- Minutes until next prediction
    CASE
      WHEN days_until_next = 0 AND next_pred_hour_today IS NOT NULL THEN
        (next_pred_hour_today * 60) - (et_hour * 60 + et_minute)
      ELSE
        TIMESTAMP_DIFF(
          TIMESTAMP(CONCAT(CAST(DATE_ADD(et_date, INTERVAL days_until_next DAY) AS STRING), ' 10:00:00')),
          TIMESTAMP(CONCAT(CAST(et_date AS STRING), ' ',
            LPAD(CAST(et_hour AS STRING), 2, '0'), ':',
            LPAD(CAST(et_minute AS STRING), 2, '0'), ':00')),
          MINUTE
        )
    END AS minutes_until_next_prediction,

    -- Human-readable countdown
    CASE
      WHEN is_prediction_window AND et_hour >= 10 AND et_hour <= 16 THEN
        CONCAT('Next in ', CAST(60 - et_minute AS STRING), ' min')
      WHEN days_until_next = 0 AND next_pred_hour_today IS NOT NULL THEN
        CASE
          WHEN (next_pred_hour_today * 60) - (et_hour * 60 + et_minute) < 60 THEN
            CONCAT(CAST((next_pred_hour_today * 60) - (et_hour * 60 + et_minute) AS STRING), ' min')
          ELSE
            CONCAT(
              CAST(DIV((next_pred_hour_today * 60) - (et_hour * 60 + et_minute), 60) AS STRING), 'h ',
              CAST(MOD((next_pred_hour_today * 60) - (et_hour * 60 + et_minute), 60) AS STRING), 'm'
            )
        END
      WHEN days_until_next = 1 THEN 'Tomorrow 10:00 AM ET'
      WHEN days_until_next = 2 THEN 'Monday 10:00 AM ET'
      WHEN days_until_next = 3 THEN 'Monday 10:00 AM ET'
      ELSE CONCAT('In ', CAST(days_until_next AS STRING), ' days')
    END AS countdown_text,

    -- Last prediction time from actual data
    (SELECT MAX(TIMESTAMP(timestamp))
     FROM `trading-brains.tft_predictions.tft_predictions_logs`) AS last_prediction_time,

    -- Total predictions today
    (SELECT COUNT(*)
     FROM `trading-brains.tft_predictions.tft_predictions_logs`
     WHERE DATE(TIMESTAMP(timestamp)) = DATE(DATETIME(CURRENT_TIMESTAMP(), 'America/New_York'))
    ) AS predictions_today

  FROM next_trading_day
)
SELECT * FROM result
""")


print("\nAll views created successfully!")
print("Connect Looker Studio to these views as data sources:")
print("  1. vw_latest_predictions   → Scorecard / Table section")
print("  2. vw_prediction_history   → Prediction % change line chart")
print("  3. vw_actual_vs_predicted  → Actual vs Predicted line charts")
print("  4. vw_model_health         → Daily model health metrics")
print("  5. vw_prediction_detail    → Per-prediction drill-down")
print("  6. vw_market_status        → Market open/closed indicator & countdown")
