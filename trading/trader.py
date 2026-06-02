"""
TFT Paper Trading Service for Alpaca.

Multi-horizon trading strategy using TFT model predictions:
  - SWING trades (60m horizon): Higher confidence threshold, wider stops
  - SCALP trades (15m horizon): Lower threshold, tighter stops
  - BOOSTED_SWING: Both horizons agree strongly, larger position

Endpoints:
  /trade         – Main trading loop  (Cloud Scheduler, every 15 min)
  /check_exits   – Time-based & EOD exits + reconciliation (every 5 min)
  /status        – Current positions, account, recent trades
  /health        – Health check
  /force_close   – Emergency: close all positions & cancel orders
"""

import os
import json
import requests as http_requests
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from google.cloud import bigquery
import pytz

app = Flask(__name__)

# ── Configuration ───────────────────────────────────────────────────────────
TICKERS = os.environ.get("TICKERS", "GOOG,TSLA,AAPL").split(",")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "").strip()

BQ_PROJECT = "trading-brains"
BQ_DATASET = "tft_predictions"
BQ_TRADE_TABLE = "trade_log"

# Strategy thresholds (configurable via env vars for A/B testing)
SWING_ENTRY_THRESHOLD = float(os.environ.get("SWING_ENTRY_THRESHOLD", "8.0"))
SCALP_ENTRY_THRESHOLD = float(os.environ.get("SCALP_ENTRY_THRESHOLD", "5.0"))
MODERATE_THRESHOLD = float(os.environ.get("MODERATE_THRESHOLD", "3.0"))

# Risk management (dollar-based, configurable via env vars)
NOTIONAL_BOOSTED = float(os.environ.get("NOTIONAL_BOOSTED", "5000"))
NOTIONAL_SWING = float(os.environ.get("NOTIONAL_SWING", "3000"))
NOTIONAL_SCALP = float(os.environ.get("NOTIONAL_SCALP", "2000"))
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "0.03"))
MARKET_BUFFER_MINUTES = int(os.environ.get("MARKET_BUFFER_MINUTES", "15"))
PREDICTION_MAX_AGE_MINUTES = int(os.environ.get("PREDICTION_MAX_AGE_MINUTES", "20"))
EOD_CLOSE_MINUTES_BEFORE = 10     # Close positions 10 min before market close

ET = pytz.timezone("America/New_York")

# ── Alpaca REST helpers ─────────────────────────────────────────────────────

def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def alpaca_get(path):
    r = http_requests.get(
        f"{ALPACA_BASE_URL}/v2{path}", headers=_alpaca_headers(), timeout=10
    )
    r.raise_for_status()
    return r.json()


def alpaca_post(path, data):
    r = http_requests.post(
        f"{ALPACA_BASE_URL}/v2{path}", headers=_alpaca_headers(), json=data, timeout=10
    )
    r.raise_for_status()
    return r.json()


def alpaca_delete(path):
    r = http_requests.delete(
        f"{ALPACA_BASE_URL}/v2{path}", headers=_alpaca_headers(), timeout=10
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json() if r.content else {}


# ── Account & position helpers ──────────────────────────────────────────────

def get_account():
    return alpaca_get("/account")


def get_position(symbol):
    try:
        return alpaca_get(f"/positions/{symbol}")
    except http_requests.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def get_live_price(symbol):
    """Get latest quote from Alpaca market data."""
    r = http_requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
        headers=_alpaca_headers(),
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    quote = data.get("quote", {})
    # Use ask price; fall back to bid, then midpoint
    ap = quote.get("ap", 0)
    bp = quote.get("bp", 0)
    if ap > 0:
        return ap
    if bp > 0:
        return bp
    return None


def cancel_open_orders(symbol):
    """Cancel all open orders for a symbol."""
    try:
        orders = alpaca_get(f"/orders?status=open&symbols={symbol}")
        for order in orders:
            if order["status"] in ("new", "accepted", "pending_new", "partially_filled"):
                alpaca_delete(f"/orders/{order['id']}")
    except Exception as e:
        print(f"Error cancelling orders: {e}")


def close_position(symbol):
    """Close a position. Returns the closing order or None."""
    try:
        r = http_requests.delete(
            f"{ALPACA_BASE_URL}/v2/positions/{symbol}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error closing position: {e}")
        return None


# ── Market hours ────────────────────────────────────────────────────────────

def is_trading_window():
    """Return (ok, reason). Rejects weekends and the first/last buffer."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False, "Weekend"

    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    buf_open = open_t + timedelta(minutes=MARKET_BUFFER_MINUTES)
    buf_close = close_t - timedelta(minutes=MARKET_BUFFER_MINUTES)

    if now < buf_open:
        return False, f"Pre-market buffer (opens {buf_open.strftime('%H:%M')} ET)"
    if now > buf_close:
        return False, f"Post-market buffer (closed {buf_close.strftime('%H:%M')} ET)"
    return True, "Trading window open"


def is_eod_close_window():
    """Return True if we're within the EOD close window."""
    now = datetime.now(ET)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    eod_cutoff = close_t - timedelta(minutes=EOD_CLOSE_MINUTES_BEFORE)
    return now >= eod_cutoff and now < close_t


# ── BigQuery helpers ────────────────────────────────────────────────────────

def _bq_client():
    return bigquery.Client(project=BQ_PROJECT)


def _ensure_trade_log_table():
    """Create trade_log table if it doesn't exist."""
    client = _bq_client()
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TRADE_TABLE}"
    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("trade_id", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("side", "STRING"),
            bigquery.SchemaField("trade_type", "STRING"),
            bigquery.SchemaField("qty", "INT64"),
            bigquery.SchemaField("entry_price", "FLOAT64"),
            bigquery.SchemaField("stop_price", "FLOAT64"),
            bigquery.SchemaField("target_price", "FLOAT64"),
            bigquery.SchemaField("max_hold_minutes", "INT64"),
            bigquery.SchemaField("confidence_15m", "FLOAT64"),
            bigquery.SchemaField("confidence_60m", "FLOAT64"),
            bigquery.SchemaField("uncertainty_15m", "FLOAT64"),
            bigquery.SchemaField("uncertainty_60m", "FLOAT64"),
            bigquery.SchemaField("pred_15m", "FLOAT64"),
            bigquery.SchemaField("pred_60m", "FLOAT64"),
            bigquery.SchemaField("q10_used", "FLOAT64"),
            bigquery.SchemaField("q90_used", "FLOAT64"),
            bigquery.SchemaField("exit_price", "FLOAT64"),
            bigquery.SchemaField("exit_reason", "STRING"),
            bigquery.SchemaField("exit_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("pnl", "FLOAT64"),
            bigquery.SchemaField("pnl_pct", "FLOAT64"),
            bigquery.SchemaField("hold_minutes", "INT64"),
            bigquery.SchemaField("portfolio_value_at_entry", "FLOAT64"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f"Created table: {table_id}")


def get_latest_prediction(ticker):
    """Get most recent prediction with confidence data from BigQuery."""
    client = _bq_client()
    query = """
    SELECT *
    FROM `trading-brains.tft_predictions.tft_predictions_logs`
    WHERE ticker = @ticker
      AND confidence_60m IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 1
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ]
    )
    rows = list(client.query(query, job_config=config).result())
    return dict(rows[0]) if rows else None


def log_trade_entry(trade_data):
    """Insert a new trade entry into BigQuery via streaming insert."""
    client = _bq_client()
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TRADE_TABLE}"
    errors = client.insert_rows_json(table_id, [trade_data])
    if errors:
        print(f"BQ insert errors: {errors}")
    return not errors


def update_trade_exit(trade_id, exit_price, exit_reason, pnl, pnl_pct, hold_minutes):
    """Update trade log with exit information."""
    client = _bq_client()
    query = """
    UPDATE `trading-brains.tft_predictions.trade_log`
    SET exit_price = @exit_price,
        exit_reason = @exit_reason,
        exit_timestamp = CURRENT_TIMESTAMP(),
        pnl = @pnl,
        pnl_pct = @pnl_pct,
        hold_minutes = @hold_minutes
    WHERE trade_id = @trade_id
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("exit_price", "FLOAT64", exit_price),
            bigquery.ScalarQueryParameter("exit_reason", "STRING", exit_reason),
            bigquery.ScalarQueryParameter("pnl", "FLOAT64", pnl),
            bigquery.ScalarQueryParameter("pnl_pct", "FLOAT64", pnl_pct),
            bigquery.ScalarQueryParameter("hold_minutes", "INT64", hold_minutes),
            bigquery.ScalarQueryParameter("trade_id", "STRING", trade_id),
        ]
    )
    try:
        client.query(query, job_config=config).result()
    except Exception as e:
        # Streaming buffer may block UPDATEs for recently-inserted rows; log but don't crash
        print(f"WARNING: Could not update trade_log exit for {trade_id}: {e}")


def get_open_trade(ticker):
    """Get the most recent trade that hasn't been exited yet."""
    client = _bq_client()
    query = """
    SELECT trade_id, timestamp, trade_type, side, entry_price, qty,
           max_hold_minutes, stop_price, target_price
    FROM `trading-brains.tft_predictions.trade_log`
    WHERE ticker = @ticker
      AND exit_timestamp IS NULL
    ORDER BY timestamp DESC
    LIMIT 1
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ]
    )
    rows = list(client.query(query, job_config=config).result())
    return dict(rows[0]) if rows else None


def get_daily_pnl():
    """Sum of today's realized P&L."""
    client = _bq_client()
    today = datetime.now(ET).strftime("%Y-%m-%d")
    query = """
    SELECT COALESCE(SUM(pnl), 0) AS daily_pnl
    FROM `trading-brains.tft_predictions.trade_log`
    WHERE DATE(timestamp) = @today
      AND pnl IS NOT NULL
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("today", "STRING", today),
        ]
    )
    rows = list(client.query(query, job_config=config).result())
    return float(rows[0]["daily_pnl"]) if rows else 0.0


# ── Strategy Logic ──────────────────────────────────────────────────────────

def evaluate_signal(prediction):
    """
    Evaluate a prediction and produce a trade signal.

    Returns a dict:
        action:         "buy" | "sell" | None
        trade_type:     "BOOSTED_SWING" | "SWING" | "SCALP"
        confidence:     primary confidence value driving the trade
        stop_price:     suggested stop-loss price
        target_price:   suggested take-profit price
        max_hold_minutes: 15 or 60
        reason:         human-readable explanation
    """
    conf_60 = prediction.get("confidence_60m")
    conf_15 = prediction.get("confidence_15m")
    live = prediction.get("_live_price") or prediction.get("last_price")
    base_price = prediction.get("last_price")  # model's reference price
    pred_60 = prediction.get("pred_60m")
    pred_15 = prediction.get("pred_15m")
    q10_60 = prediction.get("q10_60m")
    q90_60 = prediction.get("q90_60m")
    q10_15 = prediction.get("q10_15m")
    q90_15 = prediction.get("q90_15m")

    if any(v is None for v in [conf_60, conf_15, live, base_price]):
        return {"action": None, "reason": "Missing prediction data"}

    conf_60 = float(conf_60)
    conf_15 = float(conf_15)
    live = float(live)
    base_price = float(base_price)

    def _rebase(model_price, fallback_pct):
        """Convert a model price to a live-price-relative price."""
        if model_price is None:
            return live * (1 + fallback_pct)
        pct = (float(model_price) - base_price) / base_price
        return live * (1 + pct)

    def _stop_long(q_low, horizon_label):
        """Compute stop for a long: must be below live price."""
        rebased = _rebase(q_low, -0.005 if "60" in horizon_label else -0.002)
        if rebased < live:
            return rebased
        # If model's q10 is above current price, use a percentage-based stop
        return live * (0.995 if "60" in horizon_label else 0.998)

    def _target_long(pred, horizon_label):
        """Compute target for a long: must be above live price."""
        rebased = _rebase(pred, 0.005 if "60" in horizon_label else 0.003)
        if rebased > live:
            return rebased
        return live * (1.005 if "60" in horizon_label else 1.003)

    def _stop_short(q_high, horizon_label):
        """Compute stop for a short: must be above live price."""
        rebased = _rebase(q_high, 0.005 if "60" in horizon_label else 0.002)
        if rebased > live:
            return rebased
        return live * (1.005 if "60" in horizon_label else 1.002)

    def _target_short(pred, horizon_label):
        """Compute target for a short: must be below live price."""
        rebased = _rebase(pred, -0.005 if "60" in horizon_label else -0.003)
        if rebased < live:
            return rebased
        return live * (0.995 if "60" in horizon_label else 0.997)

    last = live  # use live price for all comparisons

    # ── BOOSTED SWING: both horizons agree strongly ──────────────────────
    if conf_60 >= SWING_ENTRY_THRESHOLD and conf_15 >= SCALP_ENTRY_THRESHOLD:
        stop = _stop_long(q10_60, "60m")
        target = _target_long(pred_60, "60m")
        if stop >= last or target <= last:
            return {"action": None, "reason": "Invalid stop/target for boosted long"}
        return {
            "action": "buy", "trade_type": "BOOSTED_SWING",
            "confidence": conf_60, "stop_price": stop, "target_price": target,
            "max_hold_minutes": 60,
            "reason": f"Boosted swing long: c60={conf_60:.1f}, c15={conf_15:.1f}",
        }

    if conf_60 <= -SWING_ENTRY_THRESHOLD and conf_15 <= -SCALP_ENTRY_THRESHOLD:
        stop = _stop_short(q90_60, "60m")
        target = _target_short(pred_60, "60m")
        if stop <= last or target >= last:
            return {"action": None, "reason": "Invalid stop/target for boosted short"}
        return {
            "action": "sell", "trade_type": "BOOSTED_SWING",
            "confidence": conf_60, "stop_price": stop, "target_price": target,
            "max_hold_minutes": 60,
            "reason": f"Boosted swing short: c60={conf_60:.1f}, c15={conf_15:.1f}",
        }

    # ── SWING: 60m strong, 15m not opposing ──────────────────────────────
    if conf_60 >= SWING_ENTRY_THRESHOLD and conf_15 >= -MODERATE_THRESHOLD:
        stop = _stop_long(q10_60, "60m")
        target = _target_long(pred_60, "60m")
        if stop >= last or target <= last:
            return {"action": None, "reason": "Invalid stop/target for swing long"}
        return {
            "action": "buy", "trade_type": "SWING",
            "confidence": conf_60, "stop_price": stop, "target_price": target,
            "max_hold_minutes": 60,
            "reason": f"Swing long: c60={conf_60:.1f} (c15={conf_15:.1f} ok)",
        }

    if conf_60 <= -SWING_ENTRY_THRESHOLD and conf_15 <= MODERATE_THRESHOLD:
        stop = _stop_short(q90_60, "60m")
        target = _target_short(pred_60, "60m")
        if stop <= last or target >= last:
            return {"action": None, "reason": "Invalid stop/target for swing short"}
        return {
            "action": "sell", "trade_type": "SWING",
            "confidence": conf_60, "stop_price": stop, "target_price": target,
            "max_hold_minutes": 60,
            "reason": f"Swing short: c60={conf_60:.1f} (c15={conf_15:.1f} ok)",
        }

    # ── SCALP: 15m strong, 60m at least moderately agreeing ──────────────
    if conf_15 >= SCALP_ENTRY_THRESHOLD and conf_60 >= MODERATE_THRESHOLD:
        stop = _stop_long(q10_15, "15m")
        target = _target_long(pred_15, "15m")
        if stop >= last or target <= last:
            return {"action": None, "reason": "Invalid stop/target for scalp long"}
        return {
            "action": "buy", "trade_type": "SCALP",
            "confidence": conf_15, "stop_price": stop, "target_price": target,
            "max_hold_minutes": 15,
            "reason": f"Scalp long: c15={conf_15:.1f}, c60 agrees ({conf_60:.1f})",
        }

    if conf_15 <= -SCALP_ENTRY_THRESHOLD and conf_60 <= -MODERATE_THRESHOLD:
        stop = _stop_short(q90_15, "15m")
        target = _target_short(pred_15, "15m")
        if stop <= last or target >= last:
            return {"action": None, "reason": "Invalid stop/target for scalp short"}
        return {
            "action": "sell", "trade_type": "SCALP",
            "confidence": conf_15, "stop_price": stop, "target_price": target,
            "max_hold_minutes": 15,
            "reason": f"Scalp short: c15={conf_15:.1f}, c60 agrees ({conf_60:.1f})",
        }

    # ── No trade ─────────────────────────────────────────────────────────
    return {
        "action": None,
        "reason": f"No signal: c60={conf_60:.1f}, c15={conf_15:.1f}",
    }


def calculate_notional(signal):
    """
    Determine the dollar amount for this trade based on trade type.
    """
    trade_type = signal["trade_type"]
    if trade_type == "BOOSTED_SWING":
        return NOTIONAL_BOOSTED
    elif trade_type == "SWING":
        return NOTIONAL_SWING
    else:
        return NOTIONAL_SCALP


# ── Order execution ─────────────────────────────────────────────────────────

def place_bracket_order(symbol, side, qty, stop_price, target_price):
    """Place a market-entry bracket order using qty (shares).
    Falls back to a simple notional order if bracket fails (e.g. fractional-only accounts)."""
    # First try bracket order with qty
    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "stop_loss": {"stop_price": str(round(stop_price, 2))},
        "take_profit": {"limit_price": str(round(target_price, 2))},
    }
    try:
        return alpaca_post("/orders", order_data)
    except http_requests.HTTPError as e:
        if e.response.status_code in (403, 422):
            # Fallback: simple notional order (for fractional-only accounts)
            notional_amount = round(qty * get_live_price(symbol), 2)
            fallback_data = {
                "symbol": symbol,
                "notional": str(notional_amount),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }
            return alpaca_post("/orders", fallback_data)
        raise


# ── Helper: compute P&L ────────────────────────────────────────────────────

def _compute_pnl(side, entry_price, exit_price, qty):
    if side == "buy":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty
    pnl_pct = (pnl / (entry_price * qty)) * 100 if entry_price * qty else 0
    return round(pnl, 4), round(pnl_pct, 4)


# ── Flask endpoints ─────────────────────────────────────────────────────────

@app.route("/trade", methods=["GET", "POST"])
def trade():
    """Main trading loop – called every 15 min by Cloud Scheduler.
    Evaluates all tickers and places trades for each valid signal."""
    try:
        # 1. Market hours
        tradeable, reason = is_trading_window()
        if not tradeable:
            return jsonify({"action": "skip", "reason": reason}), 200

        # 2. Daily loss limit
        account = get_account()
        portfolio_value = float(account["portfolio_value"])
        buying_power = float(account["buying_power"])
        daily_pnl = get_daily_pnl()
        max_loss = portfolio_value * MAX_DAILY_LOSS_PCT
        if daily_pnl < -max_loss:
            return jsonify({
                "action": "skip",
                "reason": f"Daily loss limit: ${daily_pnl:.2f} (limit -${max_loss:.2f})",
            }), 200

        # 3. Evaluate each ticker
        results = []
        for ticker in TICKERS:
            result = _evaluate_and_trade(ticker, portfolio_value, buying_power)
            results.append(result)

        return jsonify({"results": results}), 200

    except Exception as e:
        print(f"Trade error: {e}")
        return jsonify({"error": str(e)}), 500


def _evaluate_and_trade(ticker, portfolio_value, buying_power):
    """Evaluate one ticker and place a trade if signal is strong."""
    # Skip if already holding this ticker
    position = get_position(ticker)
    if position:
        return {
            "ticker": ticker, "action": "hold",
            "reason": f"Already holding {position['qty']} shares",
        }

    # Get latest prediction
    prediction = get_latest_prediction(ticker)
    if not prediction:
        return {"ticker": ticker, "action": "skip", "reason": "No prediction data"}

    # Check freshness
    pred_ts = prediction.get("timestamp")
    if pred_ts:
        try:
            if isinstance(pred_ts, str):
                pred_dt = datetime.fromisoformat(pred_ts.replace("Z", "+00:00"))
            else:
                pred_dt = pred_ts if pred_ts.tzinfo else pred_ts.replace(tzinfo=pytz.UTC)
            age_min = (datetime.now(pytz.UTC) - pred_dt.astimezone(pytz.UTC)).total_seconds() / 60
            if age_min > PREDICTION_MAX_AGE_MINUTES:
                return {"ticker": ticker, "action": "skip", "reason": f"Stale prediction ({age_min:.0f} min old)"}
        except Exception as e:
            print(f"Timestamp parse warning for {ticker}: {e}")

    # Get live price
    live_price = get_live_price(ticker)
    if not live_price:
        return {"ticker": ticker, "action": "skip", "reason": "Could not get live price"}
    prediction["_live_price"] = live_price

    # Evaluate signal
    signal = evaluate_signal(prediction)
    if not signal.get("action"):
        return {"ticker": ticker, "action": "skip", "reason": signal.get("reason")}

    # Calculate dollar amount and qty (attempt bracket; fallback to notional)
    notional = calculate_notional(signal)
    qty = max(1, int(notional / live_price))
    order_cost = qty * live_price
    if order_cost > portfolio_value * 0.5:
        return {"ticker": ticker, "action": "skip", "reason": f"Order ${order_cost:.0f} exceeds 50% of portfolio ${portfolio_value:.0f}"}
    if order_cost > buying_power:
        return {"ticker": ticker, "action": "skip", "reason": f"Insufficient buying power ${buying_power:.0f} for order ${order_cost:.0f}"}

    # Place order (bracket if possible, fallback to notional)
    order = place_bracket_order(
        symbol=ticker,
        side=signal["action"],
        qty=qty,
        stop_price=signal["stop_price"],
        target_price=signal["target_price"],
    )

    # Use actual filled qty if available, else estimate from notional
    filled_qty = float(order.get("filled_qty") or order.get("qty") or 0)
    if filled_qty == 0:
        filled_qty = round(notional / live_price, 4)

    # Log trade entry
    trade_data = {
        "trade_id": order["id"],
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "ticker": ticker,
        "side": signal["action"],
        "trade_type": signal["trade_type"],
        "qty": filled_qty,
        "entry_price": live_price,
        "stop_price": round(signal["stop_price"], 2),
        "target_price": round(signal["target_price"], 2),
        "max_hold_minutes": signal["max_hold_minutes"],
        "confidence_15m": prediction.get("confidence_15m"),
        "confidence_60m": prediction.get("confidence_60m"),
        "uncertainty_15m": prediction.get("uncertainty_15m"),
        "uncertainty_60m": prediction.get("uncertainty_60m"),
        "pred_15m": prediction.get("pred_15m"),
        "pred_60m": prediction.get("pred_60m"),
        "q10_used": signal["stop_price"] if signal["action"] == "buy" else signal["target_price"],
        "q90_used": signal["target_price"] if signal["action"] == "buy" else signal["stop_price"],
        "portfolio_value_at_entry": portfolio_value,
    }
    log_trade_entry(trade_data)

    result = {
        "ticker": ticker,
        "action": signal["action"],
        "trade_type": signal["trade_type"],
        "notional": notional,
        "stop": round(signal["stop_price"], 2),
        "target": round(signal["target_price"], 2),
        "confidence_60m": prediction.get("confidence_60m"),
        "confidence_15m": prediction.get("confidence_15m"),
        "reason": signal["reason"],
        "order_id": order["id"],
    }
    print(f"TRADE PLACED: {json.dumps(result)}")
    return result


@app.route("/check_exits", methods=["GET", "POST"])
def check_exits():
    """
    Called every 5 min.  Handles exit scenarios for all tickers:
      1. EOD close – force-close before market close
      2. Timeout – hold time exceeded
      3. Reconcile – bracket leg (stop/TP) filled while we weren't looking
    """
    try:
        results = []
        for ticker in TICKERS:
            result = _check_exits_for_ticker(ticker)
            results.append(result)
        return jsonify({"results": results}), 200
    except Exception as e:
        print(f"check_exits error: {e}")
        return jsonify({"error": str(e)}), 500


def _check_exits_for_ticker(ticker):
    """Check exits for a single ticker."""
    position = get_position(ticker)

    # No position → check if a bracket leg filled (reconcile)
    if not position:
        trade = get_open_trade(ticker)
        if trade:
            return _reconcile_filled_trade(trade, ticker)
        return {"ticker": ticker, "action": "none", "reason": "No position, no open trade"}

    # EOD close
    if is_eod_close_window():
        return _close_with_reason(position, "eod_close", ticker)

    # Timeout close
    trade = get_open_trade(ticker)
    if trade:
        entry_ts = trade["timestamp"]
        if isinstance(entry_ts, str):
            entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
        else:
            entry_dt = entry_ts if entry_ts.tzinfo else entry_ts.replace(tzinfo=pytz.UTC)

        elapsed = (datetime.now(pytz.UTC) - entry_dt).total_seconds() / 60
        max_hold = trade.get("max_hold_minutes", 60)

        if elapsed >= max_hold:
            return _close_with_reason(position, "timeout", ticker, trade, int(elapsed))

        # Check stop-loss and take-profit (for non-bracket orders)
        current_price = float(position["current_price"])
        stop_price = float(trade.get("stop_price", 0))
        target_price = float(trade.get("target_price", 0))
        side = trade["side"]

        if side == "buy":
            if stop_price > 0 and current_price <= stop_price:
                return _close_with_reason(position, "stop_loss", ticker, trade, int(elapsed))
            if target_price > 0 and current_price >= target_price:
                return _close_with_reason(position, "take_profit", ticker, trade, int(elapsed))
        else:  # short
            if stop_price > 0 and current_price >= stop_price:
                return _close_with_reason(position, "stop_loss", ticker, trade, int(elapsed))
            if target_price > 0 and current_price <= target_price:
                return _close_with_reason(position, "take_profit", ticker, trade, int(elapsed))

        return {
            "ticker": ticker, "action": "hold",
            "elapsed_min": round(elapsed, 1), "max_hold": max_hold,
            "unrealized_pl": position.get("unrealized_pl"),
        }

    # Position exists but no trade log → orphaned, close it
    return _close_with_reason(position, "orphan_cleanup", ticker)


def _close_with_reason(position, reason, ticker, trade=None, elapsed_min=None):
    """Close position, cancel orders, update BQ."""
    cancel_open_orders(ticker)
    close_position(ticker)

    exit_price = float(position["current_price"])

    if trade:
        entry_price = float(trade["entry_price"])
        qty = float(trade["qty"])
        side = trade["side"]
        pnl, pnl_pct = _compute_pnl(side, entry_price, exit_price, qty)
        hold = elapsed_min or 0

        update_trade_exit(trade["trade_id"], exit_price, reason, pnl, pnl_pct, hold)

        return {
            "ticker": ticker, "action": "close", "reason": reason,
            "pnl": pnl, "pnl_pct": pnl_pct, "hold_minutes": hold,
        }

    return {"ticker": ticker, "action": "close", "reason": reason}


def _reconcile_filled_trade(trade, ticker):
    """Position is gone but trade_log has no exit → bracket leg filled."""
    try:
        orders = alpaca_get(f"/orders?status=closed&symbols={ticker}&limit=10")
    except Exception:
        orders = []

    exit_price = None
    exit_reason = "bracket_filled"

    for order in orders:
        if order.get("filled_avg_price") and order["id"] != trade["trade_id"]:
            if order.get("type") == "stop":
                exit_reason = "stop_loss"
            elif order.get("type") == "limit":
                exit_reason = "take_profit"
            exit_price = float(order["filled_avg_price"])
            break

    entry_price = float(trade["entry_price"])
    if exit_price is None:
        exit_price = entry_price

    qty = float(trade["qty"])
    pnl, pnl_pct = _compute_pnl(trade["side"], entry_price, exit_price, qty)

    entry_ts = trade["timestamp"]
    if isinstance(entry_ts, str):
        entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
    else:
        entry_dt = entry_ts if entry_ts.tzinfo else entry_ts.replace(tzinfo=pytz.UTC)
    hold = int((datetime.now(pytz.UTC) - entry_dt).total_seconds() / 60)

    update_trade_exit(trade["trade_id"], exit_price, exit_reason, pnl, pnl_pct, hold)

    return {
        "ticker": ticker, "action": "reconciled", "exit_reason": exit_reason,
        "pnl": pnl, "pnl_pct": pnl_pct,
    }


@app.route("/status", methods=["GET"])
def status():
    """Current account, positions, and recent trade history."""
    try:
        account = get_account()

        positions = {}
        for ticker in TICKERS:
            pos = get_position(ticker)
            if pos:
                positions[ticker] = {
                    "qty": pos["qty"],
                    "side": pos["side"],
                    "avg_entry": pos["avg_entry_price"],
                    "current_price": pos["current_price"],
                    "unrealized_pl": pos["unrealized_pl"],
                    "unrealized_plpc": pos["unrealized_plpc"],
                }

        client = _bq_client()
        query = """
        SELECT trade_id, timestamp, ticker, side, trade_type, qty,
               entry_price, exit_price, exit_reason, pnl, pnl_pct, hold_minutes
        FROM `trading-brains.tft_predictions.trade_log`
        ORDER BY timestamp DESC
        LIMIT 30
        """
        trades = []
        for row in client.query(query).result():
            t = dict(row)
            for k, v in t.items():
                if hasattr(v, "isoformat"):
                    t[k] = v.isoformat()
            trades.append(t)

        daily_pnl = get_daily_pnl()

        return jsonify({
            "account": {
                "portfolio_value": account.get("portfolio_value"),
                "cash": account.get("cash"),
                "buying_power": account.get("buying_power"),
                "equity": account.get("equity"),
            },
            "positions": positions,
            "daily_pnl": daily_pnl,
            "recent_trades": trades,
            "tickers": TICKERS,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "tickers": TICKERS,
        "config": {
            "swing_threshold": SWING_ENTRY_THRESHOLD,
            "scalp_threshold": SCALP_ENTRY_THRESHOLD,
            "moderate_threshold": MODERATE_THRESHOLD,
            "notional_boosted": NOTIONAL_BOOSTED,
            "notional_swing": NOTIONAL_SWING,
            "notional_scalp": NOTIONAL_SCALP,
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        },
    }), 200


@app.route("/force_close", methods=["POST"])
def force_close():
    """Emergency: cancel all orders and close all positions."""
    try:
        results = []
        for ticker in TICKERS:
            cancel_open_orders(ticker)
            result = close_position(ticker)

            trade = get_open_trade(ticker)
            if trade:
                position_data = get_position(ticker)
                exit_price = float(position_data["current_price"]) if position_data else float(trade["entry_price"])
                pnl, pnl_pct = _compute_pnl(
                    trade["side"], float(trade["entry_price"]), exit_price, int(trade["qty"])
                )
                update_trade_exit(trade["trade_id"], exit_price, "force_close", pnl, pnl_pct, 0)
            results.append({"ticker": ticker, "closed": result is not None})

        return jsonify({"action": "force_close", "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Startup ─────────────────────────────────────────────────────────────────

_ensure_trade_log_table()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
