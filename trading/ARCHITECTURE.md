# TFT Paper Trading System — Architecture

## 0. Abstract

An automated paper-trading system that uses a Temporal Fusion Transformer (TFT)
model to trade **GOOG, TSLA, and AAPL** simultaneously on [Alpaca](https://alpaca.markets)
paper accounts.  The model produces multi-horizon price predictions with
quantile-based confidence and uncertainty measures.  A dedicated Cloud Run
service evaluates these predictions for each ticker independently, decides
entry/exit, and places bracket orders through the Alpaca REST API.  Multiple
positions can be held concurrently across tickers.

The system runs an **A/B test** with two separate $1,000 paper accounts:
- **paper_A** (aggressive): Low confidence thresholds, trades frequently
- **paper_B** (cautious): Moderate thresholds, trades selectively

All strategy parameters (thresholds, notional amounts, risk limits) are
configurable via environment variables, allowing the same container image
to run with different risk profiles.


What this system does:

Every 15 minutes during US market hours, our TFT model predicts where each ticker's price will be in 15 and 60 minutes — along with how confident it is and how wide the range of outcomes could be. The trading service reads these predictions for GOOG, TSLA, and AAPL independently and decides whether to act on each.

Example — Bullish swing trade (GOOG):
At 10:45 AM, GOOG trades at $172.00. The model predicts $173.40 in 60 minutes (confidence: 11.2, uncertainty: 1.4) and $172.60 in 15 minutes (confidence: 6.1). Both horizons agree the price is going up, so the system places a BOOSTED_SWING long: it buys $5,000 notional of GOOG with a stop-loss at $171.15 (the model's 10th-percentile estimate) and a take-profit at $173.40.

Example — Concurrent positions:
At 11:00 AM, the system holds a GOOG long from 10:45 AND opens a new TSLA short because TSLA's 15m confidence is −7.3. Each ticker is evaluated independently — positions in one ticker don't block trades in another.

Example — No trade:
At 11:33 AM, AAPL's 60-minute model is bullish (confidence: 9.5) but the 15-minute model is bearish (confidence: −6.2). The horizons contradict each other, so the system sits out on AAPL — but may still trade GOOG or TSLA if their signals are valid.

In short: the system trades any ticker where the model is both confident and internally consistent, uses fixed dollar amounts per trade type, and always has a pre-defined exit plan before entering.


## 1. System Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                     Cloud Scheduler                              │
│  ┌──────────────────┐  ┌──────────────────┐                      │
│  │ predict-*        │  │ predict-actuals  │  (*/15 10-16 M-F)    │
│  │ */15 10-16 M-F   │  │ 15-75/15 10-16   │                     │
│  └────────┬─────────┘  └──────────────────┘                      │
│           │                                                      │
│  ┌────────▼──────────────────────────────────────┐                │
│  │ tft-trade-a / tft-trade-b (+ check-exits)    │                │
│  │ 3-58/15 10-16        */5 10-16                │                │
│  └────────┬──────────────────────┬───────────────┘                │
└───────────┼──────────────────────┼────────────────────────────────┘
            │                      │
            ▼                      ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Cloud Run: tft-trader-a     │  │  Cloud Run: tft-trader-b     │
│  (aggressive, $1K account)   │  │  (cautious, $1K account)     │
│  Thresholds: 4.0/2.5/1.5    │  │  Thresholds: 6.0/4.0/2.5    │
│  Notional: $200/$150/$100    │  │  Notional: $150/$100/$75     │
└────────────┬─────────────────┘  └────────────┬─────────────────┘
             │                                  │
     ┌───────▼───────┐                  ┌───────▼───────┐
     │   BigQuery    │                  │   Alpaca      │
     │ tft_predictions│                  │ Paper API     │
     │               │                  │ (3 accounts)  │
     │ • predictions │                  │ • paper_A     │
     │   _logs       │                  │ • paper_B     │
     │ • trade_log   │                  │ • account 1   │
     └───────────────┘                  └───────────────┘
```

## 2. Strategy

### 2.1 Multi-Horizon Approach

The strategy uses **two time horizons** from the TFT model, each serving a
different purpose:

| Horizon | Role | Hold Time | When Triggered (default) |
|---------|------|-----------|-------------------------|
| **60 min** | Swing direction | Up to 60 min | `\|confidence_60m\| ≥ SWING_ENTRY_THRESHOLD` |
| **15 min** | Scalp timing | Up to 15 min | `\|confidence_15m\| ≥ SCALP_ENTRY_THRESHOLD` |

All thresholds and notional amounts are **configurable via environment variables**
per service instance (see Section 6.2 for the A/B profiles).

### 2.2 Trade Types

Three trade types, selected by signal evaluation:

#### BOOSTED_SWING (both horizons agree strongly)
- **Entry**: `|confidence_60m| ≥ SWING_ENTRY_THRESHOLD` AND `|confidence_15m| ≥ SCALP_ENTRY_THRESHOLD`
- **Stops**: q10/q90 from 60m quantiles (rebased to live price)
- **Target**: pred_60m (median prediction)
- **Hold**: max 60 minutes
- **Notional**: `NOTIONAL_BOOSTED`
- **Rationale**: Highest conviction — both tactical and strategic signals align

#### SWING (60m strong, 15m not opposing)
- **Entry**: `|confidence_60m| ≥ SWING_ENTRY_THRESHOLD` AND `|confidence_15m| > -MODERATE_THRESHOLD`
- **Stops**: q10/q90 from 60m quantiles (rebased to live price)
- **Target**: pred_60m
- **Hold**: max 60 minutes
- **Notional**: `NOTIONAL_SWING`
- **Rationale**: Strong strategic signal; short-term at worst neutral

#### SCALP (15m strong, 60m moderately agreeing)
- **Entry**: `|confidence_15m| ≥ SCALP_ENTRY_THRESHOLD` AND `|confidence_60m| ≥ MODERATE_THRESHOLD`
- **Stops**: q10/q90 from 15m quantiles (rebased to live price)
- **Target**: pred_15m
- **Hold**: max 15 minutes
- **Notional**: `NOTIONAL_SCALP`
- **Rationale**: Quick tactical trade with strategic backdrop support

### 2.3 Signal Evaluation Priority

```
1. BOOSTED_SWING  (60m strong + 15m strong + same direction)
2. SWING          (60m strong + 15m not opposing)
3. SCALP          (15m strong + 60m moderately agreeing)
4. NO TRADE       (insufficient conviction on either horizon)
```

If the 15m and 60m horizons point in **opposite** directions, no trade is
taken — the model is conflicted.

### 2.4 Direction: Long + Short

Both long and short trades are supported:

| Signal | Action | Stop-Loss | Take-Profit |
|--------|--------|-----------|-------------|
| Positive confidence | **Buy** (long) | q10 (10th percentile, below entry) | pred (median, above entry) |
| Negative confidence | **Sell** (short) | q90 (90th percentile, above entry) | pred (median, below entry) |

Short positions are opened by placing a `sell` order when no position is held.
Alpaca's paper account handles short selling natively.

## 3. Order Execution

### 3.1 Order Type: Market-Entry Bracket Orders

**Decision: Market orders** for entries, combined with bracket order legs
for stop-loss and take-profit.

**Reasoning:**
- **Fill guarantee** — a missed trade when the model has high conviction costs
  more than a few cents of slippage
- **GOOG spread** is typically $0.01–$0.03; slippage is negligible vs the
  expected move ($2–$10 on 60m, $0.50–$2 on 15m)
- **Bracket orders** delegate stop/take-profit management to Alpaca — no
  need for constant monitoring
- **Simplicity** — fewer failure modes than limit entry + manual stop tracking

### 3.2 Bracket Order Structure

```
┌─────────────────────────────────────────┐
│           Bracket Order                  │
│                                          │
│  Primary leg:  MARKET BUY/SELL (entry)   │
│                                          │
│  OCO legs (auto-created by Alpaca):      │
│    ├── STOP order (stop-loss)            │
│    └── LIMIT order (take-profit)         │
│                                          │
│  When one OCO leg fills, the other       │
│  is automatically cancelled.             │
└─────────────────────────────────────────┘
```

**Example long trade (TSLA, BOOSTED_SWING, paper_A):**
```
Entry:       MARKET BUY 1 share TSLA  (int($200 / $180) = 1)
Stop-loss:   STOP SELL at $178.50  (q10_60m rebased to live price)
Take-profit: LIMIT SELL at $182.20 (pred_60m)
Time-in-force: day
```

## 4. Position Sizing

Uses **fixed-notional sizing** — each trade type has a configurable dollar
amount that is converted to shares at order time: `qty = notional / live_price`.

```
qty = int(NOTIONAL_X / live_price)
```

### 4.1 Notional Amounts (per-service, env-var configurable)

| Trade Type | paper_A (aggressive) | paper_B (cautious) |
|------------|---------------------|--------------------|
| BOOSTED_SWING | $200 | $150 |
| SWING | $150 | $100 |
| SCALP | $100 | $75 |

### 4.2 Safety Rails

- **Daily loss limit**: configurable (`MAX_DAILY_LOSS_PCT`), 5% (A) / 3% (B)
- **Time in force**: `day` — all orders expire at market close
- **No cooldown**: Trades can follow each other immediately
- **Multi-position**: Each ticker can hold one position; up to 3 concurrent
- **Market buffer**: configurable, 5 min (A) / 10 min (B) after open/before close

### 4.3 Example (paper_A)

```
Portfolio: $1,000
Signal: BOOSTED_SWING long TSLA at $180.00

Order: MARKET BUY 1 share TSLA (int($200 / $180) = 1)
  → Stop at $178.50 (q10_60m rebased to live price)
  → Target at $182.20 (pred_60m)
  → Max hold: 60 minutes
  → Time in force: day
```

## 5. Risk Management

### 5.1 Safety Rails

| Rule | paper_A | paper_B | Purpose |
|------|---------|---------|--------|
| **Max 1 position per ticker** | GOOG, TSLA, AAPL | Same | Allow concurrent positions |
| **Daily loss limit** | −5% | −3% | Stop trading after bad day |
| **Market buffer** | 5 min (9:35–3:55 ET) | 10 min (9:40–3:50 ET) | Skip volatile open/close |
| **EOD close** | 3:50 PM ET | Same | No overnight risk |
| **No cooldown** | Yes | Yes | Maximize opportunity capture |
| **Prediction freshness** | < 25 min | < 20 min | Don't trade on stale signals |
| **Stop/target validation** | Rebased to live price; fallback ±0.3%/±0.8% | Same | Handle model–price divergence |

### 5.2 Exit Hierarchy

Exits are evaluated in priority order:

```
1. STOP-LOSS        ← Alpaca bracket leg (automatic)
2. TAKE-PROFIT      ← Alpaca bracket leg (automatic)
3. EOD CLOSE        ← /check_exits at 3:50 PM ET
4. TIMEOUT          ← /check_exits after max_hold_minutes elapsed
5. FORCE CLOSE      ← /force_close (manual emergency)
```

### 5.3 Reconciliation

When a bracket leg fills (stop or take-profit), Alpaca closes the position
automatically.  The `/check_exits` endpoint runs every 5 minutes and detects
these fills:

1. Checks if the Alpaca position still exists
2. If position is gone but trade_log has no exit → bracket leg filled
3. Queries Alpaca's closed orders to determine which leg (stop or TP)
4. Updates BigQuery trade_log with exit price, reason, P&L

## 6. Infrastructure

### 6.1 GCP Resources

| Resource | Name | Config |
|----------|------|--------|
| Cloud Run | `tft-trader-a` | 512 Mi, 1 CPU, max 1, aggressive profile (paper_A) |
| Cloud Run | `tft-trader-b` | 512 Mi, 1 CPU, max 1, cautious profile (paper_B) |
| Cloud Run | `tft-trader` | Original service (account 1, $100K) |
| Cloud Scheduler | `tft-trade-a` / `tft-check-exits-a` | paper_A triggers |
| Cloud Scheduler | `tft-trade-b` / `tft-check-exits-b` | paper_B triggers |
| Secret Manager | `alpaca-api-key-a` / `alpaca-secret-key-a` | paper_A credentials |
| Secret Manager | `alpaca-api-key-b` / `alpaca-secret-key-b` | paper_B credentials |
| BigQuery | `tft_predictions.trade_log` | Trade entry + exit log (shared, all accounts) |
| Container | `gcr.io/trading-brains/tft-trader` | Python 3.11 + Flask + gunicorn |

### 6.2 A/B Test Configuration

All parameters are set via environment variables per service:

| Env Var | paper_A (aggressive) | paper_B (cautious) | Default |
|---------|---------------------|--------------------|---------|
| `TICKERS` | GOOG,TSLA,AAPL | GOOG,TSLA,AAPL | GOOG,TSLA,AAPL |
| `SWING_ENTRY_THRESHOLD` | 4.0 | 6.0 | 8.0 |
| `SCALP_ENTRY_THRESHOLD` | 2.5 | 4.0 | 5.0 |
| `MODERATE_THRESHOLD` | 1.5 | 2.5 | 3.0 |
| `NOTIONAL_BOOSTED` | 200 | 150 | 5000 |
| `NOTIONAL_SWING` | 150 | 100 | 3000 |
| `NOTIONAL_SCALP` | 100 | 75 | 2000 |
| `MAX_DAILY_LOSS_PCT` | 0.05 | 0.03 | 0.03 |
| `MARKET_BUFFER_MINUTES` | 5 | 10 | 15 |
| `PREDICTION_MAX_AGE_MINUTES` | 25 | 20 | 20 |

### 6.2 Data Flow

```
Every 15 min:
  Cloud Scheduler → tft-predictions/predict (GOOG, TSLA, AAPL)
     → writes predictions to BigQuery tft_predictions_logs

3 min later:
  Cloud Scheduler → tft-trader/trade
     → for each ticker in TICKERS:
        → reads latest prediction from BigQuery
        → evaluates signal (BOOSTED_SWING / SWING / SCALP / no trade)
        → if signal valid & no existing position for this ticker:
           → places notional bracket order on Alpaca
           → logs entry to BigQuery trade_log

Every 5 min:
  Cloud Scheduler → tft-trader/check_exits
     → for each ticker in TICKERS:
        → checks if position still exists
        → if gone: reconcile bracket fill → update trade_log
        → if held past max_hold_minutes: close + update trade_log
        → if 3:50 PM ET: EOD close + update trade_log
```

### 6.3 Authentication

- Cloud Scheduler → Cloud Run: **OIDC token** with the default compute
  service account (`137489665103-compute@developer.gserviceaccount.com`)
- Cloud Run → Alpaca: **API key + secret** from Secret Manager (env vars)
- Cloud Run → BigQuery: **Default service account** (implicit credentials)

### 6.4 Endpoints

| Endpoint | Method | Trigger | Purpose |
|----------|--------|---------|---------|
| `/trade` | POST | Scheduler (15 min) | Main decision loop |
| `/check_exits` | POST | Scheduler (5 min) | Exits + reconciliation |
| `/status` | GET | Manual | Account + position + trade history |
| `/health` | GET | Manual/monitoring | Liveness check |
| `/force_close` | POST | Manual | Emergency position close |

## 7. BigQuery Schema: `trade_log`

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | STRING | Alpaca order ID (primary key) |
| `timestamp` | TIMESTAMP | Entry time |
| `ticker` | STRING | GOOG / TSLA / AAPL |
| `side` | STRING | buy / sell |
| `trade_type` | STRING | BOOSTED_SWING / SWING / SCALP |
| `qty` | INT64 | Number of shares |
| `entry_price` | FLOAT64 | Price at time of order placement |
| `stop_price` | FLOAT64 | Stop-loss price |
| `target_price` | FLOAT64 | Take-profit price |
| `max_hold_minutes` | INT64 | Planned max hold (15 or 60) |
| `confidence_15m` | FLOAT64 | Model confidence at entry |
| `confidence_60m` | FLOAT64 | Model confidence at entry |
| `uncertainty_15m` | FLOAT64 | Model uncertainty at entry |
| `uncertainty_60m` | FLOAT64 | Model uncertainty at entry |
| `pred_15m` | FLOAT64 | Predicted 15m price |
| `pred_60m` | FLOAT64 | Predicted 60m price |
| `q10_used` | FLOAT64 | Quantile used for stop (long) or target (short) |
| `q90_used` | FLOAT64 | Quantile used for target (long) or stop (short) |
| `exit_price` | FLOAT64 | Actual exit fill price |
| `exit_reason` | STRING | stop_loss / take_profit / timeout / eod_close / force_close |
| `exit_timestamp` | TIMESTAMP | When position was closed |
| `pnl` | FLOAT64 | Realized profit/loss in $ |
| `pnl_pct` | FLOAT64 | Realized P&L as % of entry |
| `hold_minutes` | INT64 | Actual hold duration |
| `portfolio_value_at_entry` | FLOAT64 | Portfolio value when trade was placed |

## 8. Monitoring & Analysis

### 8.1 Quick Commands

Check service status:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://tft-trader-etpdldaita-oa.a.run.app/status
```

Force close all positions:
```bash
curl -X POST -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://tft-trader-etpdldaita-oa.a.run.app/force_close
```

### 8.2 BigQuery Queries

**Daily P&L summary:**
```sql
SELECT
  DATE(timestamp) AS trade_date,
  COUNT(*) AS trades,
  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) AS losses,
  ROUND(SUM(pnl), 2) AS total_pnl,
  ROUND(AVG(pnl_pct), 4) AS avg_pnl_pct,
  ROUND(AVG(hold_minutes), 1) AS avg_hold_min
FROM `trading-brains.tft_predictions.trade_log`
WHERE pnl IS NOT NULL
GROUP BY trade_date
ORDER BY trade_date DESC;
```

**Performance by trade type:**
```sql
SELECT
  trade_type,
  COUNT(*) AS trades,
  ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS win_rate_pct,
  ROUND(AVG(pnl), 2) AS avg_pnl,
  ROUND(SUM(pnl), 2) AS total_pnl,
  ROUND(AVG(hold_minutes), 1) AS avg_hold_min
FROM `trading-brains.tft_predictions.trade_log`
WHERE pnl IS NOT NULL
GROUP BY trade_type;
```

**Exit reason analysis:**
```sql
SELECT
  exit_reason,
  COUNT(*) AS count,
  ROUND(AVG(pnl), 2) AS avg_pnl,
  ROUND(SUM(pnl), 2) AS total_pnl
FROM `trading-brains.tft_predictions.trade_log`
WHERE exit_reason IS NOT NULL
GROUP BY exit_reason;
```

## 9. Extending the System

### 9.1 Adding More Tickers

1. Add the ticker to the `TICKERS` env var (comma-separated):
   ```bash
   # Use env-vars-file to handle commas:
   # env.yaml: TICKERS: 'GOOG,TSLA,AAPL,SPY'
   gcloud run deploy tft-trader \
     --image gcr.io/trading-brains/tft-trader \
     --env-vars-file env.yaml
   ```
2. Ensure the prediction service produces predictions for the new ticker
3. No scheduler changes needed — the service handles all tickers per call

### 9.2 Tuning Thresholds

All parameters are configurable via environment variables — no code changes
or container rebuilds needed. Deploy with `--env-vars-file`:

```yaml
# env-aggressive.yaml
TICKERS: "GOOG,TSLA,AAPL"
SWING_ENTRY_THRESHOLD: "4.0"
SCALP_ENTRY_THRESHOLD: "2.5"
MODERATE_THRESHOLD: "1.5"
NOTIONAL_BOOSTED: "200"
NOTIONAL_SWING: "150"
NOTIONAL_SCALP: "100"
MAX_DAILY_LOSS_PCT: "0.05"
MARKET_BUFFER_MINUTES: "5"
PREDICTION_MAX_AGE_MINUTES: "25"
```

| Parameter | Effect of Decrease | Effect of Increase |
|-----------|-------------------|-------------------|
| `SWING_ENTRY_THRESHOLD` | More swing trades (lower conviction) | Fewer but higher-quality swings |
| `SCALP_ENTRY_THRESHOLD` | More scalp trades | Fewer scalps |
| `MODERATE_THRESHOLD` | Easier horizon agreement | Stricter agreement required |
| `NOTIONAL_*` | Smaller positions | Larger positions |
| `MAX_DAILY_LOSS_PCT` | Stops earlier on bad days | More tolerance |
| `MARKET_BUFFER_MINUTES` | Trades closer to open/close | Avoids volatile periods |

### 9.3 Moving to Live Trading

1. Create a live Alpaca account
2. Store live API keys as new secrets
3. Change `ALPACA_BASE_URL` to `https://api.alpaca.markets`
4. **Reduce position sizes** initially (halve `MAX_RISK_PCT`)
5. **Add slippage monitoring** — compare expected vs actual fills
6. Consider **limit orders** for entry to control slippage

## 10. File Structure

```
trading_brains/
├── trading/
│   ├── trader.py          ← Flask app (strategy + execution + logging)
│   ├── requirements.txt   ← Python dependencies
│   └── Dockerfile         ← Container image
├── pipeline/
│   ├── train.py           ← Model training (Vertex AI)
│   ├── features.py        ← Feature engineering
│   └── Dockerfile         ← Training container
├── main.py                ← Prediction service (Cloud Run: tft-predictions)
├── build_and_submit.py    ← Training job submission
└── ARCHITECTURE.md        ← This document
```
