# Trading Research — Shared Brief

*Last updated: 23 Sep 2026. Covers the BTC microstructure dataset, the prediction experiments and the cluster work.*
*Scope note: this version omits work belonging to other projects.*

---

## 1. Who does what

- **Matthias** — hypotheses (which setups matter and why), ATAS testing, GCP and datasets.
- **Daniel** — development; makes setups *measurable* (event extraction, execution-faithful checks).
- **Dominik** — ML; built the Temporal Fusion Transformer; makes setups *rankable*.
- **Michael** — discretionary trader (Market Monkey / MMT Discord); source of setup ideas.

**Working principle:** hypothesis → measurable event table → small model. Never hand a model raw market data and hope it finds a strategy.

---

## 2. Infrastructure

**GCP project `trading-brains`** — two datasets in different regions (they cannot be joined in a single SQL statement):

| Dataset | Region | Contents |
|---|---|---|
| `market_microstructure` | US | One table: `mmt_btc_1m_v3` (§3) |
| `tft_predictions` | europe-west6 | Live TFT system: prediction logs, `tft_actuals`, trade logs, `volatility_context`, ~15 views. Still running. |

- **No stock candles exist in BigQuery.** `tft_actuals` only stores prices at +15/30/45/60 min after each prediction.
- **MMT API subscription: cancelled** — `mmt_btc_1m_v3` cannot be extended past 2026-07-23.
- **Data sources:** Polygon is now **massive.com** (existing keys still work; base URL `api.massive.com`). Its **free plan returns minute bars** — used for the SPY pull. Binance publishes free historical trade data for spot, USD-M and COIN-M futures at `data.binance.vision`, no account needed; that covers spot / linear / inverse in one source.

---

## 3. `mmt_btc_1m_v3` — the research dataset (audited & cleaned 21 Sep 2026)

- BTC, 1-minute, **2025-07-26 19:39 → 2026-07-23 19:39 UTC**, final.
- Two slices per minute via `exchange`: `binancef` (price, candles, funding, mark, OI) and `binancef:bybitf` (aggregate order flow). **Always filter or group by `exchange`.**
- 521,245 + 521,277 rows; one row per minute per slice; partitioned by `DATE(ts)`.

**Cleanup performed:** removed 6 duplicate minutes per slice (the backfill ran in 60-day chunks with inclusive endpoints); replaced fabricated zeros with NULL — `funding_rate` and `mark_price` on the aggregate slice, and `last_price`, `buy_vol`, `sell_vol`, `agg_delta`, `tps_*` before 2026-01-10, plus `tps_*` on 11,508 later rows.

**Regime boundary: 2026-01-10 00:00 UTC** — the upstream data source changed that night.

| Feature | binancef | binancef:bybitf |
|---|---|---|
| `vd_b1`…`vd_b11` (signed delta per trade-size bucket; sum = `candle_delta`) | full year | full year |
| trade counts | full year | full year |
| `skew_*` | full year | from 2026-01-10 |
| depth, volume, `agg_delta`, `last_price`, `tps_*` | from 2026-01-10 | from 2026-01-10 |
| `oi_close` | full year (7 NULLs) | full year |
| `mark_price`, `funding_rate` | full year | never |
| OHLC, `candle_*` | full year | never |

**Gaps:** 36 missing minutes on `binancef` (including a 19-minute outage on 2025-08-29 06:17), 4 on the aggregate slice. 58 later `binancef` minutes show zero trades and zero volume — probable collector blanks.

**ATAS exports run on fixed UTC+1 all year** (no daylight saving) — subtract 60 minutes to align with v3.

**Rule for any future loader:** half-open time windows; NULL for missing values, never 0.0; post-load audit for duplicates, gaps and constant columns.

---

## 4. Findings

### Standing
- **Cluster second-interaction — a directional read, not a strategy** (BTC 100-tick reversal clusters, 633 clusters, Jul 2025 – Jul 2026):
  - Never fade a *forming* cluster: 38%.
  - **Entry 1, retest of a proven hold: 73% raw → 67% under conservative intrabar handling** (n=224). Stable across both halves of the year.
  - Entry 2, reclaim after a failed break: 63% raw → 62% (n=373).
  - Cluster anatomy (bar count, width, stretch) does not predict outcome.
- **Volatility gate** (bar production rate used as a filter) — holds on two instruments.

### Falsified — do not retry
- **TFT / dense minute prediction on BTC:** close-to-close, magnitude, direction and barrier labels all failed. The early 85% and 57.3% results were artefacts of a single 80/20 split validating on the most favourable period; walk-forward removed the edge, and MMT order-flow features added nothing.
- **SPY control (23 Sep 2026) — the failure is not crypto-specific.** The identical experiment with only the asset changed (§5). All 12 configurations fail. BTC edges sit near zero (−0.64 to +0.19pp); SPY is consistently negative (−3.7 to −4.8pp, both seeds, 0–1 of 5 folds positive). SPY's smaller EV loss (−0.01% vs −0.07%) reflects lower trading costs, not skill. Caveats: SPY's effective sample is only 79–244, and a persistently negative edge is more likely drift (training-period majority vs validation-period majority) than genuine anti-prediction.
- **Cluster entries are not tradeable on BTC after costs (23 Sep 2026).** Entry 1 pays $200 against ~$300 of risk *measured from the actual entry* (the target is defined from the zone midpoint, not the entry), so breakeven is 60% against 67% actual — about +$33/trade gross, while fees run ~$36 on wins and ~$73 on stop-outs. Entry 2 needs 70% and delivers 62%, so it is negative by construction. Eleven exit styles (fixed targets 200–900, partial-target-plus-breakeven, three trailing variants) were fitted on the first half of the year and judged on the second: **none positive in both halves.** Partial targets and trailing stops made it worse.
- **Order flow does not rank cluster entries (23 Sep 2026).** Walk-forward logistic model over 9 pre-registered full-year features (bucket flow, composition, trade intensity, OI change, funding, book skew, wait time, entry type, direction), 592 entries but only **289 independent episodes**. Out-of-sample AUC 0.49; the top-rated half beats the rest by 2.5pp with p = 0.18 against a circular-shift null; it helped in the second half and hurt in the first; on retests specifically, 63% kept versus 65% skipped. This rules out an improvement of 5pp or more; it cannot rule out 2–3pp, which would not change the economics. *Faint thread:* the two bucket-flow features were the only ones pointing the same way in both halves (AUC ≈ 0.56, in-sample, found by looking) — a hypothesis, not a filter, and no fresh MMT data exists to test it against.

### Working principles
- Per-trade margin is the only armour against friction.
- Sparse, conditional setups survive; dense minute-by-minute prediction has now failed on both BTC and SPY.
- Methodology: walk-forward validation, seed repetition, best-constant baselines, thresholds fixed before looking, first-half/second-half splits, worst-case intrabar assumptions, and vigilance about zeros that should be NULLs.
- Measure P&L from the **actual entry price**, never from a reference level.

---

## 5. Code and data assets

- **`spy_vs_btc_control.ipynb`** — one notebook, `ASSET = "btc"|"spy"`, identical code both ways. Rebuilt 23 Sep 2026 after the original percentage-barrier cells were lost (never saved, never committed). Includes the **fold-alignment fix** (all evaluation data read from the validation frame), ATR-scaled barriers (k = 4.5 / 7.5 / 15, equivalent to BTC's earlier 0.3 / 0.5 / 1.0% barriers), session-bounded SPY trades, and timeouts exiting at market. Reproduction check passes: edge −1.23pp against the −0.89pp reference, EV −0.082% against −0.080%, with identical resolution rate, median bars and effective sample. Both assets ran with `BATCH_SIZE=256`, `16-mixed`.
- **`pull_spy_1m.py`** — Massive free plan; two years of SPY 1-minute bars → `spy_1m.csv` (427,915 bars, 2024-09-23 → 2026-09-18, ~388 regular-session bars per day).
- **Cluster event rebuild** — `100ticksClusterBTC-1year.csv` (195,894 bars) → 633 clusters; detection is ≥15 bars within a $220 envelope; entries and outcomes reproduce exactly.

---

## 6. Where things stand

**Both original questions now have answers.** The SPY control is closed: no edge, and the failure is not specific to crypto. Cluster-plus-model is closed: order flow does not rank the entries, and the setup is not tradeable standalone after costs.

Open threads, roughly in order of promise:
1. **Cluster zones as *location* for a setup with a fatter per-trade margin** — the untested combination. Needs tape data rather than v3.
2. **Speed of Waves / multi-bucket aggression** (Michael's indicator). Spot, linear and inverse bursts firing together. In the example shared on 23 Sep the signal was **continuation**, not reversal — worth confirming which way it is traded and what counts as confirmation across panels. Testable from free Binance spot + USD-M + COIN-M trade data.
3. **An in-house backtest harness**: signal → entry → exits → conservative fills → fees → first-half/second-half report, in Python. Optionally interactive, with the H1/H2 split always visible so that parameter tuning cannot hide a lucky window.

**Open questions**
- *Dominik:* what does the live TFT in Zurich read from?
- *Michael:* Speed of Waves — direction traded, confirmation rule across panels, venue set.
