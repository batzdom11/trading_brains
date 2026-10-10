# side_experiments_and_research/ — strategy research (Matthias & Claude)

Current status and all findings: [`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md).

| Folder | Contents |
|---|---|
| `btc_mmt/` | BTC 1-minute MMT dataset (`trading-brains.market_microstructure.mmt_btc_1m_v3`): direction classifier, barrier labels, order-flow ablation, walk-forward experiments |
| `spy_vs_btc/` | Control experiment: identical pipeline on SPY vs BTC (`ASSET` switch), results CSVs. Needs a local `spy_1m.csv` (Massive/Polygon minute bars) |
| `early_2026/` | Early notebook experiments (March 2026) |
| `mmt_scripts/` | Backtesters written in MMT Scripting v3 for app.mmt.gg (paste into the script editor) |

## MMT scripts

All share the same harness: conservative fills (stop before target), maker/taker
fees, a signal-free random-entry benchmark with the same exits, automatic
first/second-half split, results in bps, R and $ for a chosen account size.

| Script | Setup | Verdict (Oct 2026) |
|---|---|---|
| `volume_backtester.mmt` | Heavy-volume zone, proof, retest; CVD-divergence and whale/retail tags | No edge after costs on BTC/ETH/SOL 1m–15m |
| `trapped_traders.mmt` | Heavy bar + delta, CVD turn, reclaim of the bar; prop-firm panel | No edge on 15m |
| `multi_venue_aggression.mmt` | Same-direction delta spikes on several venues. NB: `"deribit"` is Deribit **spot**; inverse markets are `*-inverse` | No edge on 1m |
| `bubble_exhaustion.mmt` | Fade big delta bubbles at fresh highs/lows | ≈ +3 bps gross, negative net |
| `bubble_resolver.mmt` | Bubble → continuation / absorption / **reclaim** (branch 3) from $1M+ trade delta, 7 futures venues | **Reclaim: positive before costs on 4 of 6 charts (≈ +4 bps pooled), below ≈ 12 bps costs.** Next: Python on Binance trades |
| `stacked_imbalance.mmt` | Stacked footprint imbalances (9 venues) as S/R, fade first touch | No edge; footprint history only ≈ 17 % of bars |
| `exchange_names.mmt` | Lists MMT exchange ids for the chart's symbol | helper |
| `MMT.sublime-syntax` | Syntax highlighting for `.mmt` in Sublime Text | helper |
