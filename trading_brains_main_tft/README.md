# trading_brains_main_tft/ — the main Trading Brains TFT model (Dominik)

| File | Purpose |
|---|---|
| `trading.ipynb` | Original TFT notebook |
| `temporal_transformers_exploration.ipynb` | TFT exploration; writes `models/` and `checkpoints/tft` |
| `temporal_transformers_simulation.ipynb` | Simulation using `models/tft_checkpoint_latest.ckpt` |
| `tft_improvements_experiment.ipynb` | Improvement experiments |
| `trading_simulation.ipynb` | Trading simulation |
| `train_local.ipynb` | Local training (TFT + direction classifier) |
| `prediction_logs_analytics.ipynb` | Analysis of the live prediction logs in BigQuery |
| `data/` | SPY raw minute data used for training |
| `models/`, `checkpoints/`, `lightning_logs/` | Local training artifacts (git-ignored for new files) |

Run the notebooks from this folder so the relative paths resolve.
