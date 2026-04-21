"""
TFT model training script for Vertex AI Custom Training.

Usage:
  python train.py \
    --polygon_api_key <KEY> \
    --gcs_bucket tft-for-trading-brains \
    --gcs_model_path tft_checkpoint_latest.ckpt \
    --symbol SPY \
    --lookback_days 365 \
    --max_epochs 10 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_size 32 \
    --attention_head_size 2 \
    --dropout 0.1 \
    --patience 5
"""

import argparse
import os
import shutil
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from google.cloud import storage

from features import calculate_all_features


def fetch_polygon_1min_data(symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Fetch 1-minute OHLCV data from Polygon.io with pagination."""
    import requests

    all_data = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    print(f"Fetching {symbol} data from {start_date} to {end_date}...")

    while url:
        response = requests.get(url, params=params, timeout=60)
        data = response.json()

        if data.get("status") not in ("OK", "DELAYED") or "results" not in data:
            print(f"API response: {data.get('status', 'unknown')} - {data.get('message', '')}")
            break

        all_data.extend(data["results"])
        print(f"  Fetched {len(all_data)} bars so far...")

        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": api_key}
            time.sleep(0.5)  # Rate limit
        else:
            url = None

    if not all_data:
        raise ValueError(f"No data returned from Polygon for {symbol}")

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Total: {len(df)} 1-minute bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def prepare_dataset(df: pd.DataFrame):
    """Run feature engineering and prepare the TFT-ready DataFrame."""
    # Calculate VWAP
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    # Feature engineering
    df_feat = calculate_all_features(df)

    # Drop rows with NaN from indicators (but keep target NaNs for now)
    target_cols = ["target_close_60m", "target_return_60m"]
    non_target = [c for c in df_feat.columns if c not in target_cols]
    df_feat = df_feat.dropna(subset=non_target)

    # Drop rows where target is NaN (last 60 rows won't have a label)
    df_feat = df_feat.dropna(subset=target_cols)
    df_feat = df_feat.reset_index(drop=True)

    # Add required columns
    df_feat["time_idx"] = range(len(df_feat))
    df_feat["group"] = "SPY"

    # Clean infinities
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    df_feat[numeric_cols] = df_feat[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_feat[numeric_cols] = df_feat[numeric_cols].ffill().bfill()

    print(f"Prepared dataset: {len(df_feat)} rows, {len(df_feat.columns)} columns")
    return df_feat


def build_feature_lists(df: pd.DataFrame):
    """Derive the time-varying known and unknown reals lists."""
    time_varying_known_reals = ["hour_sin", "hour_cos", "minute_sin", "minute_cos", "day_sin", "day_cos"]
    time_varying_known_reals = [c for c in time_varying_known_reals if c in df.columns]

    exclude = [
        "timestamp", "time_idx", "group",
        "target_close_60m", "target_return_60m",
        "hour", "minute", "day_of_week",
        "is_morning", "is_afternoon",
    ] + time_varying_known_reals

    time_varying_unknown_reals = [c for c in df.columns if c not in exclude]

    # Ensure 'close' (the target) is first
    if "close" in time_varying_unknown_reals:
        time_varying_unknown_reals.remove("close")
    time_varying_unknown_reals = ["close"] + time_varying_unknown_reals

    print(f"Time-varying known reals: {len(time_varying_known_reals)}")
    print(f"Time-varying unknown reals: {len(time_varying_unknown_reals)}")
    return time_varying_known_reals, time_varying_unknown_reals


def train_model(args):
    """Main training function."""
    print("=" * 60)
    print("TFT Training Pipeline")
    print(f"Start time: {datetime.utcnow().isoformat()}")
    print("=" * 60)

    # 1. Fetch data
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
    df = fetch_polygon_1min_data(args.symbol, start_date, end_date, args.polygon_api_key)

    # 2. Feature engineering
    df_tft = prepare_dataset(df)

    # 3. Build feature lists
    time_varying_known_reals, time_varying_unknown_reals = build_feature_lists(df_tft)

    # Ensure all listed features exist
    time_varying_known_reals = [c for c in time_varying_known_reals if c in df_tft.columns]
    time_varying_unknown_reals = [c for c in time_varying_unknown_reals if c in df_tft.columns]

    # 4. Train/validation split (80/20 as in notebook)
    max_encoder_length = 60
    max_prediction_length = 60
    training_cutoff = int(len(df_tft) * 0.8)

    print(f"\nDataset size: {len(df_tft)} rows")
    print(f"Training cutoff: time_idx={training_cutoff}")
    print(f"Training rows: {len(df_tft[df_tft['time_idx'] <= training_cutoff])}")
    print(f"Validation rows: {len(df_tft[df_tft['time_idx'] > training_cutoff])}")

    training = TimeSeriesDataSet(
        df_tft[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="close",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group"],
        time_varying_known_categoricals=[],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["group"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_tft[lambda x: x.time_idx > training_cutoff],
        predict=False,
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    # 5. Build model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        optimizer="adam",
    )
    print(f"\nModel parameters: {tft.size() / 1e3:.1f}k")

    # 6. Callbacks
    checkpoint_dir = "/tmp/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # 7. Train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[early_stop, lr_monitor, checkpoint_cb],
        enable_model_summary=True,
        accelerator="auto",
    )

    print("\nStarting training...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = checkpoint_cb.best_model_path
    best_val_loss = checkpoint_cb.best_model_score
    print(f"\nBest model: {best_model_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")

    # 8. Upload to GCS
    print(f"\nUploading checkpoint to gs://{args.gcs_bucket}/{args.gcs_model_path}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.gcs_bucket)

    # Upload as latest
    blob = bucket.blob(args.gcs_model_path)
    blob.upload_from_filename(best_model_path)

    # Upload timestamped copy
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    timestamped_path = f"tft_checkpoint_{timestamp}.ckpt"
    blob_ts = bucket.blob(timestamped_path)
    blob_ts.upload_from_filename(best_model_path)

    print(f"Uploaded: gs://{args.gcs_bucket}/{args.gcs_model_path}")
    print(f"Uploaded: gs://{args.gcs_bucket}/{timestamped_path}")

    # 9. Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Epochs trained: {trainer.current_epoch + 1}")
    print(f"  Best val_loss:  {best_val_loss:.4f}")
    print(f"  Data range:     {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Training rows:  {len(df_tft[df_tft['time_idx'] <= training_cutoff])}")
    print(f"  Model uploaded: gs://{args.gcs_bucket}/{args.gcs_model_path}")
    print(f"End time: {datetime.utcnow().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--polygon_api_key", required=True, help="Polygon.io API key")
    parser.add_argument("--gcs_bucket", default="tft-for-trading-brains", help="GCS bucket for model")
    parser.add_argument("--gcs_model_path", default="tft_checkpoint_latest.ckpt", help="GCS path for model")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--lookback_days", type=int, default=420, help="Days of history to fetch")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=32, help="TFT hidden size")
    parser.add_argument("--attention_head_size", type=int, default=2, help="TFT attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    train_model(args)
