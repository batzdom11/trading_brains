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
from pytorch_forecasting.metrics import QuantileLoss, MultiHorizonMetric
from google.cloud import bigquery, storage

from features import calculate_all_features


BQ_DATASET = "tft_predictions"
BQ_TABLE = "retraining_eval_logs"


class DirectionAwareQuantileLoss(MultiHorizonMetric):
    """
    Composite loss: QuantileLoss + direction penalty.
    Penalizes predictions where the predicted direction of change
    disagrees with the actual direction of change.
    """

    def __init__(
        self,
        quantile_weight: float = 0.7,
        direction_weight: float = 0.3,
        quantiles: list = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.quantile_weight = quantile_weight
        self.direction_weight = direction_weight
        self.quantiles = quantiles

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute composite loss.
        y_pred: (batch, horizon, n_quantiles)
        target: (batch, horizon)
        """
        # Quantile loss component
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            q_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(q_loss.unsqueeze(-1))
        quantile_loss = torch.cat(losses, dim=-1).mean(dim=-1)  # (batch, horizon)

        # Direction penalty: penalize when consecutive step directions disagree
        median_idx = len(self.quantiles) // 2
        pred_median = y_pred[..., median_idx]  # (batch, horizon)

        pred_change = pred_median[:, 1:] - pred_median[:, :-1]
        actual_change = target[:, 1:] - target[:, :-1]

        # Soft direction penalty using tanh (differentiable sign approximation)
        direction_agreement = torch.tanh(pred_change * 10) * torch.tanh(actual_change * 10)
        direction_penalty = torch.clamp(1.0 - direction_agreement, min=0.0) / 2.0

        # Pad first step (no direction info)
        pad = torch.zeros_like(direction_penalty[:, :1])
        direction_penalty = torch.cat([pad, direction_penalty], dim=1)  # (batch, horizon)

        return self.quantile_weight * quantile_loss + self.direction_weight * direction_penalty


def log_eval_metrics_to_bq(
    best_model_path: str,
    training: TimeSeriesDataSet,
    val_dataloader,
    trainer: pl.Trainer,
    args,
    data_start: str,
    data_end: str,
    training_rows: int,
    validation_rows: int,
    best_val_loss: float,
    gcs_model_uri: str,
):
    """Compute evaluation metrics on validation set and log to BigQuery."""
    print("\nComputing evaluation metrics on validation set...")

    # Load best model (strict=False needed for lightning 2.1.0 + pytorch-forecasting 1.6.1 compat)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, strict=False)

    # Get predictions
    predictions = best_tft.predict(val_dataloader, return_x=True)
    # predictions.output may be 3D (batch, prediction_length, quantiles) or 2D (batch, prediction_length)
    if predictions.output.dim() == 3:
        pred_values = predictions.output[:, :, 3]  # median quantile
    else:
        pred_values = predictions.output  # already median
    actuals = predictions.x["decoder_target"][:, :, 0] if predictions.x["decoder_target"].dim() == 3 else predictions.x["decoder_target"]

    pred_np = pred_values.detach().cpu().numpy().flatten()
    actual_np = actuals.detach().cpu().numpy().flatten()

    # Compute metrics
    errors = actual_np - pred_np
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    # MAPE: guard against zero actuals
    mask = actual_np != 0
    mape = float(np.mean(np.abs(errors[mask] / actual_np[mask])) * 100) if mask.any() else None
    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_np - np.mean(actual_np)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else None

    print(f"  MAE:        {mae:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAPE:       {mape:.4f}%" if mape is not None else "  MAPE:       N/A")
    print(f"  R²:         {r_squared:.4f}" if r_squared is not None else "  R²:         N/A")

    # Build row
    row = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "symbol": args.symbol,
        "best_val_loss": round(float(best_val_loss), 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6) if mape is not None else None,
        "r_squared": round(r_squared, 6) if r_squared is not None else None,
        "epochs_trained": trainer.current_epoch + 1,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "attention_head_size": args.attention_head_size,
        "dropout": args.dropout,
        "patience": args.patience,
        "lookback_days": args.lookback_days,
        "data_start_date": data_start,
        "data_end_date": data_end,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "model_gcs_path": gcs_model_uri,
    }

    # Insert into BigQuery (create table if needed)
    bq_client = bigquery.Client(project="trading-brains")
    table_id = f"trading-brains.{BQ_DATASET}.{BQ_TABLE}"

    schema = [
        bigquery.SchemaField("run_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("best_val_loss", "FLOAT64"),
        bigquery.SchemaField("mae", "FLOAT64"),
        bigquery.SchemaField("rmse", "FLOAT64"),
        bigquery.SchemaField("mape", "FLOAT64"),
        bigquery.SchemaField("r_squared", "FLOAT64"),
        bigquery.SchemaField("epochs_trained", "INT64"),
        bigquery.SchemaField("max_epochs", "INT64"),
        bigquery.SchemaField("batch_size", "INT64"),
        bigquery.SchemaField("learning_rate", "FLOAT64"),
        bigquery.SchemaField("hidden_size", "INT64"),
        bigquery.SchemaField("attention_head_size", "INT64"),
        bigquery.SchemaField("dropout", "FLOAT64"),
        bigquery.SchemaField("patience", "INT64"),
        bigquery.SchemaField("lookback_days", "INT64"),
        bigquery.SchemaField("data_start_date", "STRING"),
        bigquery.SchemaField("data_end_date", "STRING"),
        bigquery.SchemaField("training_rows", "INT64"),
        bigquery.SchemaField("validation_rows", "INT64"),
        bigquery.SchemaField("model_gcs_path", "STRING"),
    ]

    # Create table if it doesn't exist
    try:
        bq_client.get_table(table_id)
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        print(f"  Created BigQuery table: {table_id}")

    errors_bq = bq_client.insert_rows_json(table_id, [row])
    if errors_bq:
        print(f"  BigQuery insert errors: {errors_bq}")
    else:
        print(f"  Metrics logged to {table_id}")


def fetch_polygon_1min_data(symbol: str, start_date: str, end_date: str, api_key: str, max_retries: int = 10) -> pd.DataFrame:
    """Fetch 1-minute OHLCV data from Polygon.io with pagination and retry.
    
    Uses aggressive backoff (base 3, up to 10 retries) and 15s pause between
    paginated requests to stay within free-tier rate limits (5 calls/min).
    """
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
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=60)
                data = response.json()
            except (requests.RequestException, ValueError) as e:
                print(f"  Request error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait = min(3 ** attempt, 120)  # 3, 9, 27, 81, 120, 120...
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise ValueError(f"Failed to fetch data from Polygon for {symbol} after {max_retries} attempts")

            if data.get("status") in ("OK", "DELAYED") and "results" in data:
                break  # success

            print(f"  API response: {data.get('status', 'unknown')} - {data.get('message', '')} (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                wait = min(3 ** attempt, 120)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Giving up after {max_retries} attempts.")
                break
        else:
            break  # max retries exhausted on request error

        if data.get("status") not in ("OK", "DELAYED") or "results" not in data:
            break

        all_data.extend(data["results"])
        print(f"  Fetched {len(all_data)} bars so far...")

        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": api_key}
            time.sleep(15)  # 15s between pages to stay within free-tier rate limit
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


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95):
    """Remove features with pairwise correlation > threshold."""
    exclude_cols = [
        "timestamp", "time_idx", "group",
        "target_close_60m", "target_return_60m",
        "open", "high", "low", "close", "volume", "vwap",
    ]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype in ["float64", "float32", "int64"]
    ]

    df_clean = df[feature_cols].dropna()
    corr_matrix = df_clean.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"Correlation filtering (threshold={threshold}):")
    print(f"  Features before: {len(feature_cols)}")
    print(f"  Features dropped: {len(to_drop)}")
    print(f"  Features remaining: {len(feature_cols) - len(to_drop)}")

    return df.drop(columns=to_drop), to_drop


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

    # Correlation filtering to remove redundant features
    df_feat, dropped_cols = remove_highly_correlated_features(df_feat, threshold=0.95)

    # Add required columns
    df_feat["time_idx"] = range(len(df_feat))
    df_feat["group"] = "default"

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

    # Save metadata before freeing memory
    data_start = str(df["timestamp"].min())
    data_end = str(df["timestamp"].max())

    # Free raw data memory
    del df
    import gc
    gc.collect()

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

    # Save row counts before freeing memory
    training_rows = len(training)
    validation_rows = len(validation)

    # Free the raw DataFrame — TimeSeriesDataSet holds its own copy
    del df_tft
    gc.collect()

    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 5. Build model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        output_size=7,
        loss=DirectionAwareQuantileLoss(quantile_weight=0.7, direction_weight=0.3),
        reduce_on_plateau_patience=4,
    )
    print(f"\nModel parameters: {tft.size() / 1e3:.1f}k")

    # 6. Callbacks
    checkpoint_dir = "/tmp/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=args.patience, verbose=True, mode="min")
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
        gradient_clip_val=0.1,
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
    timestamped_path = f"tft_checkpoint_{args.symbol}_{timestamp}.ckpt"
    blob_ts = bucket.blob(timestamped_path)
    blob_ts.upload_from_filename(best_model_path)

    print(f"Uploaded: gs://{args.gcs_bucket}/{args.gcs_model_path}")
    print(f"Uploaded: gs://{args.gcs_bucket}/{timestamped_path}")

    # 9. Log evaluation metrics to BigQuery
    log_eval_metrics_to_bq(
        best_model_path=best_model_path,
        training=training,
        val_dataloader=val_dataloader,
        trainer=trainer,
        args=args,
        data_start=data_start,
        data_end=data_end,
        training_rows=training_rows,
        validation_rows=validation_rows,
        best_val_loss=best_val_loss,
        gcs_model_uri=f"gs://{args.gcs_bucket}/{args.gcs_model_path}",
    )

    # 10. Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Epochs trained: {trainer.current_epoch + 1}")
    print(f"  Best val_loss:  {best_val_loss:.4f}")
    print(f"  Data range:     {data_start} to {data_end}")
    print(f"  Training rows:  {training_rows}")
    print(f"  Model uploaded: gs://{args.gcs_bucket}/{args.gcs_model_path}")
    print(f"End time: {datetime.utcnow().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--polygon_api_key", required=True, help="Polygon.io API key")
    parser.add_argument("--gcs_bucket", default="tft-for-trading-brains", help="GCS bucket for model")
    parser.add_argument("--gcs_model_path", default=None, help="GCS path for model (default: tft_checkpoint_{symbol}_latest.ckpt)")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--lookback_days", type=int, default=420, help="Days of history to fetch")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=32, help="TFT hidden size")
    parser.add_argument("--attention_head_size", type=int, default=2, help="TFT attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--hidden_continuous_size", type=int, default=32, help="Hidden continuous size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers (keep low to avoid memory duplication)")
    args = parser.parse_args()

    # Default GCS model path includes symbol for multi-ticker support
    if args.gcs_model_path is None:
        args.gcs_model_path = f"tft_checkpoint_{args.symbol}_latest.ckpt"

    train_model(args)
