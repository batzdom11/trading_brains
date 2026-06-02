import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'
os.environ['USE_CUDA'] = '0'

import torch
# Aggressively disable all CUDA
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: -1
torch.cuda.get_device_name = lambda x=None: ''
torch.cuda.init = lambda: None
torch.cuda.set_device = lambda x: None

# Disable cudnn
if hasattr(torch.backends, 'cudnn'):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.is_available = lambda: False

from flask import Flask, jsonify, request

# Required for loading checkpoints trained with custom loss
from pytorch_forecasting.metrics import MultiHorizonMetric

class DirectionAwareQuantileLoss(MultiHorizonMetric):
    def __init__(self, quantile_weight=0.7, direction_weight=0.3,
                 quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98], **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.quantile_weight = quantile_weight
        self.direction_weight = direction_weight
        self.quantiles = quantiles

    def loss(self, y_pred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            q_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(q_loss.unsqueeze(-1))
        quantile_loss = torch.cat(losses, dim=-1).mean(dim=-1)
        median_idx = len(self.quantiles) // 2
        pred_median = y_pred[..., median_idx]
        pred_change = pred_median[:, 1:] - pred_median[:, :-1]
        actual_change = target[:, 1:] - target[:, :-1]
        direction_agreement = torch.tanh(pred_change * 10) * torch.tanh(actual_change * 10)
        direction_penalty = torch.clamp(1.0 - direction_agreement, min=0.0) / 2.0
        pad = torch.zeros_like(direction_penalty[:, :1])
        direction_penalty = torch.cat([pad, direction_penalty], dim=1)
        return self.quantile_weight * quantile_loss + self.direction_weight * direction_penalty

# Make DirectionAwareQuantileLoss findable when loading checkpoints pickled under __main__
import sys
if '__main__' in sys.modules:
    sys.modules['__main__'].DirectionAwareQuantileLoss = DirectionAwareQuantileLoss
else:
    import types
    fake_main = types.ModuleType('__main__')
    fake_main.DirectionAwareQuantileLoss = DirectionAwareQuantileLoss
    sys.modules['__main__'] = fake_main

app = Flask(__name__)

# Global variables for lazy loading
GCS_BUCKET = "tft-for-trading-brains"

# Mapping from model ticker (GCS filename) to Twelve Data symbol
TICKER_TO_TWELVEDATA = {
    'BTC': 'BTC/USD',
    'ETH': 'ETH/USD',
}

def get_twelvedata_symbol(ticker):
    """Convert model ticker to Twelve Data symbol format."""
    return TICKER_TO_TWELVEDATA.get(ticker, ticker)

# Per-ticker model cache: {ticker: (model, dataset_params, device)}
_model_cache = {}

def _ensure_predictions_schema():
    """Add quantile/confidence columns to tft_predictions_logs if missing."""
    from google.cloud import bigquery
    client = bigquery.Client(project="trading-brains")
    table_id = "trading-brains.tft_predictions.tft_predictions_logs"
    try:
        table = client.get_table(table_id)
        existing_fields = {f.name for f in table.schema}
        new_fields = [
            ("q10_15m", "FLOAT64"), ("q90_15m", "FLOAT64"),
            ("q10_30m", "FLOAT64"), ("q90_30m", "FLOAT64"),
            ("q10_45m", "FLOAT64"), ("q90_45m", "FLOAT64"),
            ("q10_60m", "FLOAT64"), ("q90_60m", "FLOAT64"),
            ("confidence_15m", "FLOAT64"), ("confidence_30m", "FLOAT64"),
            ("confidence_45m", "FLOAT64"), ("confidence_60m", "FLOAT64"),
            ("uncertainty_15m", "FLOAT64"), ("uncertainty_30m", "FLOAT64"),
            ("uncertainty_45m", "FLOAT64"), ("uncertainty_60m", "FLOAT64"),
        ]
        to_add = [(n, t) for n, t in new_fields if n not in existing_fields]
        if to_add:
            new_schema = list(table.schema) + [
                bigquery.SchemaField(name, dtype) for name, dtype in to_add
            ]
            table.schema = new_schema
            client.update_table(table, ["schema"])
            print(f"Added {len(to_add)} columns to {table_id}: {[n for n,_ in to_add]}")
        else:
            print("Predictions table schema already up to date.")
    except Exception as e:
        print(f"Warning: Schema migration failed: {e}")

_ensure_predictions_schema()

@app.route('/')
def health():
    """Health check endpoint - must respond fast"""
    return jsonify({'status': 'healthy'})

def get_model(ticker='SPY'):
    """Lazy load model on first request, cached per ticker"""
    global _model_cache
    
    if ticker not in _model_cache:
        import torch
        from google.cloud import storage
        
        # Patch torchmetrics BEFORE importing pytorch_forecasting
        import torchmetrics
        _original_apply = torchmetrics.Metric._apply
        def _patched_apply(self, fn):
            self._device = torch.device('cpu')  # Force CPU before apply
            return _original_apply(self, fn)
        torchmetrics.Metric._apply = _patched_apply
        
        from pytorch_forecasting import TemporalFusionTransformer
        
        gcs_model_path = f"tft_checkpoint_{ticker}_latest.ckpt"
        local_model_path = f"/tmp/tft_checkpoint_{ticker}_latest.ckpt"
        
        print(f"Downloading model from gs://{GCS_BUCKET}/{gcs_model_path}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_model_path)
        blob.download_to_filename(local_model_path)
        print(f"Model for {ticker} downloaded successfully!")
        
        model = TemporalFusionTransformer.load_from_checkpoint(local_model_path, map_location='cpu')
        model.eval()
        dataset_params = model.dataset_parameters
        device = torch.device('cpu')
        print(f"Model for {ticker} loaded on device: {device}")
        
        _model_cache[ticker] = (model, dataset_params, device)
    
    return _model_cache[ticker]


def get_model_group(dataset_params):
    """Get the group value from the model's categorical encoders."""
    known_groups = dataset_params.get('categorical_encoders', {}).get('__group_id__group', None)
    if known_groups and hasattr(known_groups, 'classes_'):
        classes = known_groups.classes_
        if isinstance(classes, dict):
            return list(classes.keys())[0]
        else:
            return classes[0]
    return 'SPY'

def update_normalizer_stats(dataset_params, close_series):
    """
    Update the GroupNormalizer's stored center/scale to match current data.
    
    The model's normalizer stores training-time statistics (mean ~600, std ~25).
    When current prices are ~700, the inverse transform anchors predictions to
    the old range. By updating center/scale with current data, the post-hoc
    rescaling correctly maps model outputs to current price levels.
    """
    import numpy as np
    import copy

    params = copy.deepcopy(dataset_params)
    normalizer = params.get("target_normalizer")
    if normalizer is None or not hasattr(normalizer, "norm_"):
        return params

    values = close_series.values.astype(float)
    # Apply the same preprocessing the normalizer uses (softplus_inv)
    if hasattr(normalizer, "preprocess"):
        import torch
        preprocessed = normalizer.preprocess(torch.tensor(values)).numpy()
    else:
        preprocessed = values

    new_center = float(np.mean(preprocessed))
    new_scale = float(np.std(preprocessed) + np.finfo(np.float16).eps)

    for group_id in normalizer.norm_.index:
        normalizer.norm_.loc[group_id, "center"] = new_center
        normalizer.norm_.loc[group_id, "scale"] = new_scale

    if hasattr(normalizer, "missing_"):
        normalizer.missing_["center"] = new_center
        normalizer.missing_["scale"] = new_scale

    return params

def calculate_all_features(df):
    """
    Calculate all 75 features required for the TFT model.
    Input df must have: timestamp, open, high, low, close, volume, vwap
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    
    # ========================
    # 1. RETURNS (5 features)
    # ========================
    df['returns_1m'] = df['close'].pct_change(1)
    df['returns_5m'] = df['close'].pct_change(5)
    df['returns_15m'] = df['close'].pct_change(15)
    df['returns_30m'] = df['close'].pct_change(30)
    df['returns_60m'] = df['close'].pct_change(60)
    
    # ========================
    # 2. PRICE RATIOS (6 features)
    # ========================
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-10)
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # ========================
    # 3. SMA FEATURES (11 features)
    # ========================
    for period in [5, 10, 20, 30]:
        sma = df['close'].rolling(window=period).mean()
        df[f'sma_{period}_slope'] = sma.pct_change()
        df[f'close_to_sma_{period}'] = (df['close'] - sma) / sma
    
    # SMA 60 - only ratio, no slope in original
    sma_60 = df['close'].rolling(window=60).mean()
    df['close_to_sma_60'] = (df['close'] - sma_60) / sma_60
    
    # SMA 120 - with slope
    sma_120 = df['close'].rolling(window=120).mean()
    df['sma_120_slope'] = sma_120.pct_change()
    df['close_to_sma_120'] = (df['close'] - sma_120) / sma_120
    
    # ========================
    # 4. MACD (2 features)
    # ========================
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    signal_line = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - signal_line
    
    # ========================
    # 5. VOLATILITY (4 features)
    # ========================
    df['volatility_5'] = df['returns_1m'].rolling(window=5).std()
    df['volatility_10'] = df['returns_1m'].rolling(window=10).std()
    df['volatility_20'] = df['returns_1m'].rolling(window=20).std()
    df['volatility_60'] = df['returns_1m'].rolling(window=60).std()
    
    # ========================
    # 6. ATR (2 features)
    # ========================
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_60'] = tr.rolling(window=60).mean()
    
    # ========================
    # 7. BOLLINGER BANDS (4 features)
    # ========================
    for period in [20, 60]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_std_{period}'] = std
        df[f'bb_position_{period}'] = (df['close'] - sma) / (2 * std + 1e-10)
    
    # ========================
    # 8. RSI (3 features)
    # ========================
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = calc_rsi(df['close'], 14)
    df['rsi_20'] = calc_rsi(df['close'], 20)
    df['rsi_60'] = calc_rsi(df['close'], 60)
    
    # ========================
    # 9. STOCHASTIC (3 features)
    # ========================
    for period in [14, 60]:
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    df['stoch_d_14'] = df['stoch_k_14'].rolling(window=3).mean()
    
    # ========================
    # 10. RATE OF CHANGE (2 features)
    # ========================
    df['roc_10'] = df['close'].pct_change(10) * 100
    df['roc_20'] = df['close'].pct_change(20) * 100
    
    # ========================
    # 11. VOLUME INDICATORS (12 features)
    # ========================
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_change_5m'] = df['volume'].pct_change(5)
    
    for period in [5, 10, 20, 60]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-10)
    
    # ========================
    # 12. OBV, VPT, MFI (5 features)
    # ========================
    # OBV
    obv = np.where(df['close'] > df['close'].shift(), df['volume'],
                   np.where(df['close'] < df['close'].shift(), -df['volume'], 0))
    df['obv'] = np.cumsum(obv)
    obv_sma = pd.Series(df['obv']).rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / (obv_sma + 1e-10)
    
    # VPT (Volume Price Trend)
    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
    
    # MFI (Money Flow Index)
    def calc_mfi(df, period):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        return mfi
    
    df['mfi_14'] = calc_mfi(df, 14)
    df['mfi_60'] = calc_mfi(df, 60)
    
    # ========================
    # 13. VWAP RATIOS (2 features)
    # ========================
    vwap_20 = (df['volume'] * df['close']).rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
    vwap_60 = (df['volume'] * df['close']).rolling(window=60).sum() / (df['volume'].rolling(window=60).sum() + 1e-10)
    df['close_to_vwap_20'] = (df['close'] - vwap_20) / vwap_20
    df['close_to_vwap_60'] = (df['close'] - vwap_60) / vwap_60
    
    # ========================
    # 14. TIME FEATURES (10 features)
    # ========================
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # ========================
    # 15. VOLUME LAGS (5 features)
    # ========================
    df['volume_lag_1'] = df['volume'].shift(1)
    df['volume_lag_5'] = df['volume'].shift(5)
    df['volume_lag_15'] = df['volume'].shift(15)
    df['volume_lag_30'] = df['volume'].shift(30)
    df['volume_lag_60'] = df['volume'].shift(60)
    
    # ========================
    # 16. TARGET VARIABLES (2 features)
    # ========================
    df['target_close_60m'] = df['close'].shift(-60)
    df['target_return_60m'] = df['close'].pct_change(60).shift(-60)
    
    return df

def download_market_data(ticker='SPY', days=7):
    """Download intraday data from Twelve Data API (free tier supports 1-min US stocks)"""
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    import time
    
    api_key = os.environ.get('TWELVEDATA_API_KEY')
    if not api_key:
        raise ValueError("TWELVEDATA_API_KEY environment variable not set")
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Twelve Data supports outputsize up to 5000 for 1-min data
            # Crypto trades 24/7 (~1440 min/day), stocks ~390 min/day
            is_crypto = ticker in TICKER_TO_TWELVEDATA
            minutes_per_day = 1440 if is_crypto else 390
            outputsize = min(5000, days * minutes_per_day)
            
            print(f"Attempt {attempt + 1}: Downloading {ticker} data from Twelve Data...")
            
            url = "https://api.twelvedata.com/time_series"
            twelvedata_symbol = get_twelvedata_symbol(ticker)
            params = {
                'symbol': twelvedata_symbol,
                'interval': '1min',
                'outputsize': outputsize,
                'apikey': api_key,
                'timezone': 'America/New_York'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'code' in data and data['code'] != 200:
                error_msg = data.get('message', 'Unknown error')
                print(f"Twelve Data error: {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                last_error = Exception(error_msg)
                continue
            
            if 'values' not in data or not data['values']:
                print(f"No data returned on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
            
            # Parse the response
            df = pd.DataFrame(data['values'])
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Volume may be missing for some crypto pairs
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume'] = 1.0
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Successfully downloaded {len(df)} rows from Twelve Data")
            return df
            
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    raise ValueError(f"No market data available after {max_retries} attempts. Last error: {last_error}")

def get_predictions(ticker='SPY'):
    """Download data, engineer features, run model, return predictions"""
    import numpy as np
    import pandas as pd
    import torch
    from datetime import datetime
    from pytorch_forecasting import TimeSeriesDataSet
    
    # Get model (lazy load)
    model, dataset_params, device = get_model(ticker)
    
    # 1. Download real-time data
    df = download_market_data(ticker, days=7)
    
    # Calculate VWAP (use typical price if volume is zero/missing)
    if df['volume'].sum() > 0:
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    else:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    
    # 2. Feature engineering
    df_features = calculate_all_features(df)
    
    # 3. Prepare for prediction
    df_pred = df_features.copy()
    # Target columns only needed for training; fill to prevent dropna from dropping last 60 rows
    for col in ['target_close_60m', 'target_return_60m']:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].ffill().fillna(0)
    df_pred = df_pred.dropna()
    df_pred = df_pred.reset_index(drop=True)
    df_pred['time_idx'] = range(len(df_pred))
    df_pred['group'] = get_model_group(dataset_params)
    
    # Clean data
    numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
    df_pred[numeric_cols] = df_pred[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_pred[numeric_cols] = df_pred[numeric_cols].ffill().bfill()
    
    # Take recent data
    max_encoder_length = dataset_params.get('max_encoder_length', 60)
    max_prediction_length = dataset_params.get('max_prediction_length', 60)
    min_rows = max_encoder_length + max_prediction_length + 100
    df_recent = df_pred.iloc[-min_rows:].copy()
    df_recent['time_idx'] = range(len(df_recent))
    
    # 4. Update normalizer to current price level and predict
    updated_params = update_normalizer_stats(dataset_params, df_recent['close'])
    prediction_dataset = TimeSeriesDataSet.from_parameters(
        updated_params,
        df_recent,
        predict=True,
    )
    pred_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    # Use model.predict() which automatically applies inverse normalization
    raw_pred = model.predict(pred_dataloader, mode="prediction")
    pred_array = raw_pred.squeeze().cpu().numpy()
    if len(pred_array.shape) == 2:
        pred_array = pred_array[:, pred_array.shape[1] // 2]

    # Get full quantile output for confidence scoring
    # Quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    raw_quantiles = model.predict(pred_dataloader, mode="quantiles")
    q_array = raw_quantiles.squeeze().cpu().numpy()
    # q_array shape: (60, 7) — 60 timesteps × 7 quantiles
    # Indices: 0=q02, 1=q10, 2=q25, 3=q50, 4=q75, 5=q90, 6=q98
    
    # 5. Extract predictions at 15, 30, 45, 60 minutes
    # Last encoder position = end of lookback window (before prediction horizon)
    last_close = df_recent['close'].iloc[-(max_prediction_length + 1)]
    last_timestamp = df_recent['timestamp'].iloc[-(max_prediction_length + 1)]

    # Extract quantile bounds at each horizon
    def _quantile_confidence(q_arr, timestep, base_price):
        """Compute confidence from quantile spread.
        Returns (q10, q90, confidence_score).
        Confidence = how far the interval is from base_price relative to interval width.
        Positive = bullish confidence, negative = bearish confidence.
        """
        q10 = float(q_arr[timestep, 1])   # 10th percentile
        q90 = float(q_arr[timestep, 5])   # 90th percentile
        interval_width = q90 - q10
        if interval_width < 1e-8:
            return q10, q90, 0.0
        midpoint = (q10 + q90) / 2.0
        # How far is base_price from the interval center, normalized by width
        # Positive = model expects price above current (bullish)
        # Negative = model expects price below current (bearish)
        confidence = (midpoint - base_price) / interval_width
        return q10, q90, float(confidence)

    q10_15m, q90_15m, conf_15m = _quantile_confidence(q_array, 14, last_close)
    q10_30m, q90_30m, conf_30m = _quantile_confidence(q_array, 29, last_close)
    q10_45m, q90_45m, conf_45m = _quantile_confidence(q_array, 44, last_close)
    q10_60m, q90_60m, conf_60m = _quantile_confidence(q_array, 59, last_close)

    predictions = {
        'timestamp': datetime.utcnow().isoformat(),
        'ticker': ticker,
        'last_price': float(last_close),
        'last_price_time': str(last_timestamp),
        'pred_15m': float(pred_array[14]),
        'pred_30m': float(pred_array[29]),
        'pred_45m': float(pred_array[44]),
        'pred_60m': float(pred_array[59]),
        'return_15m': float((pred_array[14] - last_close) / last_close * 100),
        'return_30m': float((pred_array[29] - last_close) / last_close * 100),
        'return_45m': float((pred_array[44] - last_close) / last_close * 100),
        'return_60m': float((pred_array[59] - last_close) / last_close * 100),
        # Quantile bounds (10th and 90th percentile)
        'q10_15m': q10_15m,
        'q90_15m': q90_15m,
        'q10_30m': q10_30m,
        'q90_30m': q90_30m,
        'q10_45m': q10_45m,
        'q90_45m': q90_45m,
        'q10_60m': q10_60m,
        'q90_60m': q90_60m,
        # Confidence scores: positive=bullish, negative=bearish, magnitude=strength
        'confidence_15m': conf_15m,
        'confidence_30m': conf_30m,
        'confidence_45m': conf_45m,
        'confidence_60m': conf_60m,
        # Prediction interval width as % of price (model uncertainty)
        'uncertainty_15m': float((q90_15m - q10_15m) / last_close * 100),
        'uncertainty_30m': float((q90_30m - q10_30m) / last_close * 100),
        'uncertainty_45m': float((q90_45m - q10_45m) / last_close * 100),
        'uncertainty_60m': float((q90_60m - q10_60m) / last_close * 100),
    }
    
    # 6. Extract TFT attention weights and feature importance
    try:
        interpretation = _extract_interpretation(model, pred_dataloader, ticker, predictions['timestamp'])
        predictions['_interpretation'] = interpretation
    except Exception as e:
        print(f"Warning: Could not extract interpretation: {e}")
    
    # 7. Fetch VIX and compute volatility context
    try:
        vol_context = _fetch_volatility_context(ticker, df_recent)
        predictions['_volatility_context'] = vol_context
    except Exception as e:
        print(f"Warning: Could not fetch volatility context: {e}")
    
    return predictions


def _extract_interpretation(model, pred_dataloader, ticker, prediction_timestamp):
    """Extract TFT attention weights and variable importance from model output."""
    import torch
    import numpy as np

    # Get a single batch from the dataloader
    batch = next(iter(pred_dataloader))
    x, y = batch

    # Get raw model output (not the denormalized predictions)
    with torch.no_grad():
        raw_output = model(x)

    # interpret_output gives us attention and variable importance
    interpretation = model.interpret_output(raw_output, reduction="none")

    result = {}

    # Temporal attention weights: shape (batch, encoder_length)
    if "attention" in interpretation:
        attention = interpretation["attention"].squeeze().cpu().numpy()
        # attention[i] = how much the model attended to encoder timestep i
        result["temporal_attention"] = attention.tolist()

    # Encoder variable importance: shape (batch, n_encoder_vars)
    if "encoder_variables" in interpretation:
        enc_imp = interpretation["encoder_variables"].mean(dim=0).cpu().numpy()
        # Map indices to variable names
        encoder_vars = model.encoder_variables
        result["encoder_importance"] = {
            name: float(enc_imp[i]) for i, name in enumerate(encoder_vars) if i < len(enc_imp)
        }

    # Decoder variable importance: shape (batch, n_decoder_vars)
    if "decoder_variables" in interpretation:
        dec_imp = interpretation["decoder_variables"].mean(dim=0).cpu().numpy()
        decoder_vars = model.decoder_variables
        result["decoder_importance"] = {
            name: float(dec_imp[i]) for i, name in enumerate(decoder_vars) if i < len(dec_imp)
        }

    # Static variable importance (if any)
    if "static_variables" in interpretation and interpretation["static_variables"].numel() > 0:
        static_imp = interpretation["static_variables"].mean(dim=0).cpu().numpy()
        static_vars = model.static_variables
        result["static_importance"] = {
            name: float(static_imp[i]) for i, name in enumerate(static_vars) if i < len(static_imp)
        }

    return result


def _fetch_volatility_context(ticker, df_recent):
    """Fetch VIX from TwelveData and compute asset-specific realized volatility."""
    import requests
    import numpy as np

    api_key = os.environ.get('TWELVEDATA_API_KEY')
    if not api_key:
        return {}

    result = {}

    # Compute asset-specific realized volatility from recent data
    close = df_recent['close'].values
    returns = np.diff(close) / close[:-1]
    result['realized_vol_30m'] = float(np.std(returns[-30:]) * np.sqrt(390)) if len(returns) >= 30 else None
    result['realized_vol_1h'] = float(np.std(returns[-60:]) * np.sqrt(390)) if len(returns) >= 60 else None
    result['realized_vol_4h'] = float(np.std(returns[-240:]) * np.sqrt(390)) if len(returns) >= 240 else None

    # Classify volatility regime based on 1h realized vol
    if result['realized_vol_1h'] is not None:
        vol = result['realized_vol_1h']
        if vol < 0.10:
            result['vol_regime'] = 'low'
        elif vol < 0.25:
            result['vol_regime'] = 'medium'
        else:
            result['vol_regime'] = 'high'

    # Fetch VIX quote from TwelveData
    try:
        url = "https://api.twelvedata.com/quote"
        params = {'symbol': 'VIX', 'apikey': api_key}
        resp = requests.get(url, params=params, timeout=10)
        vix_data = resp.json()
        if 'close' in vix_data:
            result['vix_level'] = float(vix_data['close'])
        elif 'previous_close' in vix_data:
            result['vix_level'] = float(vix_data['previous_close'])
    except Exception as e:
        print(f"Warning: VIX fetch failed: {e}")

    return result


def _save_interpretation_to_bq(predictions):
    """Save attention weights and feature importance to BigQuery."""
    from google.cloud import bigquery
    from datetime import datetime

    interpretation = predictions.get('_interpretation')
    vol_context = predictions.get('_volatility_context')
    if not interpretation and not vol_context:
        return

    client = bigquery.Client(project="trading-brains")
    prediction_ts = predictions['timestamp']
    ticker = predictions['ticker']

    # 1. Save feature importance
    if interpretation and ('encoder_importance' in interpretation or 'decoder_importance' in interpretation):
        table_id = "trading-brains.tft_predictions.feature_importance"
        schema = [
            bigquery.SchemaField("prediction_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("variable_name", "STRING"),
            bigquery.SchemaField("variable_type", "STRING"),
            bigquery.SchemaField("importance_score", "FLOAT64"),
        ]
        # Ensure table exists
        try:
            client.get_table(table_id)
        except Exception:
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Created BigQuery table: {table_id}")

        rows = []
        for var_name, score in interpretation.get('encoder_importance', {}).items():
            rows.append({
                'prediction_timestamp': prediction_ts,
                'ticker': ticker,
                'variable_name': var_name,
                'variable_type': 'encoder',
                'importance_score': round(score, 6),
            })
        for var_name, score in interpretation.get('decoder_importance', {}).items():
            rows.append({
                'prediction_timestamp': prediction_ts,
                'ticker': ticker,
                'variable_name': var_name,
                'variable_type': 'decoder',
                'importance_score': round(score, 6),
            })
        for var_name, score in interpretation.get('static_importance', {}).items():
            rows.append({
                'prediction_timestamp': prediction_ts,
                'ticker': ticker,
                'variable_name': var_name,
                'variable_type': 'static',
                'importance_score': round(score, 6),
            })

        if rows:
            errors = client.insert_rows_json(table_id, rows)
            if errors:
                print(f"Feature importance BQ errors: {errors}")

    # 2. Save temporal attention weights
    if interpretation and 'temporal_attention' in interpretation:
        table_id = "trading-brains.tft_predictions.temporal_attention"
        schema = [
            bigquery.SchemaField("prediction_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("timestep_offset", "INT64"),
            bigquery.SchemaField("attention_weight", "FLOAT64"),
        ]
        try:
            client.get_table(table_id)
        except Exception:
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Created BigQuery table: {table_id}")

        attention = interpretation['temporal_attention']
        encoder_length = len(attention)
        rows = []
        for i, weight in enumerate(attention):
            rows.append({
                'prediction_timestamp': prediction_ts,
                'ticker': ticker,
                'timestep_offset': i - encoder_length,  # -60 to -1
                'attention_weight': round(float(weight), 6),
            })

        if rows:
            errors = client.insert_rows_json(table_id, rows)
            if errors:
                print(f"Temporal attention BQ errors: {errors}")

    # 3. Save volatility context
    if vol_context:
        table_id = "trading-brains.tft_predictions.volatility_context"
        schema = [
            bigquery.SchemaField("prediction_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("realized_vol_30m", "FLOAT64"),
            bigquery.SchemaField("realized_vol_1h", "FLOAT64"),
            bigquery.SchemaField("realized_vol_4h", "FLOAT64"),
            bigquery.SchemaField("vix_level", "FLOAT64"),
            bigquery.SchemaField("vol_regime", "STRING"),
        ]
        try:
            client.get_table(table_id)
        except Exception:
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Created BigQuery table: {table_id}")

        row = {
            'prediction_timestamp': prediction_ts,
            'ticker': ticker,
            'realized_vol_30m': vol_context.get('realized_vol_30m'),
            'realized_vol_1h': vol_context.get('realized_vol_1h'),
            'realized_vol_4h': vol_context.get('realized_vol_4h'),
            'vix_level': vol_context.get('vix_level'),
            'vol_regime': vol_context.get('vol_regime'),
        }
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print(f"Volatility context BQ errors: {errors}")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """HTTP endpoint for predictions"""
    try:
        from flask import request
        ticker = request.args.get('ticker', 'SPY')
        predictions = get_predictions(ticker)
        save_to_bigquery(predictions)
        # Strip internal keys from API response
        response = {k: v for k, v in predictions.items() if not k.startswith('_')}
        return jsonify(response)
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def save_to_bigquery(predictions):
    """Save predictions to BigQuery for analysis"""
    from google.cloud import bigquery
    
    # Save interpretation/volatility data to separate tables
    try:
        _save_interpretation_to_bq(predictions)
    except Exception as e:
        print(f"Warning: Failed to save interpretation to BQ: {e}")
    
    client = bigquery.Client()
    table_id = "trading-brains.tft_predictions.tft_predictions_logs"
    
    # Strip internal keys before saving to predictions table
    row = {k: v for k, v in predictions.items() if not k.startswith('_')}
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"BigQuery errors: {errors}")

@app.route('/record_actuals', methods=['GET', 'POST'])
def record_actuals():
    """Record actual prices 60+ minutes after predictions were made"""
    import pandas as pd
    from datetime import datetime, timedelta
    from google.cloud import bigquery
    from flask import request
    
    try:
        ticker = request.args.get('ticker', 'SPY')
        
        # Get the prediction from ~65 minutes ago
        target_time = datetime.utcnow() - timedelta(minutes=65)
        
        # Query BigQuery for the prediction made around that time
        client = bigquery.Client()
        query = """
            SELECT timestamp, last_price_time, last_price,
                   pred_15m, pred_30m, pred_45m, pred_60m
            FROM `trading-brains.tft_predictions.tft_predictions_logs`
            WHERE TIMESTAMP(timestamp) >= TIMESTAMP_SUB(@target_time, INTERVAL 10 MINUTE)
              AND TIMESTAMP(timestamp) <= TIMESTAMP_ADD(@target_time, INTERVAL 10 MINUTE)
              AND ticker = @ticker
            ORDER BY timestamp DESC
            LIMIT 1
        """
        from google.cloud.bigquery import ScalarQueryParameter, QueryJobConfig
        job_config = QueryJobConfig(query_parameters=[
            ScalarQueryParameter("target_time", "TIMESTAMP", target_time),
            ScalarQueryParameter("ticker", "STRING", ticker),
        ])
        result = list(client.query(query, job_config=job_config).result())
        
        if not result:
            return jsonify({'status': 'no_prediction_found', 'target_time': target_time.isoformat()})
        
        prediction_timestamp = result[0].timestamp
        last_price_time = pd.Timestamp(result[0].last_price_time)
        last_price = float(result[0].last_price)
        pred_15m = float(result[0].pred_15m)
        pred_30m = float(result[0].pred_30m)
        pred_45m = float(result[0].pred_45m)
        pred_60m = float(result[0].pred_60m)
        
        # Download minute data to get actual prices
        df = download_market_data(ticker, days=2)
        
        # Find actual prices at +15, +30, +45, +60 from the original prediction's base time
        def get_actual_price(df, base_time, minutes_ahead):
            target = base_time + timedelta(minutes=minutes_ahead)
            # Find closest timestamp
            df['time_diff'] = abs(df['timestamp'] - target)
            closest = df.loc[df['time_diff'].idxmin()]
            return str(closest['timestamp']), float(closest['close'])
        
        actual_15m_time, actual_15m_price = get_actual_price(df, last_price_time, 15)
        actual_30m_time, actual_30m_price = get_actual_price(df, last_price_time, 30)
        actual_45m_time, actual_45m_price = get_actual_price(df, last_price_time, 45)
        actual_60m_time, actual_60m_price = get_actual_price(df, last_price_time, 60)
        
        # Save to BigQuery
        actuals = {
            'prediction_timestamp': prediction_timestamp,
            'ticker': ticker,
            'actual_15m_time': actual_15m_time,
            'actual_15m_price': actual_15m_price,
            'actual_30m_time': actual_30m_time,
            'actual_30m_price': actual_30m_price,
            'actual_45m_time': actual_45m_time,
            'actual_45m_price': actual_45m_price,
            'actual_60m_time': actual_60m_time,
            'actual_60m_price': actual_60m_price,
            'recorded_at': datetime.utcnow().isoformat()
        }
        
        table_id = "trading-brains.tft_predictions.tft_actuals"
        errors = client.insert_rows_json(table_id, [actuals])
        if errors:
            print(f"BigQuery errors: {errors}")
            return jsonify({'error': str(errors)}), 500
        
        # Compute per-prediction error metrics
        def _ae(pred, actual):
            return abs(pred - actual)
        
        def _pct_err(pred, actual):
            return abs(pred - actual) / actual * 100 if actual != 0 else None
        
        def _direction_correct(pred, actual, base):
            """Did the model predict the right direction (up/down) from base price?"""
            return int((pred - base) * (actual - base) > 0) if (pred != base and actual != base) else None
        
        metrics = {
            'prediction_timestamp': prediction_timestamp,
            'ticker': ticker,
            'base_price': last_price,
            'ae_15m': round(_ae(pred_15m, actual_15m_price), 6),
            'ae_30m': round(_ae(pred_30m, actual_30m_price), 6),
            'ae_45m': round(_ae(pred_45m, actual_45m_price), 6),
            'ae_60m': round(_ae(pred_60m, actual_60m_price), 6),
            'pct_error_15m': round(_pct_err(pred_15m, actual_15m_price), 4) if _pct_err(pred_15m, actual_15m_price) is not None else None,
            'pct_error_30m': round(_pct_err(pred_30m, actual_30m_price), 4) if _pct_err(pred_30m, actual_30m_price) is not None else None,
            'pct_error_45m': round(_pct_err(pred_45m, actual_45m_price), 4) if _pct_err(pred_45m, actual_45m_price) is not None else None,
            'pct_error_60m': round(_pct_err(pred_60m, actual_60m_price), 4) if _pct_err(pred_60m, actual_60m_price) is not None else None,
            'direction_correct_15m': _direction_correct(pred_15m, actual_15m_price, last_price),
            'direction_correct_30m': _direction_correct(pred_30m, actual_30m_price, last_price),
            'direction_correct_45m': _direction_correct(pred_45m, actual_45m_price, last_price),
            'direction_correct_60m': _direction_correct(pred_60m, actual_60m_price, last_price),
            'pred_15m': pred_15m,
            'pred_30m': pred_30m,
            'pred_45m': pred_45m,
            'pred_60m': pred_60m,
            'actual_15m': actual_15m_price,
            'actual_30m': actual_30m_price,
            'actual_45m': actual_45m_price,
            'actual_60m': actual_60m_price,
            'recorded_at': datetime.utcnow().isoformat(),
        }
        
        # Write to model_hourly_metrics table (create if needed)
        metrics_table_id = "trading-brains.tft_predictions.model_hourly_metrics"
        metrics_schema = [
            bigquery.SchemaField("prediction_timestamp", "STRING"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("base_price", "FLOAT64"),
            bigquery.SchemaField("ae_15m", "FLOAT64"),
            bigquery.SchemaField("ae_30m", "FLOAT64"),
            bigquery.SchemaField("ae_45m", "FLOAT64"),
            bigquery.SchemaField("ae_60m", "FLOAT64"),
            bigquery.SchemaField("pct_error_15m", "FLOAT64"),
            bigquery.SchemaField("pct_error_30m", "FLOAT64"),
            bigquery.SchemaField("pct_error_45m", "FLOAT64"),
            bigquery.SchemaField("pct_error_60m", "FLOAT64"),
            bigquery.SchemaField("direction_correct_15m", "INT64"),
            bigquery.SchemaField("direction_correct_30m", "INT64"),
            bigquery.SchemaField("direction_correct_45m", "INT64"),
            bigquery.SchemaField("direction_correct_60m", "INT64"),
            bigquery.SchemaField("pred_15m", "FLOAT64"),
            bigquery.SchemaField("pred_30m", "FLOAT64"),
            bigquery.SchemaField("pred_45m", "FLOAT64"),
            bigquery.SchemaField("pred_60m", "FLOAT64"),
            bigquery.SchemaField("actual_15m", "FLOAT64"),
            bigquery.SchemaField("actual_30m", "FLOAT64"),
            bigquery.SchemaField("actual_45m", "FLOAT64"),
            bigquery.SchemaField("actual_60m", "FLOAT64"),
            bigquery.SchemaField("recorded_at", "TIMESTAMP"),
        ]
        try:
            client.get_table(metrics_table_id)
        except Exception:
            table = bigquery.Table(metrics_table_id, schema=metrics_schema)
            client.create_table(table)
            print(f"Created BigQuery table: {metrics_table_id}")
        
        metrics_errors = client.insert_rows_json(metrics_table_id, [metrics])
        if metrics_errors:
            print(f"Metrics BigQuery errors: {metrics_errors}")
        
        return jsonify({**actuals, 'metrics': metrics})
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/test', methods=['GET', 'POST'])
def test_model():
    """Test endpoint - uses historical data to verify model functionality"""
    import numpy as np
    import pandas as pd
    import torch
    from datetime import datetime
    from pytorch_forecasting import TimeSeriesDataSet
    from flask import request
    
    try:
        ticker = request.args.get('ticker', 'SPY')
        
        # Get model (lazy load)
        model, dataset_params, device = get_model(ticker)
        
        # Download historical data
        df = download_market_data(ticker, days=7)
        
        # Calculate VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Feature engineering
        df_features = calculate_all_features(df)
        
        # Prepare for prediction
        df_pred = df_features.copy()
        # Target columns only needed for training; fill to prevent dropna from dropping last 60 rows
        for col in ['target_close_60m', 'target_return_60m']:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].ffill().fillna(0)
        df_pred = df_pred.dropna()
        df_pred = df_pred.reset_index(drop=True)
        df_pred['time_idx'] = range(len(df_pred))
        df_pred['group'] = get_model_group(dataset_params)
        
        # Clean data
        numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
        df_pred[numeric_cols] = df_pred[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_pred[numeric_cols] = df_pred[numeric_cols].ffill().bfill()
        
        # Take recent data
        max_encoder_length = dataset_params.get('max_encoder_length', 60)
        max_prediction_length = dataset_params.get('max_prediction_length', 60)
        min_rows = max_encoder_length + max_prediction_length + 100
        df_recent = df_pred.iloc[-min_rows:].copy()
        df_recent['time_idx'] = range(len(df_recent))
        
        # Update normalizer to current price level and predict
        updated_params = update_normalizer_stats(dataset_params, df_recent['close'])
        prediction_dataset = TimeSeriesDataSet.from_parameters(
            updated_params,
            df_recent,
            predict=True,
        )
        pred_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        # Use model.predict() which automatically applies inverse normalization
        raw_pred = model.predict(pred_dataloader, mode="prediction")
        pred_array = raw_pred.squeeze().cpu().numpy()
        if len(pred_array.shape) == 2:
            pred_array = pred_array[:, pred_array.shape[1] // 2]
        
        last_close = df_recent['close'].iloc[-(max_prediction_length + 1)]
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded and inference completed successfully',
            'test_data': {
                'rows_downloaded': len(df),
                'rows_after_features': len(df_pred),
                'data_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'base_price': float(last_close),
                'pred_15m': float(pred_array[14]),
                'pred_30m': float(pred_array[29]),
                'pred_45m': float(pred_array[44]),
                'pred_60m': float(pred_array[59]),
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Test error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'status': 'failed', 'error': str(e)}), 500


@app.route('/backfill', methods=['GET', 'POST'])
def backfill():
    """Recalculate all historical predictions using the fixed model.predict() pipeline."""
    import numpy as np
    import pandas as pd
    import torch
    import requests
    import time as time_module
    import traceback
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from google.cloud import bigquery
    from pytorch_forecasting import TimeSeriesDataSet

    try:
        ticker = request.args.get('ticker', 'SPY')
        model, dataset_params, device = get_model(ticker)
        max_encoder_length = dataset_params.get('max_encoder_length', 60)
        max_prediction_length = dataset_params.get('max_prediction_length', 60)
        min_rows = max_encoder_length + max_prediction_length + 100

        # 1. Fetch all prediction rows from BigQuery for this ticker
        bq_client = bigquery.Client()
        query = """
            SELECT timestamp, last_price, pred_60m
            FROM `trading-brains.tft_predictions.tft_predictions_logs`
            WHERE ticker = @ticker
            ORDER BY timestamp ASC
        """
        job_cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        )
        predictions_df = bq_client.query(query, job_config=job_cfg).to_dataframe()

        if predictions_df.empty:
            return jsonify({'status': 'no_predictions_found'})

        predictions_df['pred_ts'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df['date'] = predictions_df['pred_ts'].dt.date
        days = sorted(predictions_df['date'].unique())

        results = []
        total_updated = 0
        total_skipped = 0
        data_cache = {}

        api_key = os.environ.get('TWELVEDATA_API_KEY')
        if not api_key:
            return jsonify({'error': 'TWELVEDATA_API_KEY not set'}), 500

        for day in days:
            day_preds = predictions_df[predictions_df['date'] == day]
            day_str = str(day)

            # Download data for this day with lookback
            if day_str not in data_cache:
                try:
                    start_date = (pd.Timestamp(day_str) - timedelta(days=5)).strftime('%Y-%m-%d')
                    end_date = (pd.Timestamp(day_str) + timedelta(days=1)).strftime('%Y-%m-%d')

                    url = "https://api.twelvedata.com/time_series"
                    params = {
                        'symbol': ticker,
                        'interval': '1min',
                        'outputsize': 5000,
                        'start_date': start_date,
                        'end_date': end_date,
                        'apikey': api_key,
                        'timezone': 'America/New_York',
                    }
                    response = requests.get(url, params=params, timeout=30)
                    data = response.json()

                    if 'values' not in data or not data['values']:
                        total_skipped += len(day_preds)
                        continue

                    df = pd.DataFrame(data['values'])
                    df['timestamp'] = pd.to_datetime(df['datetime'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    # VWAP + features
                    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
                    df_feat = calculate_all_features(df)
                    # Target columns only needed for training; fill to prevent dropna from dropping last 60 rows
                    for col in ['target_close_60m', 'target_return_60m']:
                        if col in df_feat.columns:
                            df_feat[col] = df_feat[col].ffill().fillna(0)
                    df_feat = df_feat.dropna().reset_index(drop=True)
                    df_feat['time_idx'] = range(len(df_feat))
                    df_feat['group'] = get_model_group(dataset_params)

                    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
                    df_feat[numeric_cols] = df_feat[numeric_cols].replace([np.inf, -np.inf], np.nan)
                    df_feat[numeric_cols] = df_feat[numeric_cols].ffill().bfill()

                    data_cache[day_str] = df_feat
                    time_module.sleep(8)  # Twelve Data rate limit
                except Exception as e:
                    total_skipped += len(day_preds)
                    results.append({'day': day_str, 'error': f"{type(e).__name__}: {e}", 'traceback': traceback.format_exc()})
                    continue

            df_feat = data_cache[day_str]

            for _, row in day_preds.iterrows():
                pred_ts = row['pred_ts']
                original_timestamp = row['timestamp']

                # Convert UTC prediction time to NY time for proper comparison with market data
                pred_ts_ny = pred_ts.tz_localize('UTC').astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
                mask = df_feat['timestamp'] <= pred_ts_ny
                df_available = df_feat[mask]

                if len(df_available) < min_rows:
                    total_skipped += 1
                    continue

                df_window = df_available.iloc[-min_rows:].copy()
                df_window['time_idx'] = range(len(df_window))

                try:
                    updated_params = update_normalizer_stats(dataset_params, df_window['close'])
                    prediction_dataset = TimeSeriesDataSet.from_parameters(
                        updated_params, df_window, predict=True,
                    )
                    pred_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

                    raw_pred = model.predict(pred_dataloader, mode="prediction")
                    pred_array = raw_pred.squeeze().cpu().numpy()
                    if len(pred_array.shape) == 2:
                        pred_array = pred_array[:, pred_array.shape[1] // 2]

                    last_close = float(df_window['close'].iloc[-(max_prediction_length + 1)])
                    last_ts = df_window['timestamp'].iloc[-(max_prediction_length + 1)]

                    new_values = {
                        'last_price': float(last_close),
                        'last_price_time': str(last_ts),
                        'pred_15m': float(pred_array[14]),
                        'pred_30m': float(pred_array[29]),
                        'pred_45m': float(pred_array[44]),
                        'pred_60m': float(pred_array[59]),
                        'return_15m': float((pred_array[14] - last_close) / last_close * 100),
                        'return_30m': float((pred_array[29] - last_close) / last_close * 100),
                        'return_45m': float((pred_array[44] - last_close) / last_close * 100),
                        'return_60m': float((pred_array[59] - last_close) / last_close * 100),
                    }

                    # Update BigQuery
                    update_query = """
                        UPDATE `trading-brains.tft_predictions.tft_predictions_logs`
                        SET last_price = @last_price,
                            last_price_time = @last_price_time,
                            pred_15m = @pred_15m, pred_30m = @pred_30m,
                            pred_45m = @pred_45m, pred_60m = @pred_60m,
                            return_15m = @return_15m, return_30m = @return_30m,
                            return_45m = @return_45m, return_60m = @return_60m
                        WHERE timestamp = @original_timestamp
                          AND ticker = @ticker
                    """
                    job_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("last_price", "FLOAT64", new_values['last_price']),
                            bigquery.ScalarQueryParameter("last_price_time", "STRING", new_values['last_price_time']),
                            bigquery.ScalarQueryParameter("pred_15m", "FLOAT64", new_values['pred_15m']),
                            bigquery.ScalarQueryParameter("pred_30m", "FLOAT64", new_values['pred_30m']),
                            bigquery.ScalarQueryParameter("pred_45m", "FLOAT64", new_values['pred_45m']),
                            bigquery.ScalarQueryParameter("pred_60m", "FLOAT64", new_values['pred_60m']),
                            bigquery.ScalarQueryParameter("return_15m", "FLOAT64", new_values['return_15m']),
                            bigquery.ScalarQueryParameter("return_30m", "FLOAT64", new_values['return_30m']),
                            bigquery.ScalarQueryParameter("return_45m", "FLOAT64", new_values['return_45m']),
                            bigquery.ScalarQueryParameter("return_60m", "FLOAT64", new_values['return_60m']),
                            bigquery.ScalarQueryParameter("original_timestamp", "STRING", original_timestamp),
                            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                        ]
                    )
                    bq_client.query(update_query, job_config=job_config).result()

                    results.append({
                        'timestamp': original_timestamp,
                        'old_pred_60m': float(row['pred_60m']) if pd.notna(row['pred_60m']) else None,
                        'new_pred_60m': new_values['pred_60m'],
                        'new_last_price': new_values['last_price'],
                    })
                    total_updated += 1

                except Exception as e:
                    total_skipped += 1
                    results.append({'timestamp': original_timestamp, 'error': str(e)})

        return jsonify({
            'status': 'completed',
            'total_updated': total_updated,
            'total_skipped': total_skipped,
            'total_rows': len(predictions_df),
            'details': results,
        })

    except Exception as e:
        import traceback
        print(f"Backfill error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/backfill_actuals', methods=['GET'])
def backfill_actuals():
    """Backfill actual prices for historical predictions so vw_actual_vs_predicted has data."""
    import pandas as pd
    import requests
    import time as time_module
    import traceback
    from datetime import datetime, timedelta
    from google.cloud import bigquery

    try:
        ticker = request.args.get('ticker', 'GOOG')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        bq_client = bigquery.Client()
        api_key = os.environ.get('TWELVEDATA_API_KEY')
        if not api_key:
            return jsonify({'error': 'TWELVEDATA_API_KEY not set'}), 500

        # 1. Fetch all predictions for this ticker that don't yet have actuals
        query = """
            SELECT p.timestamp AS prediction_timestamp, p.last_price_time, p.last_price,
                   p.pred_15m, p.pred_30m, p.pred_45m, p.pred_60m
            FROM `trading-brains.tft_predictions.tft_predictions_logs` p
            LEFT JOIN `trading-brains.tft_predictions.tft_actuals` a
              ON p.timestamp = a.prediction_timestamp AND p.ticker = a.ticker
            WHERE p.ticker = @ticker
              AND a.prediction_timestamp IS NULL
        """
        params = [bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        if start_date:
            query += " AND DATE(TIMESTAMP(p.timestamp)) >= @start_date"
            params.append(bigquery.ScalarQueryParameter("start_date", "STRING", start_date))
        if end_date:
            query += " AND DATE(TIMESTAMP(p.timestamp)) <= @end_date"
            params.append(bigquery.ScalarQueryParameter("end_date", "STRING", end_date))
        query += " ORDER BY p.timestamp ASC"

        job_cfg = bigquery.QueryJobConfig(query_parameters=params)
        preds_df = bq_client.query(query, job_config=job_cfg).to_dataframe()

        if preds_df.empty:
            return jsonify({'status': 'no_predictions_needing_actuals', 'ticker': ticker})

        preds_df['last_price_time_parsed'] = pd.to_datetime(preds_df['last_price_time'])
        preds_df['date'] = preds_df['last_price_time_parsed'].dt.date
        days = sorted(preds_df['date'].unique())

        results = []
        total_inserted = 0
        total_skipped = 0
        data_cache = {}

        for day in days:
            day_str = str(day)
            day_preds = preds_df[preds_df['date'] == day]

            # Fetch 1-min data for this day (with buffer for +60min lookahead)
            if day_str not in data_cache:
                try:
                    fetch_start = (pd.Timestamp(day_str) - timedelta(days=1)).strftime('%Y-%m-%d')
                    fetch_end = (pd.Timestamp(day_str) + timedelta(days=2)).strftime('%Y-%m-%d')

                    url = "https://api.twelvedata.com/time_series"
                    params_api = {
                        'symbol': ticker,
                        'interval': '1min',
                        'outputsize': 5000,
                        'start_date': fetch_start,
                        'end_date': fetch_end,
                        'apikey': api_key,
                        'timezone': 'America/New_York',
                    }
                    response = requests.get(url, params=params_api, timeout=30)
                    data = response.json()

                    if 'values' not in data or not data['values']:
                        total_skipped += len(day_preds)
                        results.append({'day': day_str, 'error': 'No data from Twelve Data'})
                        time_module.sleep(8)
                        continue

                    df = pd.DataFrame(data['values'])
                    df['timestamp'] = pd.to_datetime(df['datetime'])
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    df = df[['timestamp', 'close']].sort_values('timestamp').reset_index(drop=True)
                    data_cache[day_str] = df
                    time_module.sleep(8)
                except Exception as e:
                    total_skipped += len(day_preds)
                    results.append({'day': day_str, 'error': f"{type(e).__name__}: {e}"})
                    continue

            df = data_cache[day_str]
            actuals_batch = []
            metrics_batch = []

            for _, row in day_preds.iterrows():
                base_time = row['last_price_time_parsed']
                pred_ts = row['prediction_timestamp']
                base_price = float(row['last_price'])

                def get_actual(minutes_ahead):
                    target = base_time + timedelta(minutes=minutes_ahead)
                    diffs = abs(df['timestamp'] - target)
                    if diffs.min() > timedelta(minutes=5):
                        return None, None
                    idx = diffs.idxmin()
                    return str(df.loc[idx, 'timestamp']), float(df.loc[idx, 'close'])

                a15_time, a15_price = get_actual(15)
                a30_time, a30_price = get_actual(30)
                a45_time, a45_price = get_actual(45)
                a60_time, a60_price = get_actual(60)

                if a60_price is None:
                    total_skipped += 1
                    continue

                actual_row = {
                    'prediction_timestamp': pred_ts,
                    'ticker': ticker,
                    'actual_15m_time': a15_time,
                    'actual_15m_price': a15_price,
                    'actual_30m_time': a30_time,
                    'actual_30m_price': a30_price,
                    'actual_45m_time': a45_time,
                    'actual_45m_price': a45_price,
                    'actual_60m_time': a60_time,
                    'actual_60m_price': a60_price,
                    'recorded_at': datetime.utcnow().isoformat()
                }
                actuals_batch.append(actual_row)

                # Metrics
                def _ae(p, a):
                    return round(abs(p - a), 6) if a else None
                def _pct(p, a):
                    return round(abs(p - a) / a * 100, 4) if a and a != 0 else None
                def _dir(p, a, b):
                    return int((p - b) * (a - b) > 0) if (a and p != b and a != b) else None

                metrics_batch.append({
                    'prediction_timestamp': pred_ts,
                    'ticker': ticker,
                    'base_price': base_price,
                    'ae_15m': _ae(float(row['pred_15m']), a15_price),
                    'ae_30m': _ae(float(row['pred_30m']), a30_price),
                    'ae_45m': _ae(float(row['pred_45m']), a45_price),
                    'ae_60m': _ae(float(row['pred_60m']), a60_price),
                    'pct_error_15m': _pct(float(row['pred_15m']), a15_price),
                    'pct_error_30m': _pct(float(row['pred_30m']), a30_price),
                    'pct_error_45m': _pct(float(row['pred_45m']), a45_price),
                    'pct_error_60m': _pct(float(row['pred_60m']), a60_price),
                    'direction_correct_15m': _dir(float(row['pred_15m']), a15_price, base_price),
                    'direction_correct_30m': _dir(float(row['pred_30m']), a30_price, base_price),
                    'direction_correct_45m': _dir(float(row['pred_45m']), a45_price, base_price),
                    'direction_correct_60m': _dir(float(row['pred_60m']), a60_price, base_price),
                    'pred_15m': float(row['pred_15m']),
                    'pred_30m': float(row['pred_30m']),
                    'pred_45m': float(row['pred_45m']),
                    'pred_60m': float(row['pred_60m']),
                    'actual_15m': a15_price,
                    'actual_30m': a30_price,
                    'actual_45m': a45_price,
                    'actual_60m': a60_price,
                    'recorded_at': datetime.utcnow().isoformat(),
                })

            # Batch insert
            if actuals_batch:
                err1 = bq_client.insert_rows_json("trading-brains.tft_predictions.tft_actuals", actuals_batch)
                err2 = bq_client.insert_rows_json("trading-brains.tft_predictions.model_hourly_metrics", metrics_batch)
                if err1 or err2:
                    results.append({'day': day_str, 'error': str(err1 or err2)})
                    total_skipped += len(actuals_batch)
                else:
                    total_inserted += len(actuals_batch)
                    results.append({'day': day_str, 'inserted': len(actuals_batch)})

        return jsonify({
            'status': 'completed',
            'ticker': ticker,
            'total_predictions_without_actuals': len(preds_df),
            'total_inserted': total_inserted,
            'total_skipped': total_skipped,
            'details': results,
        })

    except Exception as e:
        import traceback
        print(f"Backfill actuals error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/backfill_historical', methods=['GET'])
def backfill_historical():
    """Generate historical predictions for dates where no predictions exist and insert into BigQuery."""
    import numpy as np
    import pandas as pd
    import torch
    import requests
    import time as time_module
    import traceback
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from google.cloud import bigquery
    from pytorch_forecasting import TimeSeriesDataSet

    try:
        ticker = request.args.get('ticker', 'GOOG')
        start_date = request.args.get('start_date', '2026-03-19')
        end_date = request.args.get('end_date', '2026-04-21')

        model, dataset_params, device = get_model(ticker)
        max_encoder_length = dataset_params.get('max_encoder_length', 60)
        max_prediction_length = dataset_params.get('max_prediction_length', 60)
        min_rows = max_encoder_length + max_prediction_length + 100

        group_value = get_model_group(dataset_params)

        bq_client = bigquery.Client()
        api_key = os.environ.get('TWELVEDATA_API_KEY')
        if not api_key:
            return jsonify({'error': 'TWELVEDATA_API_KEY not set'}), 500

        # Trading days in range
        all_dates = pd.bdate_range(start=start_date, end=end_date)

        # Prediction hours in ET (matching the hourly Cloud Scheduler pattern)
        prediction_hours_et = [10, 11, 12, 13, 14, 15, 16]

        results = []
        total_inserted = 0
        total_skipped = 0
        data_cache = {}

        for day in all_dates:
            day_str = day.strftime('%Y-%m-%d')

            if day_str not in data_cache:
                try:
                    fetch_start = (day - timedelta(days=5)).strftime('%Y-%m-%d')
                    fetch_end = (day + timedelta(days=1)).strftime('%Y-%m-%d')

                    url = "https://api.twelvedata.com/time_series"
                    params = {
                        'symbol': ticker,
                        'interval': '1min',
                        'outputsize': 5000,
                        'start_date': fetch_start,
                        'end_date': fetch_end,
                        'apikey': api_key,
                        'timezone': 'America/New_York',
                    }
                    response = requests.get(url, params=params, timeout=30)
                    data = response.json()

                    if 'values' not in data or not data['values']:
                        total_skipped += len(prediction_hours_et)
                        results.append({'day': day_str, 'error': 'No data from Twelve Data'})
                        time_module.sleep(8)
                        continue

                    df = pd.DataFrame(data['values'])
                    df['timestamp'] = pd.to_datetime(df['datetime'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
                    df_feat = calculate_all_features(df)
                    for col in ['target_close_60m', 'target_return_60m']:
                        if col in df_feat.columns:
                            df_feat[col] = df_feat[col].ffill().fillna(0)
                    df_feat = df_feat.dropna().reset_index(drop=True)
                    df_feat['group'] = group_value

                    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
                    df_feat[numeric_cols] = df_feat[numeric_cols].replace([np.inf, -np.inf], np.nan)
                    df_feat[numeric_cols] = df_feat[numeric_cols].ffill().bfill()

                    data_cache[day_str] = df_feat
                    time_module.sleep(8)
                except Exception as e:
                    total_skipped += len(prediction_hours_et)
                    results.append({'day': day_str, 'error': f"{type(e).__name__}: {e}"})
                    continue

            df_feat = data_cache[day_str]
            day_rows = []

            for hour_et in prediction_hours_et:
                pred_time_et = pd.Timestamp(f"{day_str} {hour_et:02d}:00:00")
                pred_time_utc = pred_time_et.tz_localize('America/New_York').astimezone(ZoneInfo('UTC'))

                mask = df_feat['timestamp'] <= pred_time_et
                df_available = df_feat[mask]

                if len(df_available) < min_rows:
                    total_skipped += 1
                    continue

                df_window = df_available.iloc[-min_rows:].copy()
                df_window['time_idx'] = range(len(df_window))

                try:
                    updated_params = update_normalizer_stats(dataset_params, df_window['close'])
                    prediction_dataset = TimeSeriesDataSet.from_parameters(
                        updated_params, df_window, predict=True,
                    )
                    pred_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

                    raw_pred = model.predict(pred_dataloader, mode="prediction")
                    pred_array = raw_pred.squeeze().cpu().numpy()
                    if len(pred_array.shape) == 2:
                        pred_array = pred_array[:, pred_array.shape[1] // 2]

                    last_close = float(df_window['close'].iloc[-(max_prediction_length + 1)])
                    last_ts = df_window['timestamp'].iloc[-(max_prediction_length + 1)]

                    row_data = {
                        'timestamp': pred_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                        'ticker': ticker,
                        'last_price': float(last_close),
                        'last_price_time': str(last_ts),
                        'pred_15m': float(pred_array[14]),
                        'pred_30m': float(pred_array[29]),
                        'pred_45m': float(pred_array[44]),
                        'pred_60m': float(pred_array[59]),
                        'return_15m': float((pred_array[14] - last_close) / last_close * 100),
                        'return_30m': float((pred_array[29] - last_close) / last_close * 100),
                        'return_45m': float((pred_array[44] - last_close) / last_close * 100),
                        'return_60m': float((pred_array[59] - last_close) / last_close * 100),
                    }
                    day_rows.append(row_data)

                except Exception as e:
                    total_skipped += 1
                    results.append({'day': day_str, 'hour_et': hour_et, 'error': str(e)})

            # Batch insert all predictions for this day
            if day_rows:
                table_id = "trading-brains.tft_predictions.tft_predictions_logs"
                errors = bq_client.insert_rows_json(table_id, day_rows)
                if errors:
                    results.append({'day': day_str, 'error': str(errors)})
                    total_skipped += len(day_rows)
                else:
                    total_inserted += len(day_rows)
                    results.append({'day': day_str, 'inserted': len(day_rows),
                                    'sample_pred_60m': day_rows[-1]['pred_60m'],
                                    'sample_last_price': day_rows[-1]['last_price']})

        return jsonify({
            'status': 'completed',
            'ticker': ticker,
            'date_range': f"{start_date} to {end_date}",
            'total_inserted': total_inserted,
            'total_skipped': total_skipped,
            'details': results,
        })

    except Exception as e:
        import traceback
        print(f"Backfill historical error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)