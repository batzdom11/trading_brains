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

from flask import Flask, jsonify

app = Flask(__name__)

# Global variables for lazy loading
GCS_BUCKET = "tft-for-trading-brains"

# Per-ticker model cache: {ticker: (model, dataset_params, device)}
_model_cache = {}

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
            outputsize = min(5000, days * 390)  # ~390 minutes per trading day
            
            print(f"Attempt {attempt + 1}: Downloading {ticker} data from Twelve Data...")
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': ticker,
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
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
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
    
    # Calculate VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
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
    df_pred['group'] = 'SPY'
    
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
    
    # 5. Extract predictions at 15, 30, 45, 60 minutes
    # Last encoder position = end of lookback window (before prediction horizon)
    last_close = df_recent['close'].iloc[-(max_prediction_length + 1)]
    last_timestamp = df_recent['timestamp'].iloc[-(max_prediction_length + 1)]
    
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
    }
    
    return predictions

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """HTTP endpoint for predictions"""
    try:
        from flask import request
        ticker = request.args.get('ticker', 'SPY')
        predictions = get_predictions(ticker)
        save_to_bigquery(predictions)
        return jsonify(predictions)
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def save_to_bigquery(predictions):
    """Save predictions to BigQuery for analysis"""
    from google.cloud import bigquery
    
    client = bigquery.Client()
    table_id = "trading-brains.tft_predictions.tft_predictions_logs"
    
    rows = [predictions]
    errors = client.insert_rows_json(table_id, rows)
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
        df_pred['group'] = 'SPY'
        
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
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from google.cloud import bigquery
    from pytorch_forecasting import TimeSeriesDataSet

    try:
        model, dataset_params, device = get_model()
        max_encoder_length = dataset_params.get('max_encoder_length', 60)
        max_prediction_length = dataset_params.get('max_prediction_length', 60)
        min_rows = max_encoder_length + max_prediction_length + 100

        # 1. Fetch all prediction rows from BigQuery
        bq_client = bigquery.Client()
        query = """
            SELECT timestamp, last_price, pred_60m
            FROM `trading-brains.tft_predictions.tft_predictions_logs`
            ORDER BY timestamp ASC
        """
        predictions_df = bq_client.query(query).to_dataframe()

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
                        'symbol': 'SPY',
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
                    df_feat['group'] = 'SPY'

                    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
                    df_feat[numeric_cols] = df_feat[numeric_cols].replace([np.inf, -np.inf], np.nan)
                    df_feat[numeric_cols] = df_feat[numeric_cols].ffill().bfill()

                    data_cache[day_str] = df_feat
                    time_module.sleep(8)  # Twelve Data rate limit
                except Exception as e:
                    total_skipped += len(day_preds)
                    results.append({'day': day_str, 'error': str(e)})
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)