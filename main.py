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
GCS_MODEL_PATH = "tft_checkpoint_latest.ckpt"
LOCAL_MODEL_PATH = "/tmp/tft_checkpoint_latest.ckpt"

model = None
dataset_params = None
device = None

@app.route('/')
def health():
    """Health check endpoint - must respond fast"""
    return jsonify({'status': 'healthy'})

def get_model():
    """Lazy load model on first request"""
    global model, dataset_params, device
    
    if model is None:
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
        
        print(f"Downloading model from gs://{GCS_BUCKET}/{GCS_MODEL_PATH}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_PATH)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model downloaded successfully!")
        
        model = TemporalFusionTransformer.load_from_checkpoint(LOCAL_MODEL_PATH, map_location='cpu')
        model.eval()
        dataset_params = model.dataset_parameters
        device = torch.device('cpu')
        print(f"Model loaded on device: {device}")
    
    return model, dataset_params, device

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

def get_predictions():
    """Download data, engineer features, run model, return predictions"""
    import numpy as np
    import pandas as pd
    import torch
    import yfinance as yf
    from datetime import datetime
    from pytorch_forecasting import TimeSeriesDataSet
    import time
    
    # Get model (lazy load)
    model, dataset_params, device = get_model()
    
    # 1. Download real-time data with retries
    max_retries = 3
    df = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            periods = ['5d', '7d', '10d']
            period = periods[attempt % len(periods)]
            
            print(f"Attempt {attempt + 1}: Downloading SPY data (period={period})...")
            df = yf.download('SPY', period=period, interval='1m', progress=False)
            
            if not df.empty:
                print(f"Successfully downloaded {len(df)} rows")
                break
            else:
                print(f"Empty dataframe on attempt {attempt + 1}")
                
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    if df is None or df.empty:
        raise ValueError(f"No market data available after {max_retries} attempts. Last error: {last_error}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    df.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 
                       'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # 2. Feature engineering
    df_features = calculate_all_features(df)
    
    # 3. Prepare for prediction
    df_pred = df_features.copy()
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
    
    # 4. Create dataset and predict
    prediction_dataset = TimeSeriesDataSet.from_parameters(
        dataset_params,
        df_recent,
        predict=True,
    )
    pred_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    with torch.no_grad():
        for x_batch, y_batch in pred_dataloader:
            pass
        x_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_batch.items()}
        output = model(x_batch)
    
    raw_pred = output.prediction.squeeze().cpu().numpy()
    if len(raw_pred.shape) == 2:
        pred_array = raw_pred[:, raw_pred.shape[1] // 2]
    else:
        pred_array = raw_pred
    
    # 5. Extract predictions at 15, 30, 45, 60 minutes
    last_close = df_recent['close'].iloc[-61]
    last_timestamp = df_recent['timestamp'].iloc[-61]
    
    predictions = {
        'timestamp': datetime.utcnow().isoformat(),
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
        predictions = get_predictions()
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
    import yfinance as yf
    from datetime import datetime, timedelta
    from google.cloud import bigquery
    
    try:
        # Get the prediction from ~65 minutes ago
        target_time = datetime.utcnow() - timedelta(minutes=65)
        
        # Query BigQuery for the prediction made around that time
        client = bigquery.Client()
        query = f"""
            SELECT timestamp, last_price_time
            FROM `trading-brains.tft_predictions.tft_predictions_logs`
            WHERE TIMESTAMP(timestamp) >= TIMESTAMP_SUB(TIMESTAMP('{target_time.isoformat()}'), INTERVAL 10 MINUTE)
              AND TIMESTAMP(timestamp) <= TIMESTAMP_ADD(TIMESTAMP('{target_time.isoformat()}'), INTERVAL 10 MINUTE)
            ORDER BY timestamp DESC
            LIMIT 1
        """
        result = list(client.query(query).result())
        
        if not result:
            return jsonify({'status': 'no_prediction_found', 'target_time': target_time.isoformat()})
        
        prediction_timestamp = result[0].timestamp
        last_price_time = pd.Timestamp(result[0].last_price_time)
        
        # Download minute data to get actual prices
        df = yf.download('SPY', period='2d', interval='1m', progress=False)
        if df.empty:
            raise ValueError("No market data available")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        df.rename(columns={'Datetime': 'timestamp', 'Close': 'close'}, inplace=True)
        
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
        
        return jsonify(actuals)
        
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
    import yfinance as yf
    from datetime import datetime
    from pytorch_forecasting import TimeSeriesDataSet
    import time
    
    try:
        # Get model (lazy load)
        model, dataset_params, device = get_model()
        
        # Download historical data with retries (same approach as get_predictions)
        max_retries = 3
        df = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                periods = ['5d', '7d', '10d']
                period = periods[attempt % len(periods)]
                
                print(f"Test attempt {attempt + 1}: Downloading SPY data (period={period})...")
                df = yf.download('SPY', period=period, interval='1m', progress=False)
                
                if not df.empty:
                    print(f"Successfully downloaded {len(df)} rows")
                    break
                else:
                    print(f"Empty dataframe on attempt {attempt + 1}")
                    
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2)
        
        if df is None or df.empty:
            return jsonify({
                'error': f'Could not download historical data after {max_retries} attempts',
                'last_error': str(last_error),
                'status': 'failed'
            }), 500
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        df.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 
                           'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Feature engineering
        df_features = calculate_all_features(df)
        
        # Prepare for prediction
        df_pred = df_features.copy()
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
        
        # Create dataset and predict
        prediction_dataset = TimeSeriesDataSet.from_parameters(
            dataset_params,
            df_recent,
            predict=True,
        )
        pred_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        with torch.no_grad():
            for x_batch, y_batch in pred_dataloader:
                pass
            x_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_batch.items()}
            output = model(x_batch)
        
        raw_pred = output.prediction.squeeze().cpu().numpy()
        if len(raw_pred.shape) == 2:
            pred_array = raw_pred[:, raw_pred.shape[1] // 2]
        else:
            pred_array = raw_pred
        
        last_close = df_recent['close'].iloc[-61]
        
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
        

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)