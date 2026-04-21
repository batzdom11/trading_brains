"""
Feature engineering for TFT model training.
Identical to calculate_all_features() in main.py / training notebook.
"""

import numpy as np
import pandas as pd


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 76 features required for the TFT model.
    Input df must have: timestamp, open, high, low, close, volume
    """
    df = df.copy()

    # 1. RETURNS (5 features)
    df['returns_1m'] = df['close'].pct_change(1)
    df['returns_5m'] = df['close'].pct_change(5)
    df['returns_15m'] = df['close'].pct_change(15)
    df['returns_30m'] = df['close'].pct_change(30)
    df['returns_60m'] = df['close'].pct_change(60)

    # 2. PRICE RATIOS (6 features)
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-10)
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-10)

    # 3. SMA FEATURES (11 features)
    for period in [5, 10, 20, 30]:
        sma = df['close'].rolling(window=period).mean()
        df[f'sma_{period}_slope'] = sma.pct_change()
        df[f'close_to_sma_{period}'] = (df['close'] - sma) / sma

    sma_60 = df['close'].rolling(window=60).mean()
    df['close_to_sma_60'] = (df['close'] - sma_60) / sma_60

    sma_120 = df['close'].rolling(window=120).mean()
    df['sma_120_slope'] = sma_120.pct_change()
    df['close_to_sma_120'] = (df['close'] - sma_120) / sma_120

    # 4. MACD (2 features)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    signal_line = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - signal_line

    # 5. VOLATILITY (4 features)
    df['volatility_5'] = df['returns_1m'].rolling(window=5).std()
    df['volatility_10'] = df['returns_1m'].rolling(window=10).std()
    df['volatility_20'] = df['returns_1m'].rolling(window=20).std()
    df['volatility_60'] = df['returns_1m'].rolling(window=60).std()

    # 6. ATR (2 features)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_60'] = tr.rolling(window=60).mean()

    # 7. BOLLINGER BANDS (4 features)
    for period in [20, 60]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_std_{period}'] = std
        df[f'bb_position_{period}'] = (df['close'] - sma) / (2 * std + 1e-10)

    # 8. RSI (3 features)
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = calc_rsi(df['close'], 14)
    df['rsi_20'] = calc_rsi(df['close'], 20)
    df['rsi_60'] = calc_rsi(df['close'], 60)

    # 9. STOCHASTIC (3 features)
    for period in [14, 60]:
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    df['stoch_d_14'] = df['stoch_k_14'].rolling(window=3).mean()

    # 10. RATE OF CHANGE (2 features)
    df['roc_10'] = df['close'].pct_change(10) * 100
    df['roc_20'] = df['close'].pct_change(20) * 100

    # 11. VOLUME INDICATORS (12 features)
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_change_5m'] = df['volume'].pct_change(5)

    for period in [5, 10, 20, 60]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-10)

    # 12. OBV, VPT, MFI (5 features)
    obv = np.where(df['close'] > df['close'].shift(), df['volume'],
                   np.where(df['close'] < df['close'].shift(), -df['volume'], 0))
    df['obv'] = np.cumsum(obv)
    obv_sma = pd.Series(df['obv']).rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / (obv_sma + 1e-10)

    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()

    def calc_mfi(df, period):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        return mfi

    df['mfi_14'] = calc_mfi(df, 14)
    df['mfi_60'] = calc_mfi(df, 60)

    # 13. VWAP RATIOS (2 features)
    vwap_20 = (df['volume'] * df['close']).rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
    vwap_60 = (df['volume'] * df['close']).rolling(window=60).sum() / (df['volume'].rolling(window=60).sum() + 1e-10)
    df['close_to_vwap_20'] = (df['close'] - vwap_20) / vwap_20
    df['close_to_vwap_60'] = (df['close'] - vwap_60) / vwap_60

    # 14. TIME FEATURES (10 features)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 15. VOLUME LAGS (5 features)
    df['volume_lag_1'] = df['volume'].shift(1)
    df['volume_lag_5'] = df['volume'].shift(5)
    df['volume_lag_15'] = df['volume'].shift(15)
    df['volume_lag_30'] = df['volume'].shift(30)
    df['volume_lag_60'] = df['volume'].shift(60)

    # 16. TARGET VARIABLES (2 features)
    df['target_close_60m'] = df['close'].shift(-60)
    df['target_return_60m'] = df['close'].pct_change(60).shift(-60)

    return df
