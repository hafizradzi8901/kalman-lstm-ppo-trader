"""
Data pipeline: fetches OHLCV via yfinance, engineers features, and returns
numpy arrays ready for TradingEnv.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from kalman_lstm import FeatureEngineer


def fetch_ohlcv(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # yfinance may return MultiIndex columns for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def build_features(
    df: pd.DataFrame,
    fe: FeatureEngineer | None = None,
    fracdiff_d: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Engineer features from OHLCV dataframe.

    Returns:
        features: (n_steps, n_features) float32 array
        prices:   (n_steps,) float32 array (close prices for PnL)
        dates:    DatetimeIndex for reference
    """
    if fe is None:
        fe = FeatureEngineer(d_value=fracdiff_d)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Log returns ---
    log_ret = np.log(close / close.shift(1))

    # --- Fractional differentiation of close ---
    fracdiff = fe.apply_fractional_differentiation(close)

    # --- Volatility (EWMV) ---
    vol = fe.compute_volatility_clusters(log_ret.dropna())
    vol = vol.reindex(close.index)

    # --- Normalized volume (z-score over rolling window) ---
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    vol_zscore = (volume - vol_mean) / (vol_std + 1e-8)

    # --- High-Low range (normalized by close) ---
    hl_range = (high - low) / close

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi_normalized = (rsi - 50) / 50  # center around 0, scale to [-1, 1]

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal
    # Normalize by close price to make it scale-invariant
    macd_norm = macd_hist / close

    # --- Bollinger Band position ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pos = (close - bb_mid) / (2 * bb_std + 1e-8)  # [-1, 1] roughly

    # Combine all features
    feature_df = pd.DataFrame(
        {
            "log_return": log_ret,
            "fracdiff": fracdiff,
            "volatility": vol,
            "volume_zscore": vol_zscore,
            "hl_range": hl_range,
            "rsi": rsi_normalized,
            "macd": macd_norm,
            "bb_position": bb_pos,
        },
        index=close.index,
    )

    # Drop NaN rows (from rolling windows / fracdiff warmup)
    valid_mask = feature_df.notna().all(axis=1)
    feature_df = feature_df[valid_mask]
    prices_aligned = close[valid_mask]

    # Final NaN safety — fill any strays with 0
    feature_df = feature_df.fillna(0.0)

    features = feature_df.values.astype(np.float32)
    prices = prices_aligned.values.astype(np.float32)
    dates = feature_df.index

    return features, prices, dates


def train_test_split(
    features: np.ndarray,
    prices: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split — no shuffling (time series!)."""
    n = len(features)
    split = int(n * train_ratio)
    return (
        features[:split],
        prices[:split],
        features[split:],
        prices[split:],
    )


if __name__ == "__main__":
    print("Fetching SPY daily data (2 years)...")
    df = fetch_ohlcv("SPY", period="2y", interval="1d")
    print(f"  Raw OHLCV: {len(df)} rows, columns: {list(df.columns)}")

    features, prices, dates = build_features(df)
    print(f"  Features: {features.shape} (steps x features)")
    print(f"  Prices: {prices.shape}")
    print(f"  Date range: {dates[0]} → {dates[-1]}")
    print(f"  Feature stats:\n{pd.DataFrame(features).describe().to_string()}")

    train_f, train_p, test_f, test_p = train_test_split(features, prices)
    print(f"\n  Train: {len(train_f)} steps, Test: {len(test_f)} steps")
