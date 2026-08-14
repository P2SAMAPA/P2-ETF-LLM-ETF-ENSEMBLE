"""
data_manager.py  —  Data loading and validation for LLM ETF Ensemble
"""

import os
import logging
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def load_master_data(token: str = None) -> tuple:
    """
    Load master data from HuggingFace.
    
    Returns:
        prices_df: DataFrame with ETF prices
        macro_df: DataFrame with macro indicators
    """
    try:
        logger.info("Downloading master parquet from P2SAMAPA/fi-etf-macro-signal-master-data …")
        
        # Download the parquet file
        local_path = hf_hub_download(
            repo_id="P2SAMAPA/fi-etf-macro-signal-master-data",
            filename="master_data.parquet",
            token=token,
            repo_type="dataset"
        )
        
        logger.info(f"  → found at '{local_path}'")
        
        # Load parquet
        df = pd.read_parquet(local_path)
        logger.info(f"Raw parquet: {df.shape[0]} rows × {df.shape[1]} cols")
        
        # Separate ETFs and macro
        etf_cols = [c for c in df.columns if c not in ['date', 'SPY'] and not c.endswith('_macro')]
        macro_cols = [c for c in df.columns if c.endswith('_macro')]
        
        # If we have SPY, keep it for performance comparison
        if 'SPY' in df.columns:
            etf_cols = ['date', 'SPY'] + etf_cols
        
        # Handle missing values
        df = df.dropna(subset=['date'] + macro_cols)
        logger.info(f"Dropped rows with NaN in core macro cols.")
        
        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Split into prices and macro
        price_cols = [c for c in df.columns if c not in macro_cols]
        prices_df = df[price_cols].copy()
        macro_df = df[macro_cols].copy()
        
        # Forward fill macro data
        macro_df = macro_df.fillna(method='ffill')
        
        logger.info(f"Dataset ready: {prices_df.shape[0]} rows | {len(price_cols)} ETFs | {len(macro_cols)} macro cols")
        
        return prices_df, macro_df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def validate_data(prices_df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
    """Validate loaded data."""
    if prices_df.empty:
        raise ValueError("Prices DataFrame is empty")
    
    if macro_df.empty:
        raise ValueError("Macro DataFrame is empty")
    
    if prices_df.shape[0] < 100:
        raise ValueError(f"Only {prices_df.shape[0]} days of data - insufficient")
    
    logger.info(f"✅ Data validation passed: {prices_df.shape[0]} days, {prices_df.shape[1]} ETFs")


def prepare_data_summary(prices_df: pd.DataFrame, macro_df: pd.DataFrame, 
                         tickers: List[str]) -> Dict:
    """Prepare a summary of recent data for LLM analysis."""
    
    summary = {}
    
    # ETF returns
    returns = prices_df[tickers].pct_change()
    
    # Recent performance
    summary["last_price"] = prices_df[tickers].iloc[-1].to_dict()
    summary["ytd_return"] = (prices_df[tickers].iloc[-1] / prices_df[tickers].iloc[0] - 1).to_dict()
    
    # Short-term momentum (20-day)
    momentum_20 = (prices_df[tickers].iloc[-1] / prices_df[tickers].iloc[-20] - 1).to_dict()
    summary["momentum_20d"] = momentum_20
    
    # Medium-term momentum (60-day)
    momentum_60 = (prices_df[tickers].iloc[-1] / prices_df[tickers].iloc[-60] - 1).to_dict()
    summary["momentum_60d"] = momentum_60
    
    # Volatility (30-day)
    vol_30 = returns[tickers].tail(30).std().to_dict()
    summary["volatility_30d"] = vol_30
    
    # Macro data
    macro_latest = macro_df.iloc[-1].to_dict()
    summary["macro"] = macro_latest
    
    # Fed rate (if available)
    if "FEDFUNDS_macro" in macro_df.columns:
        fed_rate = macro_df["FEDFUNDS_macro"].iloc[-1]
        summary["fed_rate"] = fed_rate
        summary["fed_rate_change_12m"] = fed_rate - macro_df["FEDFUNDS_macro"].iloc[-252]
    
    # Inflation (if available)
    if "CPI_macro" in macro_df.columns:
        summary["cpi"] = macro_df["CPI_macro"].iloc[-1]
        summary["cpi_change_12m"] = (macro_df["CPI_macro"].iloc[-1] / macro_df["CPI_macro"].iloc[-252] - 1)
    
    # VIX (if available)
    if "VIX_macro" in macro_df.columns:
        summary["vix"] = macro_df["VIX_macro"].iloc[-1]
    
    return summary
