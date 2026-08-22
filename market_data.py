"""
market_data.py  —  Real price context for each ETF
=====================================================

The previous version asked every LLM to "pick the best ETF" from a bare
list of tickers with no numbers attached — so every "analysis" was really
just the model's training-data priors about what GLD or QQQ "usually" do.
This module pulls actual trailing price behaviour via yfinance so each
model gets real, current numbers to reason over.

This is deliberately simple (momentum + volatility from daily closes) —
not a forecast, just grounding. The LLM still does the judgment call.
"""

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_universe_snapshot(tickers: List[str], lookback: str = "6mo") -> Dict[str, Dict]:
    """
    Return {ticker: {last_price, return_1m, return_3m, return_6m,
    annualized_volatility_pct}} using trailing daily closes.

    Missing tickers (delisted, bad data, etc.) are simply omitted — callers
    should treat an absent ticker as "no market data available" rather than
    erroring out.
    """
    snapshot: Dict[str, Dict] = {}
    if not tickers:
        return snapshot

    try:
        raw = yf.download(
            tickers,
            period=lookback,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.warning(f"Market data download failed for {tickers}: {e}")
        return snapshot

    if raw is None or raw.empty:
        logger.warning("Market data download returned no rows")
        return snapshot

    multi_ticker = len(tickers) > 1

    for ticker in tickers:
        try:
            if multi_ticker:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                closes = raw[ticker]["Close"].dropna()
            else:
                closes = raw["Close"].dropna()

            if closes.empty:
                continue

            last = float(closes.iloc[-1])

            def pct_change(trading_days: int):
                if len(closes) <= trading_days:
                    return None
                return round((last / float(closes.iloc[-trading_days - 1]) - 1) * 100, 2)

            daily_returns = closes.pct_change().dropna()
            volatility = (
                round(float(daily_returns.std()) * (252 ** 0.5) * 100, 2)
                if len(daily_returns) > 5
                else None
            )

            snapshot[ticker] = {
                "last_price": round(last, 2),
                "return_1m": pct_change(21),
                "return_3m": pct_change(63),
                "return_6m": pct_change(126),
                "annualized_volatility_pct": volatility,
            }
        except Exception as e:
            logger.warning(f"  No usable data for {ticker}: {e}")

    logger.info(f"Market data: {len(snapshot)}/{len(tickers)} tickers resolved")
    return snapshot


def format_snapshot_for_prompt(snapshot: Dict[str, Dict], tickers: List[str]) -> str:
    """Render the snapshot as a compact table the LLM can read easily."""
    lines = ["Ticker | 1M Return | 3M Return | 6M Return | Ann. Volatility"]
    lines.append("-" * 60)
    for t in tickers:
        d = snapshot.get(t)
        if not d:
            lines.append(f"{t} | no data available")
            continue
        r1 = f"{d['return_1m']}%" if d["return_1m"] is not None else "n/a"
        r3 = f"{d['return_3m']}%" if d["return_3m"] is not None else "n/a"
        r6 = f"{d['return_6m']}%" if d["return_6m"] is not None else "n/a"
        vol = f"{d['annualized_volatility_pct']}%" if d["annualized_volatility_pct"] is not None else "n/a"
        lines.append(f"{t} | {r1} | {r3} | {r6} | {vol}")
    return "\n".join(lines)
