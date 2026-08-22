"""
trainer.py  —  LLM ETF Ensemble Trainer
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from llm_analyzer import EnsembleAnalyzer
from market_data import fetch_universe_snapshot
from results_store import save_and_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_cross_universe_summary(universes: Dict) -> Dict:
    """Take the #1 consensus pick from each universe for a quick overview."""
    top_picks = []
    for universe_name, data in universes.items():
        selections = data.get("top_picks", [])
        if not selections:
            continue
        best = selections[0]
        top_picks.append({
            "universe": universe_name,
            "ticker": best["ticker"],
            "predicted_return_1m": best.get("predicted_return_1m"),
            "confidence": best.get("confidence"),
            "votes": best.get("votes"),
        })
    top_picks.sort(key=lambda p: (p["votes"] or 0), reverse=True)
    return {"top_cross_universe_picks": top_picks}


def run_trainer() -> Dict:
    logger.info("Starting LLM ETF Ensemble...")

    if not config.OPENROUTER_API_KEY and not config.OLLAMA_API_KEY:
        logger.error("Neither OPENROUTER_API_KEY nor OLLAMA_API_KEY is set — nothing to run.")
        return {}

    run_started = datetime.now(timezone.utc)
    run_date = run_started.strftime("%Y-%m-%d")

    analyzer = EnsembleAnalyzer(config)
    if not analyzer.all_models():
        logger.error("No usable free models discovered on either provider — aborting.")
        return {}

    results = {
        "run_date": run_date,
        "run_started_utc": run_started.isoformat(),
        "models": {
            "openrouter": analyzer.openrouter_models,
            "ollama": analyzer.ollama_models,
        },
        "universes": {},
    }

    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n{universe_name} ({len(tickers)} ETFs)")

        logger.info("  Fetching market data...")
        snapshot = fetch_universe_snapshot(tickers, config.MARKET_DATA_LOOKBACK)

        result = analyzer.analyze_universe(universe_name, tickers, snapshot)

        if result.get("selections"):
            results["universes"][universe_name] = {
                "top_picks": result["selections"],
                "ensemble_stats": result.get("consensus", {}),
                "all_tickers": tickers,
            }
            picks = [f"{s['ticker']} ({s['votes']} votes, {s['points']} pts)" for s in result["selections"]]
            logger.info(f"  Top picks: {picks}")
        else:
            logger.warning("  No results (no model produced a valid, parseable pick)")
            results["universes"][universe_name] = {
                "top_picks": [],
                "ensemble_stats": result.get("consensus", {}),
                "all_tickers": tickers,
            }

    results["ensemble_summary"] = build_cross_universe_summary(results["universes"])
    results["run_finished_utc"] = datetime.now(timezone.utc).isoformat()

    save_and_upload(results, run_date)
    return results


if __name__ == "__main__":
    run_trainer()
