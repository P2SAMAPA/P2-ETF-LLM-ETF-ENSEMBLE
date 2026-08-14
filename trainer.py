"""
trainer.py  —  LLM ETF Ensemble Trainer
========================================

Runs all LLM analyzers, aggregates results, and saves JSON outputs.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from llm_analyzer import EnsembleAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer() -> Dict:
    """Main LLM ETF Ensemble orchestrator."""
    
    logger.info("🤖 Starting LLM ETF Ensemble Analysis...")
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    
    # Initialize ensemble analyzer
    analyzer = EnsembleAnalyzer(vars(config))
    
    if not analyzer.analyzers:
        logger.error("❌ No LLM analyzers available. Check API keys and Ollama.")
        return {}

    results = {
        "run_date": run_date,
        "universes": {},
        "ensemble_summary": {}
    }

    # Analyze each universe
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name} with {len(tickers)} ETFs...")
        
        # Get LLM recommendations - NO DATA NEEDED
        result = analyzer.analyze_universe(universe_name, tickers)
        
        if result.get("selections"):
            results["universes"][universe_name] = {
                "top_picks": result["selections"],
                "ensemble_stats": result.get("ensemble_stats", {}),
                "all_tickers": tickers
            }
            logger.info(f"  ✅ {universe_name}: Top picks: {[s['ticker'] + ' (' + str(s['expected_return']) + '%)' for s in result['selections']]}")
        else:
            logger.warning(f"  ⚠️ {universe_name}: No recommendations received")
            results["universes"][universe_name] = {
                "top_picks": [],
                "ensemble_stats": {},
                "all_tickers": tickers,
                "error": "No recommendations from LLMs"
            }

    # Generate ensemble summary
    all_picks = {}
    for universe, data in results["universes"].items():
        for pick in data.get("top_picks", []):
            ticker = pick["ticker"]
            if ticker not in all_picks:
                all_picks[ticker] = {
                    "ticker": ticker,
                    "universe": universe,
                    "expected_return": pick["expected_return"],
                    "confidence": pick["confidence"],
                    "votes": pick.get("votes", 0)
                }
    
    results["ensemble_summary"] = {
        "top_cross_universe_picks": sorted(
            all_picks.values(), 
            key=lambda x: (x["votes"], x["expected_return"]), 
            reverse=True
        )[:10],
        "total_llm_calls": len(analyzer.analyzers) * len(results["universes"])
    }

    # Save results
    logger.info("\n💾 Saving JSON results...")
    output_path = f"llm_etf_ensemble_{run_date}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"   Saved: {output_path}")

    # Upload to HuggingFace
    if config.HF_TOKEN:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=config.HF_TOKEN)
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=output_path,
                repo_id=config.RESULTS_REPO,
                token=config.HF_TOKEN,
                repo_type="dataset"
            )
            logger.info("   ✅ Upload complete!")
        except Exception as e:
            logger.error(f"   Upload failed: {e}")

    return results


if __name__ == "__main__":
    run_trainer()
