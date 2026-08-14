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
from typing import Dict, Optional, List
import multiprocessing as mp

import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data, prepare_data_summary
from llm_analyzer import EnsembleAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """Main LLM ETF Ensemble orchestrator."""
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"✅ Loaded {len(prices_df)} days, {len(prices_df.columns)} ETFs")

    run_date = datetime.now().strftime("%Y-%m-%d")
    config.RUN_DATE = run_date

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
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            logger.warning(f"⚠️ No tickers found for universe: {universe_name}")
            continue
        
        # Prepare data summary
        data_summary = prepare_data_summary(prices_df, macro_df, available)
        
        # Get LLM recommendations
        result = analyzer.analyze_universe(universe_name, available, data_summary)
        
        if result.get("selections"):
            results["universes"][universe_name] = {
                "top_picks": result["selections"],
                "ensemble_stats": result.get("ensemble_stats", {}),
                "all_tickers": available
            }
            logger.info(f"  ✅ {universe_name}: Top picks: {[s['ticker'] for s in result['selections']]}")
        else:
            logger.warning(f"  ⚠️ {universe_name}: No recommendations received")
            results["universes"][universe_name] = {
                "top_picks": [],
                "ensemble_stats": {},
                "all_tickers": available,
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
                    "probability": pick["probability"],
                    "confidence": pick["confidence"],
                    "votes": pick.get("votes", 0)
                }
    
    results["ensemble_summary"] = {
        "top_cross_universe_picks": sorted(
            all_picks.values(), 
            key=lambda x: (x["votes"], x["probability"]), 
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
    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=output_path,
                repo_id=config.RESULTS_REPO,
                token=token,
                repo_type="dataset"
            )
            logger.info("   ✅ Upload complete!")
        except Exception as e:
            logger.error(f"   Upload failed: {e}")

    return results


if __name__ == "__main__":
    run_trainer()
