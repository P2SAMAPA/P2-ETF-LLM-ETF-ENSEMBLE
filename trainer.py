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
from huggingface_hub import HfApi, create_repo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from llm_analyzer import EnsembleAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_repo_exists(token: str) -> bool:
    """Ensure the results repository exists on HuggingFace."""
    try:
        api = HfApi(token=token)
        try:
            repo_info = api.repo_info(
                repo_id=config.RESULTS_REPO,
                repo_type="dataset"
            )
            logger.info(f"✅ Repository {config.RESULTS_REPO} exists")
            return True
        except Exception as e:
            logger.info(f"📦 Creating repository {config.RESULTS_REPO}...")
            api.create_repo(
                repo_id=config.RESULTS_REPO,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            logger.info(f"✅ Repository created")
            return True
    except Exception as e:
        logger.warning(f"Could not check/create repo: {e}")
        return False


def run_trainer() -> Dict:
    """Main LLM ETF Ensemble orchestrator."""
    
    logger.info("🤖 Starting LLM ETF Ensemble Analysis...")
    
    # Check for API keys
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        logger.error("❌ OPENROUTER_API_KEY not set. Please set it in environment variables.")
        return {}
    
    # Ensure HF repo exists
    hf_token = os.environ.get("HF_TOKEN") or config.HF_TOKEN
    if hf_token:
        ensure_repo_exists(hf_token)
    else:
        logger.warning("⚠️ HF_TOKEN not set - will not upload results")
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    
    # Initialize ensemble analyzer
    analyzer = EnsembleAnalyzer(vars(config))
    
    if not analyzer.analyzers:
        logger.error("❌ No LLM analyzers available. Check API keys and Ollama.")
        return {}

    results = {
        "run_date": run_date,
        "universes": {},
        "ensemble_summary": {},
        "models_used": []  # Track all models used
    }

    all_models_used = set()

    # Analyze each universe
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name} with {len(tickers)} ETFs...")
        
        # Get LLM recommendations
        result = analyzer.analyze_universe(universe_name, tickers)
        
        if result.get("selections"):
            # Extract models used from this universe
            for pick in result.get("selections", []):
                models = pick.get("models", [])
                if isinstance(models, list):
                    all_models_used.update(models)
                elif models:
                    all_models_used.add(str(models))
            
            # Also get models from ensemble_stats
            stats = result.get("ensemble_stats", {})
            models_used = stats.get("models_used", [])
            if isinstance(models_used, list):
                all_models_used.update(models_used)
            elif models_used:
                all_models_used.add(str(models_used))
            
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

    # Update global models used
    results["models_used"] = sorted(list(all_models_used))
    logger.info(f"✅ All models used: {len(all_models_used)} models")

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
                    "votes": pick.get("votes", 0),
                    "models": pick.get("models", [])
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
    if hf_token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=output_path,
                repo_id=config.RESULTS_REPO,
                token=hf_token,
                repo_type="dataset"
            )
            logger.info("   ✅ Upload complete!")
        except Exception as e:
            logger.error(f"   Upload failed: {e}")
    else:
        logger.warning("   ⚠️ Skipping upload (HF_TOKEN not set)")

    return results


if __name__ == "__main__":
    run_trainer()
