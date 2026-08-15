"""
trainer.py  —  LLM ETF Ensemble Trainer
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict

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
    logger.info("🤖 Starting LLM ETF Ensemble...")
    
    # Check API key
    if not os.environ.get("OLLAMA_API_KEY"):
        logger.error("❌ OLLAMA_API_KEY not set")
        return {}
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    analyzer = EnsembleAnalyzer(vars(config))
    
    results = {"run_date": run_date, "universes": {}}
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 {universe_name} ({len(tickers)} ETFs)")
        result = analyzer.analyze_universe(universe_name, tickers)
        
        if result.get("selections"):
            results["universes"][universe_name] = {
                "top_picks": result["selections"],
                "ensemble_stats": result.get("consensus", {}),
                "all_tickers": tickers
            }
            picks = [f"{s['ticker']} ({s['votes']} votes)" for s in result["selections"]]
            logger.info(f"  ✅ Top picks: {picks}")
        else:
            logger.warning(f"  ⚠️ No results")
            results["universes"][universe_name] = {"top_picks": [], "all_tickers": tickers}
    
    # Save
    output_path = f"llm_etf_ensemble_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n💾 Saved: {output_path}")
    
    # Upload to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=output_path,
                repo_id=config.RESULTS_REPO,
                token=hf_token,
                repo_type="dataset"
            )
            logger.info("✅ Uploaded to HuggingFace")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
    
    return results


if __name__ == "__main__":
    run_trainer()
