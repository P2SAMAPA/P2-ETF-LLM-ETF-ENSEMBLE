"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Simple: Each model picks ONE ETF per universe. Count votes. Show top 3.
"""

import os
import json
import logging
import requests
from typing import Dict, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)


def build_prompt(universe_name: str, tickers: List[str]) -> str:
    """Prompt for ONE ETF pick from the provided list only."""
    ticker_list = ', '.join(tickers)
    return f"""Select the SINGLE BEST ETF from this EXACT list for tomorrow.

Universe: {universe_name}
Available ETFs (pick ONLY from this list): {ticker_list}

IMPORTANT: Pick ONLY from the list above. Do NOT pick anything else.

Return ONLY: {{"ticker": "GLD", "expected_return": 1.5, "confidence": "High", "rationale": "Safe-haven"}}"""


def parse_response(text: str) -> Dict:
    """Parse response - no fallback extraction."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if "ticker" in data:
                return data
    except:
        pass
    return None


class OllamaAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get("OLLAMA_API_KEY")
        self.base_url = "https://api.ollama.com"
        self.models = ["nemotron-3-nano:30b", "gemma4:31b", "minimax-m3"]
        self.available = []
        self._check()
    
    def _check(self):
        if not self.api_key:
            return
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            r = requests.get(f"{self.base_url}/api/tags", headers=headers, timeout=5)
            if r.status_code == 200:
                available = [m["name"] for m in r.json().get("models", [])]
                self.available = [m for m in self.models if m in available]
                if self.available:
                    logger.info(f"✅ Ollama: {self.available}")
        except:
            pass
    
    def pick_one(self, universe: str, tickers: List[str]) -> Dict:
        """Each model picks ONE ETF from the provided tickers only."""
        prompt = build_prompt(universe, tickers)
        results = []
        
        for model in self.available:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                data = {"model": model, "prompt": prompt, "stream": False, "temperature": 0.3, "max_tokens": 200}
                r = requests.post(f"{self.base_url}/api/generate", json=data, headers=headers, timeout=20)
                if r.status_code == 200:
                    pick = parse_response(r.json().get("response", ""))
                    if pick:
                        ticker = pick.get("ticker", "").upper()
                        # VALIDATE: Only accept if ticker is in the universe
                        if ticker in tickers:
                            pick["model"] = f"ollama/{model}"
                            results.append(pick)
                            logger.info(f"  ✅ {model}: {ticker}")
                        else:
                            logger.warning(f"  ⚠️ {model}: {ticker} not in universe, skipping")
            except Exception as e:
                logger.warning(f"  ❌ {model}: {str(e)[:50]}")
        
        return self._count_votes(results, tickers)
    
    def _count_votes(self, results: List[Dict], valid_tickers: List[str]) -> Dict:
        """Count votes - only for valid tickers in the universe."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        # Count votes - only for valid tickers
        votes = {}
        data = {}
        for r in results:
            ticker = r.get("ticker", "").upper()
            if not ticker or ticker not in valid_tickers:
                continue
            votes[ticker] = votes.get(ticker, 0) + 1
            if ticker not in data:
                data[ticker] = {"returns": [], "confidences": [], "rationales": [], "models": []}
            data[ticker]["returns"].append(r.get("expected_return", 0.5))
            data[ticker]["confidences"].append(r.get("confidence", "Medium"))
            data[ticker]["rationales"].append(r.get("rationale", ""))
            data[ticker]["models"].append(r.get("model", "unknown"))
        
        if not votes:
            return {"selections": [], "consensus": {"total_votes": len(results), "ticker_votes": {}, "models_used": []}}
        
        # Filter: only ETFs with >= 2 votes
        qualified = {t: v for t, v in votes.items() if v >= 2}
        
        if not qualified:
            # If no ETF has 2+ votes, return top 1
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            top = sorted_votes[:1]
        else:
            # Sort by votes descending, then by return
            sorted_votes = sorted(qualified.items(), key=lambda x: (x[1], sum(data[x[0]]["returns"]) / len(data[x[0]]["returns"])), reverse=True)
            top = sorted_votes[:3]  # Top 3
        
        selections = []
        for ticker, vote_count in top:
            d = data[ticker]
            conf_counts = {}
            for c in d["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            
            selections.append({
                "ticker": ticker,
                "expected_return": round(sum(d["returns"]) / len(d["returns"]), 2),
                "confidence": max(conf_counts, key=conf_counts.get),
                "rationale": d["rationales"][0] if d["rationales"] else "",
                "votes": vote_count,
                "models": list(set(d["models"]))
            })
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": len(results),
                "ticker_votes": votes,
                "models_used": list(set([r.get("model", "") for r in results if r.get("model")]))
            }
        }


class EnsembleAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.ollama = OllamaAnalyzer()
        if self.ollama.available:
            logger.info("✅ Ollama analyzer ready")
    
    def analyze_universe(self, universe_name: str, tickers: List[str]) -> Dict:
        if not self.ollama.available:
            return {"selections": [], "ensemble_stats": {}}
        return self.ollama.pick_one(universe_name, tickers)
