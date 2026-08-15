"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Uses only working models (Ollama Cloud) with 3 calls per model.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Base class for LLM ETF analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.top_n = config.get("TOP_N", 3)
    
    def build_prompt(self, universe_name: str, tickers: List[str], 
                     existing_picks: List[str] = None) -> str:
        """Build a prompt asking for the SINGLE best ETF."""
        
        available_tickers = tickers
        if existing_picks and len(existing_picks) > 0:
            available_tickers = [t for t in tickers if t not in existing_picks]
            if not available_tickers:
                return None
        
        ticker_list = ', '.join(available_tickers)
        
        prompt = f"""You are a financial analyst. Select the SINGLE BEST ETF from this list that will perform best tomorrow.

Universe: {universe_name}
Available ETFs: {ticker_list}

Return ONLY this JSON format:
{{"ticker": "GLD", "expected_return": 1.5, "confidence": "High", "rationale": "Safe-haven demand"}}

Return ONLY the JSON, no other text."""
        
        return prompt
    
    def parse_response(self, response_text: str) -> Dict:
        """Parse a single ETF response."""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "ticker" in data:
                    return data
        except:
            pass
        
        return None


# ============================================
# OLLAMA CLOUD ANALYZER (Only working models)
# ============================================

class OllamaCloudAnalyzer(LLMAnalyzer):
    """Ollama Cloud API - 3 separate calls per model."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.api_key = os.environ.get("OLLAMA_API_KEY") or config.get("OLLAMA_API_KEY")
        self.base_url = "https://api.ollama.com"
        
        # Only models that are confirmed working
        self.models = [
            "nemotron-3-nano:30b",
            "gemma4:31b",
            "minimax-m3",
        ]
        self.available_models = []
        self._check_availability()
    
    def _check_availability(self):
        """Check which models are available."""
        if not self.api_key:
            logger.warning("⚠️ OLLAMA_API_KEY not found")
            return
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/api/tags", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                available = [m["name"] for m in data.get("models", [])]
                
                for model in self.models:
                    if model in available:
                        self.available_models.append(model)
                
                if self.available_models:
                    logger.info(f"✅ Ollama Cloud: {len(self.available_models)} models available")
                    logger.info(f"   Models: {self.available_models}")
                else:
                    logger.warning("⚠️ No matching models found")
                    logger.info(f"   Available: {available[:5]}")
            else:
                logger.warning(f"⚠️ API returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Connection error: {str(e)[:50]}")
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze - make 3 separate calls per model."""
        if not self.available_models:
            return {"selections": [], "consensus": {}}
        
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.available_models)} Ollama models (3 calls each)...")
        
        for model in self.available_models:
            model_picks = []
            existing = []
            available = tickers.copy()
            
            for pick_num in range(1, self.top_n + 1):
                try:
                    if not available:
                        break
                    
                    prompt = self.build_prompt(universe_name, available, existing)
                    if prompt is None:
                        break
                    
                    response = self._call_api(model, prompt)
                    if response:
                        pick = self.parse_response(response)
                        if pick and pick.get("ticker"):
                            ticker = pick.get("ticker", "").upper()
                            if ticker in available:
                                pick["model"] = f"ollama/{model}"
                                pick["pick_number"] = pick_num
                                model_picks.append(pick)
                                existing.append(ticker)
                                available.remove(ticker)
                                logger.info(f"    ✅ ollama/{model} pick {pick_num}: {ticker} ({pick.get('expected_return', 0)}%)")
                                continue
                    
                    # If we get here, the pick failed or ticker was invalid
                    logger.warning(f"    ⚠️ ollama/{model} pick {pick_num}: Failed, retrying...")
                    
                except Exception as e:
                    logger.warning(f"    ❌ ollama/{model} pick {pick_num}: {str(e)[:50]}")
            
            if model_picks:
                results.extend(model_picks)
                successful_models.append(f"ollama/{model}")
                logger.info(f"    ✅ ollama/{model}: {len(model_picks)} picks")
            else:
                logger.warning(f"    ⚠️ ollama/{model}: 0 picks")
        
        return self._aggregate_results(results, successful_models)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call Ollama Cloud API."""
        if not prompt:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 300,
        }
        
        response = requests.post(f"{self.base_url}/api/generate", json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["response"]
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results - count each pick as 1 vote."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_votes = {}
        ticker_data = {}
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            model = r.get("model", "unknown")
            
            if not ticker:
                continue
            
            if ticker not in ticker_votes:
                ticker_votes[ticker] = 0
                ticker_data[ticker] = {
                    "returns": [],
                    "confidences": [],
                    "rationales": [],
                    "models": []
                }
            
            ticker_votes[ticker] += 1
            ticker_data[ticker]["returns"].append(r.get("expected_return", 0.5))
            ticker_data[ticker]["confidences"].append(r.get("confidence", "Medium"))
            ticker_data[ticker]["rationales"].append(r.get("rationale", ""))
            
            if model and model != 'unknown' and model not in ticker_data[ticker]["models"]:
                ticker_data[ticker]["models"].append(model)
        
        sorted_tickers = sorted(
            ticker_votes.items(),
            key=lambda x: (x[1], sum(ticker_data[x[0]]["returns"]) / len(ticker_data[x[0]]["returns"])),
            reverse=True
        )
        
        top_tickers = [t for t, _ in sorted_tickers[:self.top_n]]
        
        selections = []
        for ticker in top_tickers:
            data = ticker_data[ticker]
            avg_return = sum(data["returns"]) / len(data["returns"])
            
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            selections.append({
                "ticker": ticker,
                "expected_return": round(avg_return, 2),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": ticker_votes[ticker],
                "models": data["models"]
            })
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": len(results),
                "ticker_votes": ticker_votes,
                "models_used": list(set(successful_models))
            }
        }


# ============================================
# ENSEMBLE ANALYZER
# ============================================

class EnsembleAnalyzer:
    """Combine all LLM analyzers into one ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzers = []
        
        # Only use Ollama Cloud (working models)
        ollama = OllamaCloudAnalyzer(config)
        if ollama.available_models:
            self.analyzers.append(ollama)
            logger.info("✅ Ollama Cloud analyzer initialized")
        else:
            logger.error("❌ Ollama Cloud not available")
        
        if not self.analyzers:
            logger.error("❌ No LLM analyzers available")
            logger.info("   Please set OLLAMA_API_KEY in your environment")
    
    def analyze_universe(self, universe_name: str, tickers: List[str]) -> Dict:
        """Run all analyzers on a universe."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=len(self.analyzers)) as executor:
            futures = []
            for analyzer in self.analyzers:
                future = executor.submit(analyzer.analyze, universe_name, tickers)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=180)
                    if result.get("selections"):
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Analyzer failed: {e}")
        
        return self._ensemble_vote(all_results)
    
    def _ensemble_vote(self, results: List[Dict]) -> Dict:
        """Vote across all analyzer results."""
        if not results:
            return {"selections": [], "ensemble_stats": {}}
        
        all_votes = {}
        all_data = {}
        all_models = set()
        
        for result in results:
            for sel in result.get("selections", []):
                ticker = sel.get("ticker", "").upper()
                if ticker:
                    all_votes[ticker] = all_votes.get(ticker, 0) + 1
                    if ticker not in all_data:
                        all_data[ticker] = {
                            "returns": [],
                            "confidences": [],
                            "rationales": [],
                            "models": []
                        }
                    all_data[ticker]["returns"].append(sel.get("expected_return", 0.5))
                    all_data[ticker]["confidences"].append(sel.get("confidence", "Medium"))
                    all_data[ticker]["rationales"].append(sel.get("rationale", ""))
                    
                    models = sel.get("models", [])
                    if isinstance(models, list):
                        for m in models:
                            if m and m != 'unknown' and m not in all_data[ticker]["models"]:
                                all_data[ticker]["models"].append(m)
                        all_models.update(models)
            
            consensus_models = result.get("consensus", {}).get("models_used", [])
            if isinstance(consensus_models, list):
                all_models.update(consensus_models)
        
        sorted_votes = sorted(
            all_votes.items(),
            key=lambda x: (x[1], sum(all_data[x[0]]["returns"]) / len(all_data[x[0]]["returns"])),
            reverse=True
        )
        
        top_tickers = [t for t, _ in sorted_votes[:self.config.get("TOP_N", 3)]]
        
        selections = []
        for ticker in top_tickers:
            data = all_data[ticker]
            avg_return = sum(data["returns"]) / len(data["returns"])
            
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            selections.append({
                "ticker": ticker,
                "expected_return": round(avg_return, 2),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": all_votes[ticker],
                "models": data["models"][:10]
            })
        
        return {
            "selections": selections,
            "ensemble_stats": {
                "total_analyzers": len(results),
                "total_votes": sum(all_votes.values()),
                "ticker_votes": all_votes,
                "models_used": sorted(list(all_models))
            }
        }
