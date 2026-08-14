"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Uses Ollama Cloud with Nemotron (free) + OpenRouter fallback.
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
    
    def build_prompt(self, universe_name: str, tickers: List[str]) -> str:
        """Build a simple prompt asking for 3 picks."""
        
        ticker_list = ', '.join(tickers)
        
        prompt = f"""You are a financial analyst. Select the top 3 ETFs from this list that will perform best tomorrow.

Universe: {universe_name}
Available ETFs: {ticker_list}

Return ONLY this JSON format with exactly 3 selections:
{{"selections": [{{"ticker": "GLD", "expected_return": 1.5, "confidence": "High", "rationale": "Safe-haven demand"}}]}}

Return ONLY the JSON, no other text. Exactly 3 selections required."""
        
        return prompt
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response."""
        selections = []
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                selections = data.get("selections", [])
                if selections:
                    return selections[:self.top_n]
        except:
            pass
        
        # Fallback: extract tickers
        tickers_found = re.findall(r'([A-Z]{1,5})', response_text)
        for ticker in tickers_found[:self.top_n]:
            if len(ticker) >= 2:
                selections.append({
                    "ticker": ticker,
                    "expected_return": 0.5,
                    "confidence": "Medium",
                    "rationale": "Extracted from response"
                })
        
        return selections


# ============================================
# OLLAMA CLOUD ANALYZER (Nemotron only)
# ============================================

class OllamaCloudAnalyzer(LLMAnalyzer):
    """Ollama Cloud API - using Nemotron 3 Nano (free)."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.api_key = os.environ.get("OLLAMA_API_KEY") or config.get("OLLAMA_API_KEY")
        self.base_url = "https://api.ollama.com"
        
        # Only use Nemotron - confirmed free model
        self.models = [
            "nemotron-3-nano:30b",   # NVIDIA Nemotron 30B - FREE
            "nemotron-3-nano:4b",    # NVIDIA Nemotron 4B - FREE (faster)
            "gemma4:31b-cloud",
        ]
        self.available_models = []
        
        if self.api_key:
            logger.info(f"✅ OLLAMA_API_KEY found (length: {len(self.api_key)})")
            self._check_availability()
        else:
            logger.warning("⚠️ OLLAMA_API_KEY not found")
    
    def _check_availability(self):
        """Check if Nemotron models are available."""
        if not self.api_key:
            return
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            logger.info(f"🔍 Checking available models...")
            
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                available = [m["name"] for m in data.get("models", [])]
                
                # Check which Nemotron models are available
                for model in self.models:
                    if model in available:
                        self.available_models.append(model)
                
                if self.available_models:
                    logger.info(f"✅ Nemotron models available: {self.available_models}")
                else:
                    logger.warning(f"⚠️ No Nemotron models found. Available: {[m for m in available if 'nemotron' in m.lower()]}")
                    # Try to find any Nemotron model
                    for m in available:
                        if 'nemotron' in m.lower():
                            self.available_models.append(m)
                            logger.info(f"   Using fallback: {m}")
                            break
                    
                    if not self.available_models:
                        logger.warning("⚠️ No Nemotron models available")
            else:
                logger.warning(f"⚠️ API returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Connection error: {str(e)[:50]}")
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using Nemotron models."""
        if not self.available_models:
            return {"selections": [], "consensus": {}}
        
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.available_models)} Nemotron models...")
        
        for model in self.available_models:
            try:
                response = self._call_api(model, prompt)
                if response:
                    picks = self.parse_response(response)
                    if picks:
                        for pick in picks:
                            pick["model"] = f"nemotron/{model}"
                        results.extend(picks)
                        successful_models.append(f"nemotron/{model}")
                        logger.info(f"    ✅ nemotron/{model}: {len(picks)} picks")
                    else:
                        logger.warning(f"    ⚠️ nemotron/{model}: No picks")
                else:
                    logger.warning(f"    ⚠️ nemotron/{model}: No response")
            except Exception as e:
                logger.warning(f"    ❌ nemotron/{model}: {str(e)[:50]}")
        
        return self._aggregate_results(results, successful_models)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call Ollama Cloud API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 500,
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results."""
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
        
        # Ollama Cloud with Nemotron (FREE)
        ollama = OllamaCloudAnalyzer(config)
        if ollama.available_models:
            self.analyzers.append(ollama)
            logger.info("✅ Nemotron analyzer initialized (FREE!)")
        
        if not self.analyzers:
            logger.error("❌ No LLM analyzers available")
            logger.info("   Please check your OLLAMA_API_KEY")
            logger.info("   Get your key from: https://ollama.com/settings/keys")
    
    def analyze_universe(self, universe_name: str, tickers: List[str]) -> Dict:
        """Run all analyzers on a universe."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=len(self.analyzers)) as executor:
            futures = []
            for analyzer in self.analyzers:
                future = executor.submit(
                    analyzer.analyze, universe_name, tickers
                )
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
