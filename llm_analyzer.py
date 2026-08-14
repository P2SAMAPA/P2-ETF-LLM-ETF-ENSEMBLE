"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Uses Ollama Cloud + OpenRouter as fallback.
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
# OLLAMA CLOUD ANALYZER
# ============================================

class OllamaCloudAnalyzer(LLMAnalyzer):
    """Ollama Cloud API - free models."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.api_key = os.environ.get("OLLAMA_API_KEY") or config.get("OLLAMA_API_KEY")
        
        # Try different possible URLs for Ollama Cloud
        self.base_urls = [
            os.environ.get("OLLAMA_URL") or config.get("OLLAMA_URL"),
            "https://ollama.com/api",
            "https://api.ollama.com",
            "https://ollama.ai/api",
        ]
        self.base_urls = [u for u in self.base_urls if u]  # Remove None/empty
        
        self.models = [
            "nemotron-3-nano:4b",
            "llama3.2:3b",
            "phi3:mini",
            "mistral:7b",
        ]
        self.available_models = []
        self.active_url = None
        
        if self.api_key:
            logger.info(f"✅ OLLAMA_API_KEY found (length: {len(self.api_key)})")
            self._check_availability()
        else:
            logger.warning("⚠️ OLLAMA_API_KEY not found")
    
    def _check_availability(self):
        """Check if Ollama Cloud API key is valid."""
        if not self.api_key:
            return
        
        for url in self.base_urls:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                logger.info(f"🔍 Trying Ollama Cloud at: {url}")
                
                response = requests.get(
                    f"{url}/api/tags",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.active_url = url
                    data = response.json()
                    available = [m["name"] for m in data.get("models", [])]
                    self.available_models = [m for m in self.models if m in available]
                    if self.available_models:
                        logger.info(f"✅ Ollama Cloud connected at {url}")
                        logger.info(f"   Available models: {self.available_models}")
                    else:
                        logger.warning(f"⚠️ No matching models at {url}")
                        if available:
                            logger.info(f"   Available: {available}")
                            # Use first available model
                            self.available_models = available[:3]
                            logger.info(f"   Using: {self.available_models}")
                    return
                elif response.status_code == 401:
                    logger.warning(f"⚠️ Authentication failed at {url}")
                else:
                    logger.warning(f"⚠️ {url} returned {response.status_code}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"⚠️ Cannot connect to {url}")
            except Exception as e:
                logger.warning(f"⚠️ {url} error: {str(e)[:50]}")
        
        if not self.available_models:
            logger.error("❌ No Ollama Cloud endpoints worked")
            logger.info("   Make sure your OLLAMA_API_KEY is valid")
            logger.info("   Get your key from: https://ollama.com/settings/keys")
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using Ollama Cloud models."""
        if not self.available_models or not self.active_url:
            return {"selections": [], "consensus": {}}
        
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.available_models)} Ollama Cloud models...")
        
        for model in self.available_models[:3]:  # Limit to 3 models for speed
            try:
                response = self._call_api(model, prompt)
                if response:
                    picks = self.parse_response(response)
                    if picks:
                        for pick in picks:
                            pick["model"] = f"ollama/{model}"
                        results.extend(picks)
                        successful_models.append(f"ollama/{model}")
                        logger.info(f"    ✅ ollama/{model}: {len(picks)} picks")
                    else:
                        logger.warning(f"    ⚠️ ollama/{model}: No picks")
                else:
                    logger.warning(f"    ⚠️ ollama/{model}: No response")
            except Exception as e:
                logger.warning(f"    ❌ ollama/{model}: {str(e)[:50]}")
        
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
            f"{self.active_url}/api/generate",
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
# OPENROUTER ANALYZER (Fallback)
# ============================================

class OpenRouterAnalyzer(LLMAnalyzer):
    """OpenRouter API - as fallback."""
    
    def __init__(self, config: Dict, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.models = [
            "meta-llama/llama-3.2-3b-instruct",
            "microsoft/phi-3-mini-128k-instruct",
        ]
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using OpenRouter models."""
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.models)} OpenRouter models...")
        
        for model in self.models:
            try:
                response = self._call_api(model, prompt)
                if response:
                    picks = self.parse_response(response)
                    if picks:
                        for pick in picks:
                            pick["model"] = f"openrouter/{model}"
                        results.extend(picks)
                        successful_models.append(f"openrouter/{model}")
                        logger.info(f"    ✅ openrouter/{model}: {len(picks)} picks")
                    else:
                        logger.warning(f"    ⚠️ openrouter/{model}: No picks")
                else:
                    logger.warning(f"    ⚠️ openrouter/{model}: No response")
            except Exception as e:
                logger.warning(f"    ❌ openrouter/{model}: {str(e)[:50]}")
        
        return self._aggregate_results(results, successful_models)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        
        response = requests.post(self.base_url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
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
        
        # 1. Ollama Cloud (FREE - with API key)
        ollama = OllamaCloudAnalyzer(config)
        if ollama.available_models:
            self.analyzers.append(ollama)
            logger.info("✅ Ollama Cloud analyzer initialized (FREE models!)")
        else:
            logger.warning("⚠️ Ollama Cloud not available")
        
        # 2. OpenRouter (fallback - requires credits)
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            self.analyzers.append(OpenRouterAnalyzer(config, openrouter_key))
            logger.info("✅ OpenRouter analyzer initialized (fallback)")
        
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
