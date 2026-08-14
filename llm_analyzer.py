"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Uses multiple free LLM models from OpenRouter + Ollama Cloud.
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
        """Build a prompt that FORCES exactly 3 picks."""
        
        ticker_list = ', '.join(tickers)
        
        prompt = f"""You are a financial analyst. Select EXACTLY 3 ETFs from this list that will perform best tomorrow.

Universe: {universe_name}
Available ETFs: {ticker_list}

CRITICAL: You MUST return EXACTLY 3 selections. No more, no less.

Return ONLY this JSON format with exactly 3 selections:
{{
    "selections": [
        {{"ticker": "GLD", "expected_return": 1.5, "confidence": "High", "rationale": "Safe-haven demand"}},
        {{"ticker": "XLE", "expected_return": 1.2, "confidence": "Medium", "rationale": "Oil price recovery"}},
        {{"ticker": "XLK", "expected_return": 0.9, "confidence": "Medium", "rationale": "Tech earnings"}}
    ]
}}

Return ONLY the JSON, no other text. EXACTLY 3 selections."""
        
        return prompt
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response - ensures exactly top_n selections."""
        selections = []
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                selections = data.get("selections", [])
                if selections:
                    # Ensure we have exactly top_n
                    if len(selections) > self.top_n:
                        selections = selections[:self.top_n]
                    return selections
        except:
            pass
        
        # Fallback: extract tickers from response
        tickers_found = re.findall(r'([A-Z]{1,5})', response_text)
        unique_tickers = []
        for ticker in tickers_found:
            if len(ticker) >= 2 and ticker not in unique_tickers:
                unique_tickers.append(ticker)
                if len(unique_tickers) >= self.top_n:
                    break
        
        for ticker in unique_tickers[:self.top_n]:
            selections.append({
                "ticker": ticker,
                "expected_return": 0.5,
                "confidence": "Medium",
                "rationale": "Extracted from response"
            })
        
        # If still no selections, add some defaults
        while len(selections) < self.top_n:
            selections.append({
                "ticker": "N/A",
                "expected_return": 0.0,
                "confidence": "Low",
                "rationale": "No valid response"
            })
        
        return selections[:self.top_n]


# ============================================
# OPENROUTER ANALYZER (Free models)
# ============================================

class OpenRouterAnalyzer(LLMAnalyzer):
    """OpenRouter API - using free models only."""
    
    def __init__(self, config: Dict, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.models = [
            "poolside/laguna",
            "poolside/coral",
            "cohere/command-r-plus",
            "meta-llama/llama-3.2-3b-instruct",
            "microsoft/phi-3.5-mini-128k-instruct",
            "mistralai/mistral-7b-instruct",
            "qwen/qwen-2.5-7b-instruct",
            "google/gemini-flash-1.5",
        ]
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.working_models = []
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using OpenRouter free models."""
        if not self.api_key:
            return {"selections": [], "consensus": {}}
        
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        models_to_try = self.models[:4]  # Try first 4 models
        logger.info(f"  Querying {len(models_to_try)} OpenRouter models...")
        
        for model in models_to_try:
            try:
                response = self._call_api(model, prompt)
                if response:
                    picks = self.parse_response(response)
                    if picks and len(picks) == self.top_n:
                        for pick in picks:
                            pick["model"] = f"openrouter/{model}"
                        results.extend(picks)
                        successful_models.append(f"openrouter/{model}")
                        logger.info(f"    ✅ openrouter/{model}: {len(picks)} picks")
                    else:
                        logger.warning(f"    ⚠️ openrouter/{model}: Invalid picks ({len(picks) if picks else 0})")
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
            "HTTP-Referer": "https://github.com/P2SAMAPA/P2-ETF-LLM-ETF-ENSEMBLE",
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 600,
        }
        
        response = requests.post(self.base_url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results - properly count all votes."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_votes = {}
        ticker_data = {}
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            model = r.get("model", "unknown")
            
            if not ticker or ticker == "N/A":
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
# OLLAMA CLOUD ANALYZER (Multiple free models)
# ============================================

class OllamaCloudAnalyzer(LLMAnalyzer):
    """Ollama Cloud API - multiple free models."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.api_key = os.environ.get("OLLAMA_API_KEY") or config.get("OLLAMA_API_KEY")
        self.base_url = "https://api.ollama.com"
        
        self.models = [
            "nemotron-3-nano:30b",
            "gemma4:31b",
            "deepseek-v4-flash:preview",
            "qwen3.5:397b",
            "glm-5.1",
            "mistral-large-3:675b",
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
                    logger.info(f"   Models: {self.available_models[:3]}...")
                else:
                    logger.warning("⚠️ No matching models found")
            else:
                logger.warning(f"⚠️ API returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Connection error: {str(e)[:50]}")
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using Ollama Cloud models."""
        if not self.available_models:
            return {"selections": [], "consensus": {}}
        
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        models_to_use = self.available_models[:4]
        logger.info(f"  Querying {len(models_to_use)} Ollama Cloud models...")
        
        for model in models_to_use:
            try:
                response = self._call_api(model, prompt)
                if response:
                    picks = self.parse_response(response)
                    if picks and len(picks) == self.top_n:
                        for pick in picks:
                            pick["model"] = f"ollama/{model}"
                        results.extend(picks)
                        successful_models.append(f"ollama/{model}")
                        logger.info(f"    ✅ ollama/{model}: {len(picks)} picks")
                    else:
                        logger.warning(f"    ⚠️ ollama/{model}: Invalid picks ({len(picks) if picks else 0})")
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
            "max_tokens": 600,
        }
        
        response = requests.post(f"{self.base_url}/api/generate", json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["response"]
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results - properly count all votes."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_votes = {}
        ticker_data = {}
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            model = r.get("model", "unknown")
            
            if not ticker or ticker == "N/A":
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
# ENSEMBLE ANALYZER (Combines BOTH)
# ============================================

class EnsembleAnalyzer:
    """Combine ALL LLM analyzers (OpenRouter + Ollama) into one ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzers = []
        
        # 1. OpenRouter (free models)
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            self.analyzers.append(OpenRouterAnalyzer(config, openrouter_key))
            logger.info("✅ OpenRouter analyzer initialized")
        else:
            logger.warning("⚠️ OPENROUTER_API_KEY not set")
        
        # 2. Ollama Cloud (multiple free models)
        ollama = OllamaCloudAnalyzer(config)
        if ollama.available_models:
            self.analyzers.append(ollama)
            logger.info(f"✅ Ollama Cloud analyzer initialized")
        else:
            logger.warning("⚠️ Ollama Cloud not available")
        
        if not self.analyzers:
            logger.error("❌ No LLM analyzers available")
    
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
                if ticker and ticker != "N/A":
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
