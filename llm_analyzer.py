"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Queries ALL LLMs (OpenRouter + Ollama) and aggregates their picks.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Base class for LLM ETF analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.top_n = config.get("TOP_N", 3)
    
    def build_single_pick_prompt(self, universe_name: str, tickers: List[str], 
                                  existing_picks: List[str] = None) -> str:
        """Build a prompt asking for the single best ETF, excluding already picked ones."""
        
        ticker_list = ', '.join(tickers)
        
        # Build exclusion text
        exclude_text = ""
        if existing_picks and len(existing_picks) > 0:
            exclude_text = f"\n\nIMPORTANT: You have already picked these ETFs: {', '.join(existing_picks)}. Do NOT pick them again. Pick a DIFFERENT ETF."
        
        prompt = f"""You are a professional financial analyst.

UNIVERSE: {universe_name}
TICKERS TO CHOOSE FROM: {ticker_list}

TASK: Select the SINGLE BEST ETF from this universe that will outperform tomorrow.{exclude_text}

Return JSON with exactly this format:
{{"ticker": "GDX", "expected_return": 1.5, "confidence": "High", "rationale": "Safe-haven demand."}}

Return ONLY the JSON, no other text.
"""
        return prompt


class OpenRouterAnalyzer(LLMAnalyzer):
    """OpenRouter API implementation - forces 3 picks by making 3 calls with exclusion."""
    
    def __init__(self, config: Dict, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.models = [
            "openai/gpt-4o-mini",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3.1-8b-instruct",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-72b-instruct",
        ]
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using ALL OpenRouter models - force 3 picks per model."""
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.models)} OpenRouter models (3 picks each)...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for model in self.models:
                # Submit 3 jobs per model with exclusion of previous picks
                future1 = executor.submit(
                    self._get_single_pick, 
                    model, 
                    universe_name, 
                    tickers, 
                    []  # No exclusions for first pick
                )
                futures[future1] = (model, 1)
            
            # We need to collect results to feed back for exclusion
            # This requires a two-pass approach:
            # 1. Get first picks
            # 2. Get second picks (excluding first)
            # 3. Get third picks (excluding first and second)
            
            # First pass: get all first picks
            first_results = {}
            for future in as_completed([f for f in futures if futures[f][1] == 1]):
                model, pick_num = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        result["model"] = model
                        result["pick_number"] = pick_num
                        first_results[model] = result
                        if model not in successful_models:
                            successful_models.append(model)
                        logger.info(f"    ✅ {model} pick 1: {result.get('ticker', 'N/A')}")
                    else:
                        logger.warning(f"    ⚠️ {model} pick 1: No result")
                except Exception as e:
                    logger.warning(f"    ❌ {model} pick 1: {str(e)[:50]}...")
            
            # Second pass: get second picks (excluding first)
            second_futures = {}
            for model in self.models:
                first_pick = first_results.get(model)
                existing = [first_pick.get("ticker")] if first_pick else []
                future = executor.submit(
                    self._get_single_pick, 
                    model, 
                    universe_name, 
                    tickers, 
                    existing
                )
                second_futures[future] = (model, 2)
            
            second_results = {}
            for future in as_completed(second_futures):
                model, pick_num = second_futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        result["model"] = model
                        result["pick_number"] = pick_num
                        second_results[model] = result
                        if model not in successful_models:
                            successful_models.append(model)
                        logger.info(f"    ✅ {model} pick 2: {result.get('ticker', 'N/A')}")
                    else:
                        logger.warning(f"    ⚠️ {model} pick 2: No result")
                except Exception as e:
                    logger.warning(f"    ❌ {model} pick 2: {str(e)[:50]}...")
            
            # Third pass: get third picks (excluding first and second)
            third_futures = {}
            for model in self.models:
                first_pick = first_results.get(model)
                second_pick = second_results.get(model)
                existing = []
                if first_pick:
                    existing.append(first_pick.get("ticker"))
                if second_pick:
                    existing.append(second_pick.get("ticker"))
                future = executor.submit(
                    self._get_single_pick, 
                    model, 
                    universe_name, 
                    tickers, 
                    existing
                )
                third_futures[future] = (model, 3)
            
            for future in as_completed(third_futures):
                model, pick_num = third_futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        result["model"] = model
                        result["pick_number"] = pick_num
                        results.append(result)
                        if model not in successful_models:
                            successful_models.append(model)
                        logger.info(f"    ✅ {model} pick 3: {result.get('ticker', 'N/A')}")
                    else:
                        logger.warning(f"    ⚠️ {model} pick 3: No result")
                except Exception as e:
                    logger.warning(f"    ❌ {model} pick 3: {str(e)[:50]}...")
            
            # Add first and second results to final results
            for result in first_results.values():
                results.append(result)
            for result in second_results.values():
                results.append(result)
        
        logger.info(f"  ✅ {len(results)} total picks from {len(successful_models)} models")
        
        return self._aggregate_results(results, successful_models)
    
    def _get_single_pick(self, model: str, universe_name: str, 
                         tickers: List[str], existing_picks: List[str]) -> Optional[Dict]:
        """Get a single pick from a model, excluding already picked ones."""
        prompt = self.build_single_pick_prompt(universe_name, tickers, existing_picks)
        
        try:
            response = self._call_api(model, prompt)
            if response:
                return self.parse_single_response(response)
        except Exception as e:
            logger.warning(f"  {model} failed: {str(e)[:50]}")
        
        return None
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/P2SAMAPA/P2-ETF-LLM-ETF-ENSEMBLE",
            "X-Title": "LLM ETF Ensemble"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a financial analyst. Return ONLY valid JSON with a single ETF selection."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300,
        }
        
        response = requests.post(self.base_url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def parse_single_response(self, response_text: str) -> Dict:
        """Parse a single ETF response."""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "ticker" in data:
                    return data
                elif "selections" in data and data["selections"]:
                    return data["selections"][0]
        except:
            pass
        
        # Fallback: extract ticker
        tickers_found = re.findall(r'([A-Z]{1,5})', response_text)
        for ticker in tickers_found:
            if len(ticker) >= 2:
                return {
                    "ticker": ticker,
                    "expected_return": 0.5,
                    "confidence": "Medium",
                    "rationale": response_text[:100]
                }
        
        return None
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results - count each selection as 1 vote."""
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


class OllamaAnalyzer(LLMAnalyzer):
    """Ollama (local) implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("OLLAMA_URL", "http://localhost:11434")
        self.models = config.get("OLLAMA_MODELS", [
            "llama3.2:3b",
            "phi3:mini",
            "mistral:7b",
        ])
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                available_models = [m["name"] for m in response.json().get("models", [])]
                self.models = [m for m in self.models if m in available_models]
                if self.models:
                    logger.info(f"✅ Ollama available with models: {self.models}")
                else:
                    logger.warning("⚠️ No Ollama models available")
                    self.models = []
            else:
                self.models = []
        except:
            self.models = []
            logger.warning("⚠️ Ollama not available")
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using Ollama models - force 3 picks per model with exclusion."""
        if not self.models:
            return {"selections": [], "consensus": {}}
        
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.models)} Ollama models (3 picks each)...")
        
        for model in self.models:
            model_results = []
            
            # Pick 1: No exclusions
            try:
                prompt = self.build_single_pick_prompt(universe_name, tickers, [])
                response = self._call_api(model, prompt)
                if response:
                    result = self.parse_single_response(response)
                    if result:
                        model_name = f"ollama/{model}"
                        result["model"] = model_name
                        result["pick_number"] = 1
                        model_results.append(result)
                        if model_name not in successful_models:
                            successful_models.append(model_name)
                        logger.info(f"    ✅ ollama/{model} pick 1: {result.get('ticker', 'N/A')}")
            except Exception as e:
                logger.warning(f"    ❌ ollama/{model} pick 1: {str(e)[:50]}...")
            
            # Pick 2: Exclude pick 1
            if model_results:
                try:
                    existing = [model_results[0].get("ticker")]
                    prompt = self.build_single_pick_prompt(universe_name, tickers, existing)
                    response = self._call_api(model, prompt)
                    if response:
                        result = self.parse_single_response(response)
                        if result:
                            model_name = f"ollama/{model}"
                            result["model"] = model_name
                            result["pick_number"] = 2
                            model_results.append(result)
                            logger.info(f"    ✅ ollama/{model} pick 2: {result.get('ticker', 'N/A')}")
                except Exception as e:
                    logger.warning(f"    ❌ ollama/{model} pick 2: {str(e)[:50]}...")
            
            # Pick 3: Exclude picks 1 and 2
            if len(model_results) >= 2:
                try:
                    existing = [r.get("ticker") for r in model_results[:2]]
                    prompt = self.build_single_pick_prompt(universe_name, tickers, existing)
                    response = self._call_api(model, prompt)
                    if response:
                        result = self.parse_single_response(response)
                        if result:
                            model_name = f"ollama/{model}"
                            result["model"] = model_name
                            result["pick_number"] = 3
                            model_results.append(result)
                            logger.info(f"    ✅ ollama/{model} pick 3: {result.get('ticker', 'N/A')}")
                except Exception as e:
                    logger.warning(f"    ❌ ollama/{model} pick 3: {str(e)[:50]}...")
            
            results.extend(model_results)
        
        logger.info(f"  ✅ {len(results)} total picks from {len(successful_models)} Ollama models")
        
        return self._aggregate_results(results, successful_models)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 300,
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["response"]
    
    def parse_single_response(self, response_text: str) -> Dict:
        """Parse a single ETF response."""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "ticker" in data:
                    return data
                elif "selections" in data and data["selections"]:
                    return data["selections"][0]
        except:
            pass
        
        # Fallback: extract ticker
        tickers_found = re.findall(r'([A-Z]{1,5})', response_text)
        for ticker in tickers_found:
            if len(ticker) >= 2:
                return {
                    "ticker": ticker,
                    "expected_return": 0.5,
                    "confidence": "Medium",
                    "rationale": response_text[:100]
                }
        
        return None
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results from Ollama models."""
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


class EnsembleAnalyzer:
    """Combine all LLM analyzers into one ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzers = []
        
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            self.analyzers.append(OpenRouterAnalyzer(config, openrouter_key))
            logger.info("✅ OpenRouter analyzer initialized")
        else:
            logger.warning("⚠️ OPENROUTER_API_KEY not set")
        
        ollama = OllamaAnalyzer(config)
        if ollama.models:
            self.analyzers.append(ollama)
            logger.info("✅ Ollama analyzer initialized")
        
        if not self.analyzers:
            logger.error("❌ No LLM analyzers available")
    
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
                    result = future.result(timeout=300)
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
