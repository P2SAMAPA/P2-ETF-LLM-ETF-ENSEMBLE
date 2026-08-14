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

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Base class for LLM ETF analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.top_n = config.get("TOP_N", 3)
        
    def build_prompt(self, universe_name: str, tickers: List[str]) -> str:
        """Build the prompt for LLM analysis."""
        
        prompt = f"""You are a professional financial analyst with access to real-time market data, macroeconomic indicators, Fed policy, and global news flow.

UNIVERSE: {universe_name}
ETF TICKERS TO ANALYZE: {', '.join(tickers)}

Based on your analysis of current market conditions (including but not limited to):
- Recent price action and technical levels
- Sector rotation and relative strength
- Macroeconomic data (inflation, GDP, employment)
- Federal Reserve policy expectations
- Global geopolitical risks and opportunities
- Fund flows and sentiment indicators

Select the top {self.top_n} ETFs from this universe that are MOST LIKELY to outperform in the NEXT US TRADING DAY.

For each selected ETF, provide:
1. Ticker symbol
2. Expected return (%) for next trading day
3. Confidence level (High/Medium/Low)
4. Brief rationale (1-2 sentences)

Format your response as JSON:
{{
    "selections": [
        {{
            "ticker": "GDX",
            "expected_return": 1.5,
            "confidence": "High",
            "rationale": "Gold miners benefit from safe-haven demand amid geopolitical tensions."
        }}
    ]
}}

Return ONLY the JSON, no other text. Use TODAY'S date ({datetime.now().strftime('%Y-%m-%d')}) for your analysis.
"""
        return prompt
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response into structured selections."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("selections", [])
        except:
            pass
        
        # Fallback: manual parsing
        selections = []
        lines = response_text.strip().split('\n')
        for line in lines:
            if 'ticker' in line.lower() or 'etf' in line.lower():
                parts = line.split('|')
                if len(parts) >= 2:
                    ticker = parts[0].strip().split(':')[-1].strip()[:5]
                    selections.append({
                        "ticker": ticker,
                        "expected_return": 0.5,
                        "confidence": "Medium",
                        "rationale": line.strip()
                    })
        
        return selections[:self.top_n]


class OpenRouterAnalyzer(LLMAnalyzer):
    """OpenRouter API implementation - queries ALL available models."""
    
    def __init__(self, config: Dict, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.models = [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/mistral-large-2407",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-72b-instruct",
        ]
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using ALL OpenRouter models in parallel."""
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.models)} OpenRouter models...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for model in self.models:
                future = executor.submit(self._call_api_safe, model, prompt)
                futures[future] = model
            
            completed = 0
            for future in as_completed(futures):
                model = futures[future]
                completed += 1
                try:
                    response = future.result(timeout=60)
                    if response:
                        selections = self.parse_response(response)
                        if selections:
                            for sel in selections:
                                sel["model"] = model
                            results.extend(selections)
                            successful_models.append(model)
                            logger.info(f"    ✅ {model}: {len(selections)} picks")
                        else:
                            logger.warning(f"    ⚠️ {model}: No valid selections")
                    else:
                        logger.warning(f"    ⚠️ {model}: Empty response")
                except Exception as e:
                    logger.warning(f"    ❌ {model}: {str(e)[:50]}...")
                
                if completed % 5 == 0:
                    logger.info(f"    Progress: {completed}/{len(self.models)} models done")
        
        logger.info(f"  ✅ {len(results)} total selections from {len(successful_models)} models")
        
        return self._aggregate_results(results, successful_models)
    
    def _call_api_safe(self, model: str, prompt: str) -> Optional[str]:
        """Call OpenRouter API with retries."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return self._call_api(model, prompt)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
    
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
                {"role": "system", "content": "You are a professional financial analyst with real-time market data access. Always respond in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 600,
        }
        
        response = requests.post(self.base_url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results from ALL models."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        # Count votes per ticker
        ticker_votes = {}
        ticker_data = {}
        total_responses = len(results)
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            if ticker:
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
                ticker_data[ticker]["models"].append(r.get("model", "unknown"))
        
        # Sort by votes (descending), then by average return
        sorted_tickers = sorted(
            ticker_votes.items(),
            key=lambda x: (x[1], sum(ticker_data[x[0]]["returns"]) / len(ticker_data[x[0]]["returns"])),
            reverse=True
        )
        
        # Take top N
        top_tickers = [t for t, _ in sorted_tickers[:self.top_n]]
        
        selections = []
        for ticker in top_tickers:
            data = ticker_data[ticker]
            avg_return = sum(data["returns"]) / len(data["returns"])
            
            # Most common confidence
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            # Unique models that voted for this ticker
            unique_models = list(set([m for m in data["models"] if m and m != 'unknown']))
            
            selections.append({
                "ticker": ticker,
                "expected_return": round(avg_return, 2),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": ticker_votes[ticker],  # Number of models that picked this ticker
                "models": unique_models
            })
        
        # All unique models used
        all_models_used = list(set(successful_models))
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": total_responses,
                "ticker_votes": ticker_votes,
                "models_used": all_models_used
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
            else:
                self.models = []
        except:
            self.models = []
            logger.warning("⚠️ Ollama not available")
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using Ollama models."""
        if not self.models:
            return {"selections": [], "consensus": {}}
        
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        successful_models = []
        
        logger.info(f"  Querying {len(self.models)} Ollama models...")
        
        for model in self.models:
            try:
                response = self._call_api(model, prompt)
                if response:
                    selections = self.parse_response(response)
                    if selections:
                        model_name = f"ollama/{model}"
                        for sel in selections:
                            sel["model"] = model_name
                        results.extend(selections)
                        successful_models.append(model_name)
                        logger.info(f"    ✅ ollama/{model}: {len(selections)} picks")
                    else:
                        logger.warning(f"    ⚠️ ollama/{model}: No valid selections")
                else:
                    logger.warning(f"    ⚠️ ollama/{model}: Empty response")
            except Exception as e:
                logger.warning(f"    ❌ ollama/{model}: {str(e)[:50]}...")
        
        logger.info(f"  ✅ {len(results)} total selections from {len(successful_models)} Ollama models")
        
        return self._aggregate_results(results, successful_models)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 600,
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["response"]
    
    def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
        """Aggregate results from Ollama models."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_votes = {}
        ticker_data = {}
        total_responses = len(results)
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            if ticker:
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
                ticker_data[ticker]["models"].append(r.get("model", "unknown"))
        
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
            
            unique_models = list(set([m for m in data["models"] if m and m != 'unknown']))
            
            selections.append({
                "ticker": ticker,
                "expected_return": round(avg_return, 2),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": ticker_votes[ticker],
                "models": unique_models
            })
        
        all_models_used = list(set(successful_models))
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": total_responses,
                "ticker_votes": ticker_votes,
                "models_used": all_models_used
            }
        }


class EnsembleAnalyzer:
    """Combine all LLM analyzers into one ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzers = []
        
        # OpenRouter (with API key)
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            self.analyzers.append(OpenRouterAnalyzer(config, openrouter_key))
            logger.info("✅ OpenRouter analyzer initialized")
        else:
            logger.warning("⚠️ OPENROUTER_API_KEY not set")
        
        # Ollama (local)
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
        
        # Aggregate all votes across all analyzers
        all_votes = {}
        all_data = {}
        all_models = set()
        
        for result in results:
            # Get selections from this analyzer
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
                    
                    # Collect models for this ticker
                    models = sel.get("models", [])
                    if isinstance(models, list):
                        all_data[ticker]["models"].extend(models)
                        all_models.update(models)
            
            # Also collect models from consensus
            consensus_models = result.get("consensus", {}).get("models_used", [])
            if isinstance(consensus_models, list):
                all_models.update(consensus_models)
        
        # Sort by votes (descending), then by average return
        sorted_votes = sorted(
            all_votes.items(),
            key=lambda x: (x[1], sum(all_data[x[0]]["returns"]) / len(all_data[x[0]]["returns"])),
            reverse=True
        )
        
        # Take top N
        top_tickers = [t for t, _ in sorted_votes[:self.config.get("TOP_N", 3)]]
        
        selections = []
        for ticker in top_tickers:
            data = all_data[ticker]
            avg_return = sum(data["returns"]) / len(data["returns"])
            
            # Most common confidence
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            # Unique models that voted for this ticker
            unique_models = list(set([m for m in data["models"] if m and m != 'unknown']))
            
            selections.append({
                "ticker": ticker,
                "expected_return": round(avg_return, 2),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": all_votes[ticker],  # Total votes across all analyzers
                "models": unique_models[:10]  # Show up to 10 models
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
