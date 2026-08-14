"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Uses multiple LLMs to analyze market data and pick top ETFs.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Base class for LLM ETF analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.top_n = config.get("TOP_N", 3)
        
    def build_prompt(self, universe_name: str, tickers: List[str], 
                     data_summary: Dict) -> str:
        """Build the prompt for LLM analysis."""
        
        prompt = f"""You are a professional financial analyst. Analyze the following ETFs and select the top {self.top_n} that are most likely to outperform in the next 1-3 months.

UNIVERSE: {universe_name}
ETFS: {', '.join(tickers)}

MARKET DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

MACRO CONTEXT:
- Fed Policy: Recent signals and expectations
- Inflation Trends: Current CPI/PCE readings
- Economic Growth: GDP, employment, consumer spending
- Market Sentiment: VIX, put/call ratios, fund flows
- Global Risks: Geopolitical, trade, commodity prices

For each of the top {self.top_n} ETFs, provide:
1. Ticker symbol
2. Probability of positive return (%)
3. Confidence level (High/Medium/Low)
4. Brief rationale (1-2 sentences)

Format your response as JSON:
{{
    "selections": [
        {{
            "ticker": "GDX",
            "probability": 75.5,
            "confidence": "High",
            "rationale": "Gold miners benefit from Fed rate cuts and safe-haven demand."
        }}
    ]
}}

Return ONLY the JSON, no other text.
"""
        return prompt
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response into structured selections."""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("selections", [])
        except:
            pass
        
        # Fallback: try to parse manually
        selections = []
        lines = response_text.strip().split('\n')
        for line in lines:
            if 'ticker' in line.lower() or 'etf' in line.lower():
                # Attempt to extract ticker and probability
                parts = line.split('|')
                if len(parts) >= 2:
                    ticker = parts[0].strip().split(':')[-1].strip()[:5]
                    prob = 50.0
                    conf = "Medium"
                    selections.append({
                        "ticker": ticker,
                        "probability": prob,
                        "confidence": conf,
                        "rationale": line.strip()
                    })
        
        return selections[:self.top_n]


class OpenRouterAnalyzer(LLMAnalyzer):
    """OpenRouter API implementation."""
    
    def __init__(self, config: Dict, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.models = config.get("OPENROUTER_MODELS", [])
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def analyze(self, universe_name: str, tickers: List[str], 
                data_summary: Dict) -> Dict:
        """Analyze using OpenRouter models."""
        prompt = self.build_prompt(universe_name, tickers, data_summary)
        results = []
        
        for model in self.models[:2]:  # Use top 2 models for speed
            try:
                response = self._call_api(model, prompt)
                selections = self.parse_response(response)
                for sel in selections:
                    sel["model"] = model
                results.extend(selections)
            except Exception as e:
                logger.error(f"OpenRouter {model} failed: {e}")
        
        return self._aggregate_results(results)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        
        response = requests.post(self.base_url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple models."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        # Count votes per ticker
        ticker_votes = {}
        ticker_data = {}
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            if ticker:
                ticker_votes[ticker] = ticker_votes.get(ticker, 0) + 1
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "probabilities": [],
                        "confidences": [],
                        "rationales": []
                    }
                ticker_data[ticker]["probabilities"].append(r.get("probability", 50))
                ticker_data[ticker]["confidences"].append(r.get("confidence", "Medium"))
                ticker_data[ticker]["rationales"].append(r.get("rationale", ""))
        
        # Sort by votes
        sorted_votes = sorted(ticker_votes.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [t for t, _ in sorted_votes[:self.top_n]]
        
        # Build consensus
        selections = []
        for ticker in top_tickers:
            data = ticker_data[ticker]
            avg_prob = np.mean(data["probabilities"])
            # Most common confidence
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            selections.append({
                "ticker": ticker,
                "probability": round(avg_prob, 1),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": ticker_votes[ticker]
            })
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": len(results),
                "ticker_votes": ticker_votes
            }
        }


class OllamaAnalyzer(LLMAnalyzer):
    """Ollama (local) implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("OLLAMA_URL", "http://localhost:11434")
        self.models = config.get("OLLAMA_MODELS", [])
    
    def analyze(self, universe_name: str, tickers: List[str], 
                data_summary: Dict) -> Dict:
        """Analyze using Ollama models."""
        prompt = self.build_prompt(universe_name, tickers, data_summary)
        results = []
        
        for model in self.models[:2]:  # Use top 2 models
            try:
                response = self._call_api(model, prompt)
                selections = self.parse_response(response)
                for sel in selections:
                    sel["model"] = model
                results.extend(selections)
            except Exception as e:
                logger.error(f"Ollama {model} failed: {e}")
        
        return self._aggregate_results(results)
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 500,
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["response"]
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Same aggregation as OpenRouter."""
        # Reuse same aggregation logic
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_votes = {}
        ticker_data = {}
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            if ticker:
                ticker_votes[ticker] = ticker_votes.get(ticker, 0) + 1
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "probabilities": [],
                        "confidences": [],
                        "rationales": []
                    }
                ticker_data[ticker]["probabilities"].append(r.get("probability", 50))
                ticker_data[ticker]["confidences"].append(r.get("confidence", "Medium"))
                ticker_data[ticker]["rationales"].append(r.get("rationale", ""))
        
        sorted_votes = sorted(ticker_votes.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [t for t, _ in sorted_votes[:self.top_n]]
        
        selections = []
        for ticker in top_tickers:
            data = ticker_data[ticker]
            avg_prob = np.mean(data["probabilities"])
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            selections.append({
                "ticker": ticker,
                "probability": round(avg_prob, 1),
                "confidence": top_conf,
                "rationale": data["rationales"][0] if data["rationales"] else "",
                "votes": ticker_votes[ticker]
            })
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": len(results),
                "ticker_votes": ticker_votes
            }
        }


class EnsembleAnalyzer:
    """Combine all LLM analyzers into one ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzers = []
        
        # OpenRouter
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            self.analyzers.append(OpenRouterAnalyzer(config, openrouter_key))
            logger.info("✅ OpenRouter analyzer initialized")
        
        # Ollama
        try:
            response = requests.get(f"{config.get('OLLAMA_URL', 'http://localhost:11434')}/api/tags", timeout=2)
            if response.status_code == 200:
                self.analyzers.append(OllamaAnalyzer(config))
                logger.info("✅ Ollama analyzer initialized")
        except:
            logger.warning("⚠️ Ollama not available")
        
        if not self.analyzers:
            logger.warning("⚠️ No LLM analyzers available")
    
    def analyze_universe(self, universe_name: str, tickers: List[str], 
                         data_summary: Dict) -> Dict:
        """Run all analyzers on a universe."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=len(self.analyzers)) as executor:
            futures = []
            for analyzer in self.analyzers:
                future = executor.submit(
                    analyzer.analyze, universe_name, tickers, data_summary
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    if result.get("selections"):
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Analyzer failed: {e}")
        
        return self._ensemble_vote(all_results)
    
    def _ensemble_vote(self, results: List[Dict]) -> Dict:
        """Vote across all analyzer results."""
        if not results:
            return {"selections": [], "ensemble_stats": {}}
        
        # Weighted voting: each analyzer gets 1 vote per selection
        all_votes = {}
        all_data = {}
        
        for result in results:
            for sel in result.get("selections", []):
                ticker = sel.get("ticker", "").upper()
                if ticker:
                    all_votes[ticker] = all_votes.get(ticker, 0) + 1
                    if ticker not in all_data:
                        all_data[ticker] = {
                            "probabilities": [],
                            "confidences": [],
                            "rationales": [],
                            "models": []
                        }
                    all_data[ticker]["probabilities"].append(sel.get("probability", 50))
                    all_data[ticker]["confidences"].append(sel.get("confidence", "Medium"))
                    all_data[ticker]["rationales"].append(sel.get("rationale", ""))
                    all_data[ticker]["models"].append(sel.get("model", "unknown"))
        
        # Sort by votes
        sorted_votes = sorted(all_votes.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [t for t, _ in sorted_votes[:self.config.get("TOP_N", 3)]]
        
        selections = []
        for ticker in top_tickers:
            data = all_data[ticker]
            avg_prob = np.mean(data["probabilities"])
            
            # Most common confidence
            conf_counts = {}
            for c in data["confidences"]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_conf = max(conf_counts, key=conf_counts.get)
            
            # Most common rationale (first occurrence)
            rationale = data["rationales"][0] if data["rationales"] else ""
            
            selections.append({
                "ticker": ticker,
                "probability": round(avg_prob, 1),
                "confidence": top_conf,
                "rationale": rationale,
                "votes": all_votes[ticker],
                "models": list(set(data["models"]))
            })
        
        return {
            "selections": selections,
            "ensemble_stats": {
                "total_analyzers": len(results),
                "total_votes": sum(all_votes.values()),
                "ticker_votes": all_votes
            }
        }
