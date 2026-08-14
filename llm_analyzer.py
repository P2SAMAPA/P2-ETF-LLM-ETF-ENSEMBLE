"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Queries LLMs directly for market analysis and ETF picks.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """OpenRouter API implementation."""
    
    def __init__(self, config: Dict, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        # Updated with working model names
        self.models = config.get("OPENROUTER_MODELS", [
            "openai/gpt-4o-mini",  # Fast and cheap
            "meta-llama/llama-3.1-70b-instruct",  # Good quality
            "mistralai/mistral-7b-instruct",  # Fast
            "google/gemini-flash-1.5",  # Google's fast model
        ])
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def analyze(self, universe_name: str, tickers: List[str]) -> Dict:
        """Analyze using OpenRouter models."""
        prompt = self.build_prompt(universe_name, tickers)
        results = []
        
        # Try models in order until one works
        for model in self.models:
            try:
                logger.info(f"  Querying OpenRouter: {model}")
                response = self._call_api(model, prompt)
                selections = self.parse_response(response)
                if selections:
                    for sel in selections:
                        sel["model"] = model
                    results.extend(selections)
                    break  # Stop after first successful model
                else:
                    logger.warning(f"  {model} returned no selections")
            except Exception as e:
                logger.warning(f"  {model} failed: {str(e)[:50]}...")
                continue
        
        if not results:
            # Fallback: try all models one more time with lower temperature
            for model in self.models[:2]:
                try:
                    logger.info(f"  Retrying with {model} (fallback)")
                    response = self._call_api(model, prompt, temperature=0.5)
                    selections = self.parse_response(response)
                    if selections:
                        for sel in selections:
                            sel["model"] = f"{model}(fallback)"
                        results.extend(selections)
                        break
                except:
                    continue
        
        return self._aggregate_results(results)
    
    def _call_api(self, model: str, prompt: str, temperature: float = 0.3) -> str:
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
            "temperature": temperature,
            "max_tokens": 600,
        }
        
        response = requests.post(self.base_url, json=data, headers=headers, timeout=45)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple models."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_scores = {}
        ticker_data = {}
        
        for r in results:
            ticker = r.get("ticker", "").upper()
            if ticker:
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = 0
                    ticker_data[ticker] = {
                        "returns": [],
                        "confidences": [],
                        "rationales": [],
                        "models": []
                    }
                ticker_scores[ticker] += 1
                ticker_data[ticker]["returns"].append(r.get("expected_return", 0.5))
                ticker_data[ticker]["confidences"].append(r.get("confidence", "Medium"))
                ticker_data[ticker]["rationales"].append(r.get("rationale", ""))
                ticker_data[ticker]["models"].append(r.get("model", "unknown"))
        
        sorted_tickers = sorted(
            ticker_scores.items(),
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
                "votes": ticker_scores[ticker],
                "models": list(set(data["models"]))
            })
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": len(results),
                "ticker_scores": ticker_scores
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
        
        if not self.analyzers:
            logger.warning("⚠️ No LLM analyzers available")
    
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
        
        all_votes = {}
        all_data = {}
        
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
                    all_data[ticker]["models"].append(sel.get("model", "unknown"))
        
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
