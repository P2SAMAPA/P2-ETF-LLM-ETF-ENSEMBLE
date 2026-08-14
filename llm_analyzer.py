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
                selections = data.get("selections", [])
                if selections:
                    return selections
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
        all_models_used = []
        
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
                            # Add model name to each selection
                            for sel in selections:
                                sel["model"] = model
                            results.extend(selections)
                            all_models_used.append(model)
                            logger.info(f"    ✅ {model}: {len(selections)} picks")
                        else:
                            logger.warning(f"    ⚠️ {model}: No valid selections")
                    else:
                        logger.warning(f"    ⚠️ {model}: Empty response")
                except Exception as e:
                    logger.warning(f"    ❌ {model}: {str(e)[:50]}...")
                
                if completed % 5 == 0:
                    logger.info(f"    Progress: {completed}/{len(self.models)} models done")
        
        logger.info(f"  ✅ {len(results)} total selections from {len(all_models_used)} models")
        return self._aggregate_results(results, all_models_used)
    
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
    
    def _aggregate_results(self, results: List[Dict], all_models_used: List[str]) -> Dict:
        """Aggregate results from ALL models."""
        if not results:
            return {"selections": [], "consensus": {}}
        
        ticker_scores = {}
        ticker_data = {}
        total_responses = len(results)
        
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
                "models": list(set(data["models"]))  # Unique models that voted for this ticker
            })
        
        return {
            "selections": selections,
            "consensus": {
                "total_votes": total_responses,
                "ticker_scores": ticker_scores,
                "models_used": list(set(all_models_used))  # All unique models used
            }
        }


class EnsembleAnalyzer:
    """Combine all LLM analyzers into one ensemble."""
    
    def __init
