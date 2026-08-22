"""
llm_analyzer.py  —  LLM ETF Analysis Engine
============================================

Every currently-free OpenRouter model and every model available on your
Ollama account are queried in parallel. Each model ranks its top N ETFs
for the universe (grounded in real trailing price data, see
market_data.py). Results are combined into a single consensus ranking
using a Borda-style points system, weighted by each model's stated
confidence.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

import config
from market_data import format_snapshot_for_prompt
from model_discovery import get_available_ollama_models, get_free_openrouter_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_prompt(universe_name: str, tickers: List[str], market_table: str, picks_per_model: int) -> str:
    ticker_list = ", ".join(tickers)
    return f"""You are analyzing ETFs for the "{universe_name}" universe.

Available ETFs (you may ONLY pick from this exact list): {ticker_list}

Recent price data (trailing returns, not a prediction — use it as context):
{market_table}

Rank your top {picks_per_model} ETF picks from the list above, best first.
Base your ranking on the price data shown plus your general knowledge of
what each ETF/sector represents. Do not pick anything outside the list.

Respond with ONLY this JSON structure and nothing else:
{{"picks": [
  {{"rank": 1, "ticker": "TICK", "confidence": "High", "rationale": "one short sentence"}},
  {{"rank": 2, "ticker": "TICK", "confidence": "Medium", "rationale": "one short sentence"}},
  {{"rank": 3, "ticker": "TICK", "confidence": "Medium", "rationale": "one short sentence"}}
]}}"""


def parse_response(text: str, valid_tickers: List[str], picks_per_model: int) -> Optional[List[Dict]]:
    """Parse the model's JSON, keep only valid/well-formed/deduplicated picks."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        raw_picks = data.get("picks", [])
    except Exception:
        return None

    seen = set()
    picks = []
    for p in raw_picks:
        ticker = str(p.get("ticker", "")).upper().strip()
        rank = p.get("rank")
        if ticker in seen or ticker not in valid_tickers:
            continue
        if not isinstance(rank, int) or not (1 <= rank <= picks_per_model):
            continue
        seen.add(ticker)
        picks.append({
            "rank": rank,
            "ticker": ticker,
            "confidence": str(p.get("confidence", "Medium")).title(),
            "rationale": str(p.get("rationale", ""))[:300],
        })

    picks.sort(key=lambda x: x["rank"])
    return picks or None


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

def _call_with_retries(fn, max_retries: int, backoff: int):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff * (attempt + 1))
    raise last_err


def query_openrouter(model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.OPENROUTER_SITE_URL,
        "X-Title": config.OPENROUTER_APP_NAME,
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }

    def do_call():
        r = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers, json=body, timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        if r.status_code == 429:
            raise RuntimeError("rate limited")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return _call_with_retries(do_call, config.MAX_RETRIES, config.RETRY_BACKOFF_SECONDS)


def query_ollama(model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {config.OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"model": model, "prompt": prompt, "stream": False, "temperature": 0.3}

    def do_call():
        r = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            headers=headers, json=body, timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json().get("response", "")

    return _call_with_retries(do_call, config.MAX_RETRIES, config.RETRY_BACKOFF_SECONDS)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleAnalyzer:
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.openrouter_models = get_free_openrouter_models(
            cfg.OPENROUTER_BASE_URL, cfg.EXCLUDE_MODEL_KEYWORDS, cfg.MAX_MODELS_PER_PROVIDER,
        ) if cfg.OPENROUTER_API_KEY else []
        self.ollama_models = get_available_ollama_models(
            cfg.OLLAMA_BASE_URL, cfg.OLLAMA_API_KEY, cfg.EXCLUDE_MODEL_KEYWORDS, cfg.MAX_MODELS_PER_PROVIDER,
        )

        if not cfg.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set — skipping OpenRouter models")

        logger.info(
            f"Ensemble ready: {len(self.openrouter_models)} OpenRouter (free) + "
            f"{len(self.ollama_models)} Ollama models"
        )

    def all_models(self) -> List[str]:
        return [f"openrouter/{m}" for m in self.openrouter_models] + \
               [f"ollama/{m}" for m in self.ollama_models]

    def _query_one(self, tagged_model: str, prompt: str, valid_tickers: List[str]) -> Optional[Dict]:
        provider, model = tagged_model.split("/", 1)
        try:
            if provider == "openrouter":
                raw = query_openrouter(model, prompt)
            else:
                raw = query_ollama(model, prompt)
        except Exception as e:
            logger.warning(f"  x {tagged_model}: {str(e)[:80]}")
            return None

        picks = parse_response(raw, valid_tickers, self.cfg.PICKS_PER_MODEL)
        if not picks:
            logger.warning(f"  x {tagged_model}: unparseable/invalid response")
            return None

        logger.info(f"  ok {tagged_model}: {[p['ticker'] for p in picks]}")
        return {"model": tagged_model, "picks": picks}

    def analyze_universe(self, universe_name: str, tickers: List[str], market_snapshot: Dict) -> Dict:
        tagged_models = self.all_models()
        if not tagged_models:
            return {"selections": [], "consensus": {}}

        market_table = format_snapshot_for_prompt(market_snapshot, tickers)
        prompt = build_prompt(universe_name, tickers, market_table, self.cfg.PICKS_PER_MODEL)

        responses = []
        with ThreadPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._query_one, m, prompt, tickers): m for m in tagged_models
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    responses.append(result)

        return self._build_consensus(responses, tickers, tagged_models, market_snapshot)

    def _build_consensus(self, responses: List[Dict], valid_tickers: List[str],
                          tagged_models: List[str], market_snapshot: Dict) -> Dict:
        if not responses:
            return {
                "selections": [],
                "consensus": {
                    "models_queried": len(tagged_models),
                    "models_responded": 0,
                    "ticker_points": {},
                    "ticker_votes": {},
                    "models_used": [],
                },
            }

        points: Dict[str, float] = {}
        votes: Dict[str, int] = {}
        confidences: Dict[str, List[str]] = {}
        rationales: Dict[str, List[str]] = {}
        models_by_ticker: Dict[str, List[str]] = {}

        rank_points = self.cfg.RANK_POINTS
        conf_weight = self.cfg.CONFIDENCE_WEIGHT

        for resp in responses:
            model = resp["model"]
            for pick in resp["picks"]:
                ticker = pick["ticker"]
                rank_idx = pick["rank"] - 1
                base_points = rank_points[rank_idx] if rank_idx < len(rank_points) else 1
                weight = conf_weight.get(pick["confidence"].lower(), 1.0)

                points[ticker] = points.get(ticker, 0) + base_points * weight
                votes[ticker] = votes.get(ticker, 0) + 1
                confidences.setdefault(ticker, []).append(pick["confidence"])
                rationales.setdefault(ticker, []).append(pick["rationale"])
                models_by_ticker.setdefault(ticker, []).append(model)

        qualified = {t: p for t, p in points.items() if votes.get(t, 0) >= self.cfg.MIN_VOTES}
        pool_for_ranking = qualified if qualified else points

        ranked = sorted(
            pool_for_ranking.items(),
            key=lambda kv: (kv[1], votes.get(kv[0], 0)),
            reverse=True,
        )[: self.cfg.TOP_N]

        selections = []
        for ticker, pts in ranked:
            conf_counts: Dict[str, int] = {}
            for c in confidences[ticker]:
                conf_counts[c] = conf_counts.get(c, 0) + 1
            top_confidence = max(conf_counts, key=conf_counts.get)

            md = market_snapshot.get(ticker, {})
            selections.append({
                "ticker": ticker,
                "points": round(pts, 2),
                "votes": votes[ticker],
                "confidence": top_confidence,
                "rationale": rationales[ticker][0] if rationales[ticker] else "",
                "models": sorted(set(models_by_ticker[ticker])),
                # Real trailing return, not a forecast — kept as
                # "expected_return" for dashboard compatibility.
                "expected_return": md.get("return_1m"),
                "return_3m": md.get("return_3m"),
                "annualized_volatility_pct": md.get("annualized_volatility_pct"),
            })

        return {
            "selections": selections,
            "consensus": {
                "models_queried": len(tagged_models),
                "models_responded": len(responses),
                "ticker_points": {t: round(p, 2) for t, p in points.items()},
                "ticker_votes": votes,
                "models_used": sorted(set(r["model"] for r in responses)),
            },
        }
