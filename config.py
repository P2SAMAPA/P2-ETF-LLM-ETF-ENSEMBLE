"""
config.py  —  Configuration for LLM ETF Ensemble
"""

import os

# ---------------------------------------------------------------------------
# HuggingFace (results storage only — not used for market data)
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
RESULTS_REPO = "P2SAMAPA/p2-llm-etf-ensemble-results"

# ---------------------------------------------------------------------------
# Universes to analyze
# ---------------------------------------------------------------------------
UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD",
        "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO",
        "XLB", "XLRE"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD",
        "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO",
        "XLB", "XLRE"
    ]
}

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
# Lookback window used to compute momentum/volatility features that get
# handed to every LLM as grounding context (instead of letting them guess
# from ticker names alone).
MARKET_DATA_LOOKBACK = "6mo"

# ---------------------------------------------------------------------------
# OpenRouter — free models are discovered dynamically every run, see
# model_discovery.py. Nothing to hardcode here.
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Sent to OpenRouter for their leaderboard / rate-limit attribution. Update
# to your own repo if you fork this.
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://github.com/P2SAMAPA/P2-ETF-LLM-ETF-ENSEMBLE")
OPENROUTER_APP_NAME = "P2-LLM-ETF-ENSEMBLE"

# ---------------------------------------------------------------------------
# Ollama Cloud — models available to your account are discovered dynamically
# every run, see model_discovery.py.
# ---------------------------------------------------------------------------
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "https://api.ollama.com")

# ---------------------------------------------------------------------------
# Model discovery / execution limits
# ---------------------------------------------------------------------------
# Safety cap per provider so a sudden flood of new free models doesn't blow
# past the GitHub Actions job timeout. Set to 0 for "no cap".
MAX_MODELS_PER_PROVIDER = int(os.environ.get("MAX_MODELS_PER_PROVIDER", "25"))

# Model id/name substrings that disqualify a model from this text-reasoning
# task (vision-only, embeddings, moderation, audio, etc). Checked
# case-insensitively against the model id.
EXCLUDE_MODEL_KEYWORDS = [
    "embed", "moderation", "guard", "vision", "whisper", "tts",
    "image", "vae", "clip", "rerank", "audio",
]

# How many models can be queried concurrently, per provider.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))

# Per-request timeout and retry behaviour.
REQUEST_TIMEOUT_SECONDS = 45
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3

# ---------------------------------------------------------------------------
# Consensus / ranking
# ---------------------------------------------------------------------------
# Each model is asked to rank its top N picks per universe.
PICKS_PER_MODEL = 3

# Borda-style points awarded for each rank position (rank 1 = first index).
RANK_POINTS = [3, 2, 1]

# Confidence multiplier applied on top of rank points.
CONFIDENCE_WEIGHT = {"high": 1.15, "medium": 1.0, "low": 0.85}

# Final number of consensus picks reported per universe.
TOP_N = 3

# A ticker needs at least this many distinct models backing it to be
# eligible for the final list (falls back to "best available" if nothing
# clears the bar, so a quiet run still returns something).
MIN_VOTES = 2
