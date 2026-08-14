"""
config.py  —  Configuration for LLM ETF Ensemble
"""

import os

# HuggingFace for results only
HF_TOKEN = os.environ.get("HF_TOKEN")
RESULTS_REPO = "P2SAMAPA/p2-llm-etf-ensemble-results"

# Universes to analyze
UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", 
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", 
        "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", 
        "XLB", "XLRE"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", 
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", 
        "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", 
        "XLB", "XLRE"
    ]
}

# ============================================
# LLM CONFIGURATION
# ============================================

# ---------- OpenRouter (Cloud) ----------
# Get your API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# OpenRouter models (working with your plan)
OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
]

# ---------- Ollama (Local/Remote) ----------
# Ollama URL (default: http://localhost:11434)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Ollama API Key (only if using remote/cloud Ollama)
# For local Ollama, leave this as None
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", None)

# Ollama models to try (will auto-detect available ones)
OLLAMA_MODELS = [
    "llama3.2:3b",
    "phi3:mini",
    "mistral:7b",
    "llama3.1:8b",
]

# ---------- General Settings ----------
# Number of ETFs to select per universe
TOP_N = 3

# ============================================
# END OF CONFIGURATION
# ============================================

# Print configuration summary
def print_config():
    """Print current configuration."""
    print("\n" + "="*50)
    print("LLM ETF ENSEMBLE CONFIGURATION")
    print("="*50)
    print(f"Universes: {list(UNIVERSES.keys())}")
    print(f"Top N: {TOP_N}")
    print(f"Results Repo: {RESULTS_REPO}")
    print("\n--- LLM Providers ---")
    
    if OPENROUTER_API_KEY:
        print(f"✅ OpenRouter: Enabled ({len(OPENROUTER_MODELS)} models)")
        for model in OPENROUTER_MODELS:
            print(f"   - {model}")
    else:
        print("❌ OpenRouter: Disabled (no API key)")
    
    print(f"✅ Ollama: URL={OLLAMA_URL}")
    print(f"   Models: {OLLAMA_MODELS}")
    if OLLAMA_API_KEY:
        print("   🔑 API Key: Set (remote mode)")
    else:
        print("   🔑 API Key: Not set (local mode)")
    print("="*50 + "\n")

# Auto-print config if run directly
if __name__ == "__main__":
    print_config()
