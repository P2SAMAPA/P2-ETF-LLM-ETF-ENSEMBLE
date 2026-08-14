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

# ---------- OpenRouter (Free models) ----------
# Get your API key from: https://openrouter.ai/keys
# Free models available with no payment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# ---------- Groq (Completely free, no API key needed) ----------
# Optional: Get a free API key from: https://console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------- General Settings ----------
TOP_N = 3


# ============================================
# END OF CONFIGURATION
# ============================================

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
        print("✅ OpenRouter: Enabled (free models)")
    else:
        print("❌ OpenRouter: Disabled (no API key)")
    
    print("✅ Groq: Enabled (free, no API key needed)")
    print("="*50 + "\n")


if __name__ == "__main__":
    print_config()
