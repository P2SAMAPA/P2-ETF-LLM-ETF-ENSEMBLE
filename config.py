"""
config.py  —  Configuration for LLM ETF Ensemble
"""

# HuggingFace credentials
HF_TOKEN = None  # Set via environment variable
RESULTS_REPO = "P2SAMAPA/p2-llm-etf-ensemble-results"

# Data source
MASTER_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
MASTER_DATA_FILE = "master_data.parquet"

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

# LLM Configuration
# OpenRouter
OPENROUTER_API_KEY = None  # Set via environment variable
OPENROUTER_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-exp",
    "microsoft/phi-3-medium-128k-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-large-2407",
]

# Ollama (local)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODELS = [
    "llama3.2:3b",
    "phi3:mini",
    "mistral:7b",
]

# Free APIs (no key required)
FREE_MODELS = [
    "cohere/command-r-plus",      # Free tier
    "deepseek/deepseek-chat",     # Free tier
]

# All models to use (in priority order)
ALL_MODELS = OPENROUTER_MODELS + OLLAMA_MODELS + FREE_MODELS

# Number of ETFs to select per universe
TOP_N = 3

# Run date
RUN_DATE = None  # Set dynamically
