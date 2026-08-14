# P2-LLM-ETF-ENSEMBLE

## LLM-Powered ETF Selection Engine

This engine uses **multiple LLMs** (via OpenRouter and Ollama) to analyze market data and select the top ETFs for each universe.

### How It Works

1. **Load Data**: Fetches ETF prices and macro data from HuggingFace
2. **LLM Analysis**: Each LLM analyzes the data and picks top 3 ETFs per universe
3. **Ensemble Voting**: Aggregates all LLM outputs into consensus rankings
4. **Results**: Saves and uploads JSON with top picks and confidence scores

### Supported LLM Providers

- **OpenRouter**: Claude 3.5 Sonnet, Gemini 2.0 Flash, LLaMA 3.1, Mistral Large
- **Ollama**: Local models (llama3.2, phi3, mistral)
- **Free APIs**: Cohere Command R+, DeepSeek

### Setup

```bash
# Clone and install
git clone https://github.com/P2SAMAPA/P2-LLM-ETF-ENSEMBLE
cd P2-LLM-ETF-ENSEMBLE
pip install -r requirements.txt

# Set API keys (for OpenRouter)
export OPENROUTER_API_KEY="your-key-here"
export HF_TOKEN="your-hf-token"

# Run
python trainer.py
