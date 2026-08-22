# P2-LLM-ETF-ENSEMBLE

## LLM-Powered ETF Selection Engine

This engine queries **every free model currently available** on OpenRouter and
every model on your Ollama account, has each one rank its top ETF picks for
a universe, and combines the results into a single consensus ranking.

### How It Works

1. **Discover models**: `model_discovery.py` calls OpenRouter's public
   `/models` endpoint and filters for free models (`pricing == 0` or a
   `:free` id suffix), and calls your Ollama account's `/api/tags` for
   whatever models you currently have available. This runs fresh **every
   time the workflow fires** — no hardcoded model list to maintain. If a
   model gets retired it just quietly drops out of the next run; if a new
   free model shows up, it's queried automatically next time.
2. **Fetch market data**: `market_data.py` pulls trailing 1/3/6-month
   returns and annualized volatility for every ticker via `yfinance`, so
   each model is reasoning over real, current numbers instead of guessing
   from ticker names and training-data priors.
3. **LLM analysis**: Every discovered model ranks its top 3 ETFs per
   universe (`llm_analyzer.py`), run concurrently with retries and
   per-call timeouts.
4. **Ensemble consensus**: Picks are combined with a Borda-style points
   system (rank 1 > rank 2 > rank 3), weighted by each model's stated
   confidence. A ticker needs at least `MIN_VOTES` distinct models behind
   it to be eligible, so a single outlier model can't win a universe.
5. **Results**: Saved locally as `llm_etf_ensemble_<date>.json` and (if
   `HF_TOKEN` is set) uploaded to a HuggingFace dataset for the dashboard
   to read.

### What counts as "free"

- **OpenRouter**: a model whose id ends in `:free`, or whose prompt/output
  pricing is exactly `0`.
- **Ollama**: whatever your Ollama Cloud account currently has access to
  (no separate free/paid split at the API level — set `OLLAMA_API_KEY`
  for a free-tier account and that's what gets used).

Model lists, prices, and availability change over time — check
`https://openrouter.ai/models?max_price=0` and your Ollama account
directly if you want to see exactly what will be queried on the next run.

### Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-LLM-ETF-ENSEMBLE
cd P2-ETF-LLM-ETF-ENSEMBLE
pip install -r requirements.txt

# At least one of these is required
export OPENROUTER_API_KEY="your-openrouter-key"   # free, from openrouter.ai/keys
export OLLAMA_API_KEY="your-ollama-cloud-key"      # from ollama.com

# Optional
export HF_TOKEN="your-hf-token"                    # to publish results
export MAX_MODELS_PER_PROVIDER=25                  # safety cap per provider
export MAX_WORKERS=8                                # concurrent requests

python trainer.py
```

### Dashboard

```bash
streamlit run streamlit_app.py
```

Reads the latest results from HuggingFace (with a local-file fallback), and
shows per-universe consensus picks, model/provider breakdowns, and a
cross-universe summary.

### Configuration

All tunables live in `config.py`: universes/tickers, market-data lookback
window, per-provider model caps, request timeouts/retries, and the
consensus scoring (`RANK_POINTS`, `CONFIDENCE_WEIGHT`, `MIN_VOTES`,
`TOP_N`).

### Automation

`.github/workflows/daily_run.yml` runs the pipeline Monday–Saturday at
00:30 UTC via GitHub Actions, using repo secrets for the API keys. Because
model discovery happens at runtime, there's nothing to update in this repo
when providers add or retire free models.
