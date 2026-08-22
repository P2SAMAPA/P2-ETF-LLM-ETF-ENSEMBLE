"""
model_discovery.py  —  Dynamic discovery of free LLMs
=======================================================

Both provider model lists are fetched fresh on every run instead of being
hardcoded. This means:

  * A model that gets retired just silently drops out of the next run
    (no crash — the earlier hardcoded list would eventually 404/expire).
  * A newly released free model is picked up automatically the next time
    the workflow runs, with zero code changes.

OpenRouter: GET /api/v1/models is public (no auth needed) and includes a
`pricing` object per model. A model is treated as free if its id ends in
":free" (OpenRouter's own convention) or if both prompt/completion pricing
are exactly zero.

Ollama Cloud: GET /api/tags (with your account's bearer token) returns the
models currently available to you. There's no separate "free" tier concept
on the API itself — whatever your account can see is what gets used.
"""

import logging
from typing import List

import requests

logger = logging.getLogger(__name__)


def _looks_usable(model_id: str, exclude_keywords: List[str]) -> bool:
    lower = model_id.lower()
    return not any(kw in lower for kw in exclude_keywords)


def get_free_openrouter_models(
    base_url: str,
    exclude_keywords: List[str],
    max_models: int = 25,
    timeout: int = 15,
) -> List[str]:
    """Return the ids of currently-free OpenRouter models."""
    url = f"{base_url}/models"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        models = r.json().get("data", [])
    except Exception as e:
        logger.warning(f"OpenRouter model list fetch failed: {e}")
        return []

    free = []
    for m in models:
        model_id = m.get("id", "")
        if not model_id:
            continue
        pricing = m.get("pricing", {}) or {}
        try:
            prompt_price = float(pricing.get("prompt", 1) or 0)
            completion_price = float(pricing.get("completion", 1) or 0)
        except (TypeError, ValueError):
            prompt_price, completion_price = 1.0, 1.0

        is_free = model_id.endswith(":free") or (prompt_price == 0.0 and completion_price == 0.0)
        if is_free and _looks_usable(model_id, exclude_keywords):
            free.append(model_id)

    free = sorted(set(free))
    if max_models:
        free = free[:max_models]
    logger.info(f"OpenRouter free models discovered: {len(free)}")
    return free


def get_available_ollama_models(
    base_url: str,
    api_key: str,
    exclude_keywords: List[str],
    max_models: int = 25,
    timeout: int = 15,
) -> List[str]:
    """Return the model tags currently available to this Ollama account."""
    if not api_key:
        return []

    url = f"{base_url}/api/tags"
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", []) if m.get("name")]
    except Exception as e:
        logger.warning(f"Ollama model list fetch failed: {e}")
        return []

    usable = sorted(set(m for m in models if _looks_usable(m, exclude_keywords)))
    if max_models:
        usable = usable[:max_models]
    logger.info(f"Ollama models discovered: {len(usable)}")
    return usable
