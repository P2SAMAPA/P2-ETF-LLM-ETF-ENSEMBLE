"""
results_store.py  —  Persist ensemble results locally and to HuggingFace
"""

import json
import logging
from typing import Dict

import config

logger = logging.getLogger(__name__)


def save_and_upload(results: Dict, run_date: str) -> str:
    output_path = f"llm_etf_ensemble_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved: {output_path}")

    if config.HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=config.HF_TOKEN)
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=output_path,
                repo_id=config.RESULTS_REPO,
                token=config.HF_TOKEN,
                repo_type="dataset",
            )
            logger.info("Uploaded to HuggingFace")
        except Exception as e:
            logger.error(f"HuggingFace upload failed: {e}")
    else:
        logger.info("HF_TOKEN not set — skipping upload, local file only")

    return output_path
