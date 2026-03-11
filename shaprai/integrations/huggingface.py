"""HuggingFace integration for model management.

Handles downloading, caching, and loading base models from the
HuggingFace Hub for agent training and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default cache directory for downloaded models
DEFAULT_CACHE_DIR = Path.home() / ".shaprai" / "models"

# Recommended models for Elyan-class agents by size tier
RECOMMENDED_MODELS = {
    "tiny": [
        {"id": "Qwen/Qwen3-0.6B-Instruct", "vram_gb": 1, "description": "Tiny agent, edge deployment"},
    ],
    "small": [
        {"id": "Qwen/Qwen3-1.7B-Instruct", "vram_gb": 2, "description": "Small agent, fast inference"},
        {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "vram_gb": 2, "description": "Efficient small model"},
    ],
    "medium": [
        {"id": "Qwen/Qwen3-7B-Instruct", "vram_gb": 6, "description": "Standard Elyan-class agent"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "vram_gb": 6, "description": "Strong reasoning"},
    ],
    "large": [
        {"id": "Qwen/Qwen3-14B-Instruct", "vram_gb": 10, "description": "Enhanced capabilities"},
        {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "vram_gb": 24, "description": "MoE, broad knowledge"},
    ],
    "xl": [
        {"id": "Qwen/Qwen3-32B-Instruct", "vram_gb": 20, "description": "Near-frontier performance"},
    ],
}


def load_base_model(
    model_id: str,
    quantize: bool = True,
    cache_dir: Optional[Path] = None,
) -> Any:
    """Load a base model from HuggingFace for training or inference.

    Args:
        model_id: HuggingFace model identifier (e.g., 'Qwen/Qwen3-7B-Instruct').
        quantize: Whether to load in 4-bit quantization (QLoRA-ready).
        cache_dir: Local cache directory. Defaults to ~/.shaprai/models.

    Returns:
        Loaded model object (AutoModelForCausalLM).

    Raises:
        ImportError: If transformers or bitsandbytes not installed.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s (quantize=%s)", model_id, quantize)

        load_kwargs: Dict[str, Any] = {
            "cache_dir": str(cache_dir),
            "trust_remote_code": True,
        }

        if quantize:
            try:
                from transformers import BitsAndBytesConfig
                import torch

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not available -- loading without quantization")

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        logger.info("Model loaded successfully: %s", model_id)
        return model

    except ImportError as e:
        raise ImportError(
            f"Required package not installed: {e}. "
            "Install with: pip install shaprai[training]"
        ) from e


def load_tokenizer(
    model_id: str,
    cache_dir: Optional[Path] = None,
) -> Any:
    """Load a tokenizer for a model.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Local cache directory.

    Returns:
        Loaded tokenizer object.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )


def list_compatible_models(
    size_filter: Optional[str] = None,
    max_vram_gb: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List models compatible with ShaprAI agent training.

    Args:
        size_filter: Filter by size tier (tiny, small, medium, large, xl).
        max_vram_gb: Maximum VRAM budget in GB.

    Returns:
        List of compatible model specifications.
    """
    results: List[Dict[str, Any]] = []

    tiers = [size_filter] if size_filter and size_filter in RECOMMENDED_MODELS else RECOMMENDED_MODELS.keys()

    for tier in tiers:
        for model in RECOMMENDED_MODELS.get(tier, []):
            if max_vram_gb is not None and model["vram_gb"] > max_vram_gb:
                continue
            results.append({**model, "tier": tier})

    return results


def download_model(
    model_id: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download a model to local cache without loading it.

    Useful for pre-caching models before training.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Local cache directory.

    Returns:
        Path to the cached model directory.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
        )
        logger.info("Model downloaded to: %s", local_dir)
        return Path(local_dir)

    except ImportError:
        from transformers import AutoModelForCausalLM

        # Fallback: use transformers download
        AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        return cache_dir / model_id.replace("/", "--")
