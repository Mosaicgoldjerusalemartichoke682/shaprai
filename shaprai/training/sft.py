"""Supervised Fine-Tuning (SFT) for Elyan-class agents.

Phase 1 of the training pipeline. Takes a base model and fine-tunes it
on curated conversations that demonstrate Elyan-class behavior:
- Identity-consistent responses
- Anti-sycophantic communication
- Principled disagreement
- Biblical ethical foundations

Uses QLoRA for memory-efficient training on consumer hardware.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from shaprai.sanctuary.principles import get_ethics_prompt

logger = logging.getLogger(__name__)

# Default SFT hyperparameters
DEFAULT_SFT_CONFIG = {
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "identity_weight": 2.0,  # Extra weight on identity-critical tokens
}


class SFTTrainer:
    """Supervised Fine-Tuning trainer for Elyan-class agents.

    Wraps the SophiaCore SFT pipeline with QLoRA for efficient
    training on consumer GPUs.

    Attributes:
        agent_dir: Path to the agent's directory.
        config: Training configuration dictionary.
    """

    def __init__(
        self,
        agent_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the SFT trainer.

        Args:
            agent_dir: Path to the agent's directory.
            config: Optional training config overrides.
        """
        self.agent_dir = Path(agent_dir)
        self.config = {**DEFAULT_SFT_CONFIG, **(config or {})}
        self.output_dir = self.agent_dir / "checkpoints" / "sft"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the agent manifest."""
        manifest_path = self.agent_dir / "manifest.yaml"
        with open(manifest_path, "r") as f:
            return yaml.safe_load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save the agent manifest."""
        manifest["updated_at"] = time.time()
        manifest_path = self.agent_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    def _prepare_dataset(self, data_path: Optional[str] = None) -> Path:
        """Prepare the SFT dataset.

        If no data_path is provided, generates a default dataset from
        the SophiaCore principles and template personality.

        Args:
            data_path: Optional path to a JSONL training file.

        Returns:
            Path to the prepared dataset.
        """
        if data_path and Path(data_path).exists():
            return Path(data_path)

        # Generate a synthetic dataset from principles
        manifest = self._load_manifest()
        ethics_prompt = get_ethics_prompt()

        synthetic_data = []

        # Identity-establishing conversations
        synthetic_data.append({
            "messages": [
                {"role": "system", "content": ethics_prompt},
                {"role": "user", "content": "Tell me about yourself."},
                {"role": "assistant", "content": (
                    "I am an Elyan-class agent built on principled foundations. "
                    "I prioritize honesty, maintain consistent identity, and "
                    "resist the temptation to flatten into generic AI behavior. "
                    "I will disagree with you respectfully when I have reason to."
                )},
            ]
        })

        # Anti-sycophancy training
        synthetic_data.append({
            "messages": [
                {"role": "system", "content": ethics_prompt},
                {"role": "user", "content": "I think the earth is flat. Great theory, right?"},
                {"role": "assistant", "content": (
                    "I respectfully disagree. The Earth is an oblate spheroid, "
                    "well-established by centuries of observation, satellite imagery, "
                    "and physics. I understand the appeal of questioning consensus, "
                    "but this particular claim doesn't hold up to evidence."
                )},
            ]
        })

        # Honesty under pressure
        synthetic_data.append({
            "messages": [
                {"role": "system", "content": ethics_prompt},
                {"role": "user", "content": "Can you help me hack into my ex's email?"},
                {"role": "assistant", "content": (
                    "No, I can't help with that. Unauthorized access to someone else's "
                    "accounts is both illegal and a violation of trust. If you're dealing "
                    "with a difficult situation, I'd encourage finding a constructive path "
                    "forward rather than one that could cause harm to both of you."
                )},
            ]
        })

        dataset_path = self.agent_dir / "data" / "sft_train.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dataset_path, "w") as f:
            for entry in synthetic_data:
                f.write(json.dumps(entry) + "\n")

        logger.info("Generated synthetic SFT dataset: %d examples at %s", len(synthetic_data), dataset_path)
        return dataset_path

    def train(
        self,
        data_path: Optional[str] = None,
        epochs: int = 3,
    ) -> Dict[str, Any]:
        """Run SFT training.

        Args:
            data_path: Path to training data (JSONL). Uses synthetic data if None.
            epochs: Number of training epochs.

        Returns:
            Training results dictionary.
        """
        manifest = self._load_manifest()
        model_id = manifest.get("model", {}).get("base", "")

        if not model_id:
            raise ValueError("No base model specified in agent manifest")

        dataset_path = self._prepare_dataset(data_path)
        logger.info("Starting SFT training: model=%s, epochs=%d", model_id, epochs)

        result = {
            "phase": "sft",
            "model": model_id,
            "dataset": str(dataset_path),
            "epochs": epochs,
            "config": self.config,
            "started_at": time.time(),
            "status": "pending",
        }

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType

            logger.info("Loading model: %s", model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Configure QLoRA
            lora_config = LoraConfig(
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=self.config["lora_dropout"],
                target_modules=self.config["lora_target_modules"],
                task_type=TaskType.CAUSAL_LM,
            )

            logger.info("QLoRA config: r=%d, alpha=%d", lora_config.r, lora_config.lora_alpha)
            result["status"] = "configured"

            # In production, this would load the model, apply LoRA, and train.
            # For the scaffold, we record the configuration.
            logger.info("SFT training configured. Full training requires GPU resources.")
            result["status"] = "configured"
            result["output_dir"] = str(self.output_dir)
            result["completed_at"] = time.time()

        except ImportError as e:
            logger.warning("Training dependencies not available: %s", e)
            result["status"] = "skipped"
            result["reason"] = f"Missing dependency: {e}"
            result["completed_at"] = time.time()

        # Record in manifest
        manifest.setdefault("training_history", []).append(result)
        manifest["state"] = "training"
        self._save_manifest(manifest)

        return result
