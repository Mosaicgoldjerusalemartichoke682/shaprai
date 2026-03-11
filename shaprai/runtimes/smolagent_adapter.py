"""smolagents adapter for Elyan-class agents.

Wraps HuggingFace's smolagents with SophiaCore principle injection,
ensuring lightweight tool-using agents maintain Elyan-class identity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from shaprai.sanctuary.principles import get_ethics_prompt

logger = logging.getLogger(__name__)


class ShaprSmolagent:
    """smolagents wrapper with SophiaCore principles injected.

    Ensures that smolagents-based agents maintain their Elyan-class
    identity and ethical framework during tool-use interactions.

    Attributes:
        name: Agent identifier.
        model_id: HuggingFace model identifier.
        tools: List of tool objects.
        system_prompt: System prompt with SophiaCore principles.
    """

    def __init__(
        self,
        name: str,
        model_id: str = "Qwen/Qwen3-7B-Instruct",
        tools: Optional[List[Any]] = None,
        additional_prompt: str = "",
    ) -> None:
        """Initialize a ShaprAI-wrapped smolagent.

        Args:
            name: Unique agent identifier.
            model_id: HuggingFace model to use.
            tools: List of tool objects for the agent.
            additional_prompt: Extra system prompt content appended after ethics.
        """
        self.name = name
        self.model_id = model_id
        self.tools = tools or []

        # Build system prompt with SophiaCore principles
        ethics = get_ethics_prompt()
        self.system_prompt = ethics
        if additional_prompt:
            self.system_prompt += f"\n\n---\n\n{additional_prompt}"

        self._agent = None

    def build(self) -> Any:
        """Build the smolagents agent instance.

        Returns:
            smolagents agent object.

        Raises:
            ImportError: If smolagents is not installed.
        """
        try:
            from smolagents import CodeAgent, HfApiModel

            model = HfApiModel(model_id=self.model_id)

            self._agent = CodeAgent(
                tools=self.tools,
                model=model,
                system_prompt=self.system_prompt,
            )

            logger.info("Built smolagent '%s' with model %s and %d tools",
                         self.name, self.model_id, len(self.tools))
            return self._agent

        except ImportError:
            raise ImportError(
                "smolagents not installed. Install with: pip install smolagents"
            )

    def run(self, task: str) -> str:
        """Run a task through the smolagent.

        Args:
            task: Natural language task description.

        Returns:
            Agent's response string.
        """
        if self._agent is None:
            self.build()

        logger.info("Running task on '%s': %s", self.name, task[:100])
        return self._agent.run(task)

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any]) -> "ShaprSmolagent":
        """Create a ShaprSmolagent from an agent manifest.

        Args:
            manifest: Agent manifest dictionary.

        Returns:
            Configured ShaprSmolagent instance.
        """
        return cls(
            name=manifest.get("name", "unnamed"),
            model_id=manifest.get("model", {}).get("base", "Qwen/Qwen3-7B-Instruct"),
            additional_prompt=manifest.get("personality", {}).get("backstory", ""),
        )
