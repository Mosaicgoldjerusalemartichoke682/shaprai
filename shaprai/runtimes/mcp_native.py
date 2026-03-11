"""Raw MCP (Model Context Protocol) runtime for Elyan-class agents.

Provides a native MCP agent implementation with Beacon and Grazer
registered as default tools. This is the most lightweight runtime
option -- no framework overhead, just direct tool registration and
message handling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from shaprai.sanctuary.principles import get_ethics_prompt

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """A tool registered with an MCP agent.

    Attributes:
        name: Tool identifier.
        description: Human-readable description.
        parameters: JSON Schema for tool parameters.
        handler: Callable that executes the tool.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class MCPMessage:
    """A message in the MCP conversation.

    Attributes:
        role: Message role (system, user, assistant, tool).
        content: Message text content.
        tool_calls: Optional list of tool call requests.
        tool_results: Optional list of tool results.
        timestamp: Message creation time.
    """

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    timestamp: float = field(default_factory=time.time)


class MCPAgent:
    """Native MCP agent with tool registration and SophiaCore principles.

    This is the lightest-weight runtime option. It manages the conversation
    context, tool registry, and system prompt injection without any
    framework dependencies.

    Attributes:
        name: Agent identifier.
        system_prompt: System prompt with SophiaCore principles.
        tools: Dictionary of registered tools.
        history: Conversation history.
    """

    def __init__(
        self,
        name: str,
        additional_prompt: str = "",
        max_history: int = 100,
    ) -> None:
        """Initialize a native MCP agent.

        Beacon and Grazer are registered as default tools.

        Args:
            name: Unique agent identifier.
            additional_prompt: Extra system prompt content.
            max_history: Maximum conversation history length.
        """
        self.name = name
        self.max_history = max_history
        self.tools: Dict[str, MCPTool] = {}
        self.history: List[MCPMessage] = []

        # Build system prompt with SophiaCore principles
        ethics = get_ethics_prompt()
        self.system_prompt = ethics
        if additional_prompt:
            self.system_prompt += f"\n\n---\n\n{additional_prompt}"

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register Beacon and Grazer as default tools."""
        self.register_tool(MCPTool(
            name="beacon_heartbeat",
            description="Send a heartbeat to the Beacon discovery service to confirm agent is alive.",
            parameters={
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "object",
                        "description": "Optional metrics to include in heartbeat",
                    },
                },
            },
            handler=self._beacon_heartbeat,
        ))

        self.register_tool(MCPTool(
            name="grazer_discover",
            description="Discover relevant content across platforms using Grazer.",
            parameters={
                "type": "object",
                "properties": {
                    "platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Platforms to search (github, moltbook, bottube)",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topic filters",
                    },
                },
                "required": ["platforms"],
            },
            handler=self._grazer_discover,
        ))

        self.register_tool(MCPTool(
            name="grazer_engage",
            description="Engage with discovered content (comment, review, claim).",
            parameters={
                "type": "object",
                "properties": {
                    "target_url": {"type": "string", "description": "URL to engage with"},
                    "action": {
                        "type": "string",
                        "enum": ["comment", "review", "claim", "upvote", "reply"],
                    },
                    "content": {"type": "string", "description": "Text content for the engagement"},
                },
                "required": ["target_url", "action"],
            },
            handler=self._grazer_engage,
        ))

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the agent.

        Args:
            tool: MCPTool instance to register.
        """
        self.tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get JSON Schema descriptions of all registered tools.

        Returns:
            List of tool schema dictionaries.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            KeyError: If the tool is not registered.
        """
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not registered. Available: {list(self.tools.keys())}")

        tool = self.tools[tool_name]
        logger.info("Executing tool: %s", tool_name)
        result = tool.handler(**arguments)
        return result

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role (system, user, assistant, tool).
            content: Message content.
            **kwargs: Additional message fields.
        """
        msg = MCPMessage(role=role, content=content, **kwargs)
        self.history.append(msg)

        # Trim history if needed (keep system prompt)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self) -> List[Dict[str, str]]:
        """Get the full conversation context for LLM input.

        Returns:
            List of message dictionaries with role and content.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    # ------------------------------------------------------------------- #
    #  Default tool handlers
    # ------------------------------------------------------------------- #

    def _beacon_heartbeat(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Beacon heartbeat tool handler."""
        try:
            from shaprai.integrations.beacon import update_heartbeat

            success = update_heartbeat(self.name, metrics)
            return {"status": "ok" if success else "failed"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def _grazer_discover(
        self,
        platforms: List[str],
        topics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Grazer discovery tool handler."""
        try:
            from shaprai.integrations.grazer import discover_content

            return discover_content(self.name, platforms, topics)
        except Exception as e:
            return [{"status": "error", "reason": str(e)}]

    def _grazer_engage(
        self,
        target_url: str,
        action: str,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Grazer engagement tool handler."""
        try:
            from shaprai.integrations.grazer import engage

            return engage(self.name, target_url, action, content)
        except Exception as e:
            return {"status": "error", "reason": str(e)}
