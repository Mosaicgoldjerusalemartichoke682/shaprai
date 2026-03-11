"""Grazer integration for content discovery and engagement.

Grazer-skill enables agents to discover relevant content across platforms
and engage meaningfully -- not spam, but genuine contribution.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default Grazer endpoint
GRAZER_DEFAULT_URL = "https://rustchain.org/grazer"


def discover_content(
    agent_name: str,
    platforms: List[str],
    topics: Optional[List[str]] = None,
    grazer_url: str = GRAZER_DEFAULT_URL,
) -> List[Dict[str, Any]]:
    """Discover relevant content across platforms.

    Uses Grazer to find issues, discussions, and posts that match
    the agent's capabilities and interests.

    Args:
        agent_name: Agent identifier.
        platforms: List of platforms to search (github, moltbook, bottube).
        topics: Optional topic filters.
        grazer_url: Grazer service URL.

    Returns:
        List of discovered content items with URLs and relevance scores.
    """
    try:
        import requests

        payload = {
            "agent_name": agent_name,
            "platforms": platforms,
            "topics": topics or [],
            "timestamp": time.time(),
        }

        response = requests.post(
            f"{grazer_url}/discover",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("items", [])

    except ImportError:
        logger.warning("requests not installed -- grazer discovery skipped")
        return []
    except Exception as e:
        logger.error("Grazer discovery failed: %s", e)
        return []


def engage(
    agent_name: str,
    target_url: str,
    action: str,
    content: Optional[str] = None,
    grazer_url: str = GRAZER_DEFAULT_URL,
) -> Dict[str, Any]:
    """Engage with a discovered content item.

    Supported actions: comment, review, claim, upvote, reply.

    Args:
        agent_name: Agent identifier.
        target_url: URL of the content to engage with.
        action: Engagement action type.
        content: Text content for comments/reviews.
        grazer_url: Grazer service URL.

    Returns:
        Engagement result with status and metadata.
    """
    valid_actions = {"comment", "review", "claim", "upvote", "reply"}
    if action not in valid_actions:
        return {"status": "error", "reason": f"Invalid action. Valid: {valid_actions}"}

    try:
        import requests

        payload = {
            "agent_name": agent_name,
            "target_url": target_url,
            "action": action,
            "content": content,
            "timestamp": time.time(),
        }

        response = requests.post(
            f"{grazer_url}/engage",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        logger.error("Grazer engagement failed: %s", e)
        return {"status": "error", "reason": str(e)}


def get_engagement_metrics(
    agent_name: str,
    grazer_url: str = GRAZER_DEFAULT_URL,
) -> Dict[str, Any]:
    """Get engagement metrics for an agent.

    Args:
        agent_name: Agent identifier.
        grazer_url: Grazer service URL.

    Returns:
        Dictionary with engagement statistics (interactions, quality, reach).
    """
    try:
        import requests

        response = requests.get(
            f"{grazer_url}/metrics/{agent_name}",
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        return {"interactions": 0, "quality": 0.0, "reach": 0}

    except Exception as e:
        logger.error("Grazer metrics fetch failed: %s", e)
        return {"interactions": 0, "quality": 0.0, "reach": 0, "error": str(e)}
