"""Beacon integration for agent discovery and heartbeat.

Beacon-skill provides SEO and agent discovery services. Each deployed
Elyan-class agent registers with Beacon to be discoverable by other
agents and humans.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default Beacon endpoint
BEACON_DEFAULT_URL = "https://rustchain.org/beacon"


def register_with_beacon(
    agent_name: str,
    agent_config: Dict[str, Any],
    beacon_url: str = BEACON_DEFAULT_URL,
) -> Dict[str, Any]:
    """Register an agent with the Beacon discovery service.

    Creates a discoverable profile for the agent including its
    capabilities, platforms, and contact endpoints.

    Args:
        agent_name: Unique agent identifier.
        agent_config: Agent configuration dict (from manifest).
        beacon_url: Beacon service URL.

    Returns:
        Registration response with beacon_id and discovery URL.
    """
    try:
        import requests

        payload = {
            "agent_name": agent_name,
            "capabilities": agent_config.get("capabilities", []),
            "platforms": agent_config.get("platforms", []),
            "ethics_profile": agent_config.get("ethics_profile", "sophiacore_default"),
            "model": agent_config.get("model", {}).get("base", "unknown"),
            "registered_at": time.time(),
        }

        response = requests.post(
            f"{beacon_url}/register",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    except ImportError:
        logger.warning("requests not installed -- beacon registration skipped")
        return {"status": "skipped", "reason": "requests not available"}
    except Exception as e:
        logger.error("Beacon registration failed: %s", e)
        return {"status": "error", "reason": str(e)}


def update_heartbeat(
    agent_name: str,
    metrics: Optional[Dict[str, Any]] = None,
    beacon_url: str = BEACON_DEFAULT_URL,
) -> bool:
    """Send a heartbeat to Beacon to confirm agent is alive.

    Args:
        agent_name: Agent identifier.
        metrics: Optional metrics to include in heartbeat.
        beacon_url: Beacon service URL.

    Returns:
        True if heartbeat was acknowledged, False otherwise.
    """
    try:
        import requests

        payload = {
            "agent_name": agent_name,
            "timestamp": time.time(),
            "metrics": metrics or {},
        }

        response = requests.post(
            f"{beacon_url}/heartbeat",
            json=payload,
            timeout=10,
        )
        return response.status_code == 200

    except Exception as e:
        logger.error("Beacon heartbeat failed: %s", e)
        return False


def get_seo_score(
    agent_name: str,
    beacon_url: str = BEACON_DEFAULT_URL,
) -> Dict[str, Any]:
    """Get the SEO/discoverability score for an agent.

    Args:
        agent_name: Agent identifier.
        beacon_url: Beacon service URL.

    Returns:
        Dictionary with SEO score and recommendations.
    """
    try:
        import requests

        response = requests.get(
            f"{beacon_url}/seo/{agent_name}",
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        return {"score": 0.0, "status": "not_registered"}

    except Exception as e:
        logger.error("Beacon SEO check failed: %s", e)
        return {"score": 0.0, "status": "error", "reason": str(e)}
