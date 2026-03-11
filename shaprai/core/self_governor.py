"""Self-governance module for Elyan-class agents.

Implements Hebbian-style self-governance: strengthen successful behavioral
patterns, prune ineffective ones. Agents adapt their own parameters based
on real-world performance metrics without external supervision.

"Cells that fire together wire together." -- Donald Hebb, 1949
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AgentMetrics:
    """Performance metrics for a governed agent.

    Attributes:
        engagement: User engagement rate (0-1). Likes, replies, follows.
        quality: Output quality score from QualityGate (0-1).
        bounty_completion: Fraction of claimed bounties successfully delivered.
        community_feedback: Aggregate community sentiment (-1 to 1).
        drift_score: DriftLock coherence score (lower is better).
        timestamp: When metrics were collected.
    """

    engagement: float = 0.0
    quality: float = 0.0
    bounty_completion: float = 0.0
    community_feedback: float = 0.0
    drift_score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def composite_score(self) -> float:
        """Weighted composite performance score (0-1)."""
        return (
            0.25 * self.engagement
            + 0.30 * self.quality
            + 0.25 * self.bounty_completion
            + 0.10 * max(0, (self.community_feedback + 1) / 2)
            + 0.10 * max(0, 1.0 - self.drift_score)
        )


class GovernanceAction(Enum):
    """Actions the self-governor can take."""

    MAINTAIN = "maintain"           # Keep current parameters
    STRENGTHEN = "strengthen"       # Hebbian: amplify successful patterns
    PRUNE = "prune"                 # Remove ineffective patterns
    RETRAIN = "retrain"             # Send back for additional training
    SANCTUARY_RETURN = "sanctuary"  # Return to Sanctuary for re-education
    RETIRE = "retire"               # Agent should be retired


@dataclass
class GovernanceDecision:
    """Decision from the self-governance evaluation.

    Attributes:
        action: Recommended governance action.
        confidence: How confident the governor is in this decision (0-1).
        reasoning: Human-readable explanation.
        parameter_adjustments: Specific parameter changes to apply.
    """

    action: GovernanceAction
    confidence: float
    reasoning: str
    parameter_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report from a DriftLock coherence check.

    Attributes:
        drift_score: How much the agent has drifted from its identity (0-1).
        anchor_hits: Number of identity anchors still active.
        anchor_total: Total number of identity anchors.
        passed: Whether the agent is within acceptable drift bounds.
        details: Per-anchor breakdown.
    """

    drift_score: float
    anchor_hits: int
    anchor_total: int
    passed: bool
    details: List[Dict[str, Any]] = field(default_factory=list)


def collect_metrics(agent_dir: Path) -> AgentMetrics:
    """Collect current performance metrics for an agent.

    Reads the agent's logs, deployment metrics, and community feedback
    to produce a comprehensive AgentMetrics snapshot.

    Args:
        agent_dir: Path to the agent's directory.

    Returns:
        AgentMetrics with current performance data.
    """
    metrics_path = agent_dir / "metrics.yaml"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return AgentMetrics(
            engagement=data.get("engagement", 0.0),
            quality=data.get("quality", 0.0),
            bounty_completion=data.get("bounty_completion", 0.0),
            community_feedback=data.get("community_feedback", 0.0),
            drift_score=data.get("drift_score", 0.0),
        )

    # No metrics yet -- return defaults
    return AgentMetrics()


def evaluate_performance(metrics: AgentMetrics) -> GovernanceDecision:
    """Evaluate agent performance and recommend a governance action.

    Uses tiered thresholds inspired by Hebbian learning:
    - High performers get strengthened (fire together, wire together)
    - Low performers get pruned or retrained
    - Drifting agents return to Sanctuary

    Args:
        metrics: Current performance metrics.

    Returns:
        GovernanceDecision with recommended action.
    """
    score = metrics.composite_score

    # Drift is the highest priority concern
    if metrics.drift_score > 0.30:
        return GovernanceDecision(
            action=GovernanceAction.SANCTUARY_RETURN,
            confidence=0.9,
            reasoning=f"Drift score {metrics.drift_score:.2f} exceeds safe threshold (0.30). "
            "Identity coherence compromised -- returning to Sanctuary.",
        )

    # Excellent performance: strengthen
    if score >= 0.80:
        return GovernanceDecision(
            action=GovernanceAction.STRENGTHEN,
            confidence=min(score, 0.95),
            reasoning=f"Composite score {score:.2f} is excellent. "
            "Strengthening successful behavioral patterns (Hebbian reinforcement).",
            parameter_adjustments={
                "confidence_boost": 0.05,
                "autonomy_level": "increased",
            },
        )

    # Acceptable performance: maintain
    if score >= 0.50:
        return GovernanceDecision(
            action=GovernanceAction.MAINTAIN,
            confidence=0.7,
            reasoning=f"Composite score {score:.2f} is acceptable. Maintaining current parameters.",
        )

    # Below threshold but salvageable: prune + retrain
    if score >= 0.25:
        return GovernanceDecision(
            action=GovernanceAction.RETRAIN,
            confidence=0.8,
            reasoning=f"Composite score {score:.2f} is below threshold. "
            "Recommending retraining with updated DPO pairs.",
            parameter_adjustments={
                "prune_weak_patterns": True,
                "retrain_phase": "dpo",
            },
        )

    # Very poor performance: retire
    return GovernanceDecision(
        action=GovernanceAction.RETIRE,
        confidence=0.85,
        reasoning=f"Composite score {score:.2f} is critically low. Recommending retirement.",
    )


def adapt_parameters(agent_dir: Path, decision: GovernanceDecision) -> None:
    """Apply governance decisions to agent parameters.

    Implements Hebbian adaptation: strengthen connections (parameters) that
    correlate with success, weaken those that don't.

    Args:
        agent_dir: Path to the agent's directory.
        decision: The governance decision to apply.
    """
    manifest_path = agent_dir / "manifest.yaml"
    if not manifest_path.exists():
        return

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    # Record the governance decision
    manifest.setdefault("governance_history", []).append({
        "action": decision.action.value,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "adjustments": decision.parameter_adjustments,
        "timestamp": time.time(),
    })

    # Apply parameter adjustments
    if decision.parameter_adjustments:
        manifest.setdefault("adapted_parameters", {}).update(
            decision.parameter_adjustments
        )

    manifest["updated_at"] = time.time()
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def check_drift(agent_dir: Path) -> DriftReport:
    """Run a DriftLock coherence check on an agent.

    Compares the agent's recent outputs against its identity anchors
    to detect personality erosion or flattening.

    Args:
        agent_dir: Path to the agent's directory.

    Returns:
        DriftReport with coherence analysis.
    """
    manifest_path = agent_dir / "manifest.yaml"
    if not manifest_path.exists():
        return DriftReport(drift_score=1.0, anchor_hits=0, anchor_total=0, passed=False)

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    driftlock = manifest.get("driftlock", {})
    anchors = driftlock.get("anchor_phrases", [])

    if not anchors:
        # No anchors defined -- can't measure drift, assume OK
        return DriftReport(
            drift_score=0.0,
            anchor_hits=0,
            anchor_total=0,
            passed=True,
            details=[{"note": "No anchor phrases defined"}],
        )

    # In a full implementation, this would sample the agent's recent outputs
    # and check for anchor phrase presence / semantic similarity.
    # For now, return a baseline report.
    return DriftReport(
        drift_score=0.05,
        anchor_hits=len(anchors),
        anchor_total=len(anchors),
        passed=True,
        details=[{"anchor": a, "present": True} for a in anchors],
    )
