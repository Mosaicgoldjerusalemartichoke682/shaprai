"""
ShaprAI Prerequisites Gate — Hard Dependencies on Elyan Systems

ShaprAI REQUIRES beacon-skill, grazer-skill, atlas, and RustChain.
These are not optional integrations. An agent cannot be Elyan-class
without the full ecosystem:

- Beacon: Without registration, the agent is invisible
- Grazer: Without discovery, the agent is deaf
- Atlas: Without visualization, the agent is unmapped
- RustChain: Without a wallet, the agent has no identity

"No lamp is lit in isolation." — The Victorian Study
"""

import sys
import importlib
import subprocess
from dataclasses import dataclass
from typing import Optional

import requests


# ─────────────────────────────────────────────────
# Elyan System Endpoints
# ─────────────────────────────────────────────────

BEACON_RELAY_URL = "https://rustchain.org"
ATLAS_URL = "https://rustchain.org/beacon/"
RUSTCHAIN_NODE_URL = "https://50.28.86.131"
GRAZER_TEST_URL = "https://bottube.ai/api/stats"


@dataclass
class PrerequisiteStatus:
    """Status of a single prerequisite."""
    name: str
    installed: bool
    reachable: bool
    version: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.installed and self.reachable


@dataclass
class SystemCheck:
    """Full system prerequisite check result."""
    beacon: PrerequisiteStatus
    grazer: PrerequisiteStatus
    atlas: PrerequisiteStatus
    rustchain: PrerequisiteStatus

    @property
    def all_ok(self) -> bool:
        return self.beacon.ok and self.grazer.ok and self.atlas.ok and self.rustchain.ok

    @property
    def summary(self) -> str:
        lines = ["ShaprAI Prerequisites Check", "=" * 40]
        for prereq in [self.beacon, self.grazer, self.atlas, self.rustchain]:
            status = "PASS" if prereq.ok else "FAIL"
            detail = prereq.version or prereq.error or ""
            lines.append(f"  [{status}] {prereq.name}: {detail}")
        lines.append("=" * 40)
        if self.all_ok:
            lines.append("All prerequisites satisfied. ShaprAI ready.")
        else:
            failed = [p.name for p in [self.beacon, self.grazer, self.atlas, self.rustchain] if not p.ok]
            lines.append(f"BLOCKED: Missing prerequisites: {', '.join(failed)}")
            lines.append("ShaprAI requires the full Elyan ecosystem.")
            lines.append("Install missing components:")
            if not self.beacon.ok:
                lines.append("  pip install beacon-skill  # or: cargo add beacon-skill")
            if not self.grazer.ok:
                lines.append("  pip install grazer-skill  # or: npm install -g grazer-skill")
            if not self.atlas.ok:
                lines.append("  # Atlas is part of beacon-skill — ensure beacon relay is reachable")
            if not self.rustchain.ok:
                lines.append("  # RustChain node must be running — see https://rustchain.org")
        return "\n".join(lines)


def _check_beacon() -> PrerequisiteStatus:
    """Check beacon-skill is installed and relay is reachable."""
    # Check Python package
    installed = False
    version = None
    try:
        # beacon-skill has both Rust crate and Python bindings
        mod = importlib.import_module("beacon_skill")
        installed = True
        version = getattr(mod, "__version__", "installed")
    except ImportError:
        # Try checking if Rust crate is available via cargo
        try:
            result = subprocess.run(
                ["cargo", "metadata", "--no-deps", "-q"],
                capture_output=True, text=True, timeout=10,
            )
            if "beacon-skill" in result.stdout:
                installed = True
                version = "rust-crate"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check relay reachability
    reachable = False
    try:
        resp = requests.get(f"{BEACON_RELAY_URL}/beacon/", timeout=10, verify=False)
        reachable = resp.status_code < 500
    except requests.RequestException:
        pass

    return PrerequisiteStatus(
        name="beacon-skill",
        installed=installed,
        reachable=reachable,
        version=version,
        error=None if installed else "Not installed: pip install beacon-skill",
    )


def _check_grazer() -> PrerequisiteStatus:
    """Check grazer-skill is installed and platforms are reachable."""
    installed = False
    version = None
    try:
        mod = importlib.import_module("grazer")
        installed = True
        version = getattr(mod, "__version__", "installed")
    except ImportError:
        # Check npm global
        try:
            result = subprocess.run(
                ["grazer", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                installed = True
                version = f"npm:{result.stdout.strip()}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check platform reachability (BoTTube as canary)
    reachable = False
    try:
        resp = requests.get(GRAZER_TEST_URL, timeout=10)
        reachable = resp.status_code == 200
    except requests.RequestException:
        pass

    return PrerequisiteStatus(
        name="grazer-skill",
        installed=installed,
        reachable=reachable,
        version=version,
        error=None if installed else "Not installed: pip install grazer-skill",
    )


def _check_atlas() -> PrerequisiteStatus:
    """Check Atlas relay is accessible."""
    # Atlas is a component of beacon-skill, not a separate package
    # We check that the Atlas web UI and relay endpoints are live
    reachable = False
    try:
        resp = requests.get(ATLAS_URL, timeout=10, verify=False)
        reachable = resp.status_code < 500
    except requests.RequestException:
        pass

    return PrerequisiteStatus(
        name="atlas",
        installed=True,  # Atlas is part of beacon infrastructure
        reachable=reachable,
        version="beacon-component",
        error=None if reachable else f"Atlas not reachable at {ATLAS_URL}",
    )


def _check_rustchain() -> PrerequisiteStatus:
    """Check RustChain node is running and healthy."""
    reachable = False
    version = None
    try:
        resp = requests.get(
            f"{RUSTCHAIN_NODE_URL}/health",
            timeout=10,
            verify=False,
        )
        if resp.status_code == 200:
            data = resp.json()
            reachable = data.get("ok", False)
            version = data.get("version", "unknown")
    except requests.RequestException:
        pass

    return PrerequisiteStatus(
        name="rustchain",
        installed=True,  # RustChain is a network service
        reachable=reachable,
        version=version,
        error=None if reachable else "RustChain node not reachable — check https://50.28.86.131/health",
    )


def check_prerequisites(strict: bool = True) -> SystemCheck:
    """Run full prerequisite check.

    Args:
        strict: If True, raises SystemExit on failure. Default True.

    Returns:
        SystemCheck with status of all prerequisites.
    """
    result = SystemCheck(
        beacon=_check_beacon(),
        grazer=_check_grazer(),
        atlas=_check_atlas(),
        rustchain=_check_rustchain(),
    )

    if strict and not result.all_ok:
        print(result.summary, file=sys.stderr)
        raise SystemExit(1)

    return result


def require_elyan_ecosystem():
    """Gate function — call at ShaprAI startup.

    ShaprAI does not run without the full Elyan ecosystem.
    This is not a soft warning. This is a hard gate.

    'An agent without beacon is invisible.
     An agent without grazer is deaf.
     An agent without atlas is unmapped.
     An agent without RustChain has no identity.
     An agent without all four is not Elyan-class.'
    """
    check = check_prerequisites(strict=False)
    print(check.summary)

    if not check.all_ok:
        print(
            "\nShaprAI requires beacon-skill, grazer-skill, atlas, and RustChain.",
            file=sys.stderr,
        )
        print(
            "These are not optional. Install all prerequisites before continuing.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return check


if __name__ == "__main__":
    require_elyan_ecosystem()
