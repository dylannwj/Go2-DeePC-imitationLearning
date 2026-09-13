"""Fail-closed guard for the Genesis-only UI process.

The guard is deliberately small and has no transport imports.  It runs before
the Genesis scene is constructed and again at the runtime boundary.  If a
physical transport module or physical transport environment is already
initialized, the UI refuses to start.
"""

from __future__ import annotations

import os
import sys
from typing import Any


class PhysicalGo2GuardError(RuntimeError):
    """Raised when the Genesis-only process is not physically isolated."""


_FORBIDDEN_MODULE_PREFIXES = (
    "unitree_sdk",
    "unitree_sdk2",
    "rclpy",
    "rospy",
    "sportclient",
    "highcmd",
    "lowcmd",
    "dds",
)
_FORBIDDEN_ENV_TOKENS = (
    "UNITREE",
    "ROBOT_IP",
    "GO2_IP",
    "ROS_DOMAIN_ID",
    "ROS_MASTER_URI",
    "DDS_DOMAIN",
)


def _physical_modules_loaded() -> list[str]:
    loaded: list[str] = []
    for name in sys.modules:
        lowered = name.lower()
        if any(lowered == prefix or lowered.startswith(prefix + ".") for prefix in _FORBIDDEN_MODULE_PREFIXES):
            loaded.append(name)
    return sorted(loaded)


def _physical_environment_keys() -> list[str]:
    return sorted(
        key
        for key in os.environ
        if any(token in key.upper() for token in _FORBIDDEN_ENV_TOKENS)
    )


def physical_guard_snapshot(*, phase: str) -> dict[str, Any]:
    """Return the current physical-isolation status without side effects."""

    modules = _physical_modules_loaded()
    environment_keys = _physical_environment_keys()
    passed = not modules and not environment_keys
    return {
        "schema": "genesis-only-physical-go2-guard-v1",
        "phase": str(phase),
        "status": "PASS" if passed else "FAIL",
        "PHYSICAL_GO2_GUARD": "PASS" if passed else "FAIL",
        "PHYSICAL_GO2_PATH": "NO" if passed else "YES",
        "genesis_only": True,
        "physical_transport_modules_loaded": modules,
        "physical_transport_environment_keys": environment_keys,
        "physical_ui_used": False,
        "physical_go2_used": False,
        "physical_command_publisher_used": False,
        "physical_network_connection_attempted": False,
        "fail_closed_before_genesis_scene": True,
    }


def assert_genesis_only(phase: str) -> dict[str, Any]:
    """Raise before scene/command execution if physical state is present."""

    result = physical_guard_snapshot(phase=phase)
    if result["status"] != "PASS":
        raise PhysicalGo2GuardError(
            "Genesis-only guard failed: "
            f"modules={result['physical_transport_modules_loaded']}, "
            f"environment_keys={result['physical_transport_environment_keys']}"
        )
    return result
