#!/usr/bin/env python3
"""Check whether the Stage-CAP handoff environment is ready to run."""

from __future__ import annotations

import argparse
import importlib
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Check:
    name: str
    status: str
    detail: str
    required: bool = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_kite_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    kite_root = root / "third_party" / "kite"
    if kite_root.exists() and str(kite_root) not in sys.path:
        sys.path.insert(0, str(kite_root))


def _check_import(module: str, *, required: bool = True, package: str | None = None) -> Check:
    label = package or module
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", None)
        suffix = f" {version}" if version else ""
        return Check(label, "OK", f"imported{suffix}", required)
    except Exception as exc:
        return Check(label, "FAIL" if required else "WARN", f"{type(exc).__name__}: {exc}", required)


def _check_env(name: str, *, required: bool = False) -> Check:
    value = os.getenv(name)
    placeholders = {"replace_with_your_key", "<your_key>", "your_key"}
    if value and value.strip() not in placeholders:
        shown = "<set>" if "KEY" in name or "TOKEN" in name else value
        return Check(name, "OK", shown, required)
    detail = "placeholder value" if value else "not set"
    return Check(name, "FAIL" if required else "WARN", detail, required)


def _check_path(path: Path, name: str, *, required: bool = True) -> Check:
    if path.exists():
        return Check(name, "OK", str(path), required)
    return Check(name, "FAIL" if required else "WARN", f"missing: {path}", required)


def _check_url(url: str, name: str, *, required: bool = False, timeout: float = 1.0) -> Check:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return Check(name, "WARN", f"invalid or no port in URL: {url}", required)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return Check(name, "OK", f"reachable at {host}:{port}", required)
    except Exception as exc:
        return Check(name, "FAIL" if required else "WARN", f"not reachable at {host}:{port}: {exc}", required)


def _print(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks) if checks else 16
    for c in checks:
        print(f"[{c.status:<4}] {c.name:<{width}}  {c.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--strict-services", action="store_true", help="require local service ports to be reachable")
    parser.add_argument("--require-mimo-key", action="store_true", help="fail if MIMO_API_KEY is not set")
    args = parser.parse_args()

    root = _repo_root()
    _add_kite_path(root)

    checks: list[Check] = []
    checks.append(Check("python", "OK", sys.version.replace("\n", " ")))
    checks.append(_check_path(root / "capx", "capx source"))
    checks.append(_check_path(root / "vlm_planner_cap", "stage planner source"))
    checks.append(_check_path(root / "third_party" / "kite" / "kite", "vendored kite"))

    for module in [
        "numpy",
        "PIL",
        "cv2",
        "torch",
        "tyro",
        "yaml",
        "scipy",
        "matplotlib",
        "requests",
        "open3d",
        "viser",
        "capx",
        "vlm_planner_cap",
        "kite.video.keyframes",
    ]:
        checks.append(_check_import(module))

    for module in ["libero", "robosuite", "bddl", "robomimic"]:
        checks.append(_check_import(module, required=False))

    checks.extend(
        [
            _check_env("PYTHONPATH", required=False),
            _check_env("MUJOCO_GL", required=False),
            _check_env("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", required=False),
            _check_env("MIMO_BASE_URL", required=False),
            _check_env("MIMO_API_KEY", required=args.require_mimo_key),
            _check_env("SAM3_SERVICE_URL", required=False),
            _check_env("MOLMO_BASE_URL", required=False),
        ]
    )

    checks.extend(
        [
            _check_url(os.getenv("PYROKI_SERVICE_URL", "http://127.0.0.1:8116"), "PyRoKi service", required=args.strict_services),
            _check_url(os.getenv("SAM3_SERVICE_URL", "http://127.0.0.1:8114"), "SAM3 service", required=args.strict_services),
            _check_url(os.getenv("GRASPNET_SERVICE_URL", "http://127.0.0.1:8115"), "GraspNet service", required=args.strict_services),
        ]
    )

    _print(checks)

    has_fail = any(c.status == "FAIL" and c.required for c in checks)
    has_warn = any(c.status == "WARN" for c in checks)
    if has_fail or (args.strict and has_warn):
        print("\nEnvironment check did not pass. See ENVIRONMENT.md for setup details.")
        return 1
    print("\nEnvironment check passed for required local imports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
