from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "config/default.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_artifacts_dir(cfg: dict[str, Any], smoke: bool = False) -> Path:
    key = "smoke_artifacts_dir" if smoke else "artifacts_dir"
    out = Path(cfg["paths"][key])
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    return out


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_manifest(artifacts_dir: Path) -> dict[str, Any]:
    mpath = artifacts_dir / "run_manifest.json"
    if not mpath.exists():
        return {}
    return json.loads(mpath.read_text(encoding="utf-8"))


def save_manifest(artifacts_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest(artifacts_dir)
    manifest.update(updates)
    mpath = artifacts_dir / "run_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest