
from __future__ import annotations
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def load_configs(controller: str, variant: Optional[str]) -> Dict[str, Any]:
    env_path = ROOT / "config" / "env.yaml"
    ctrl_path = ROOT / "config" / "controllers" / f"{controller}.yaml"
    cfg = load_yaml(env_path)
    cfg_ctrl = load_yaml(ctrl_path)
    cfg = deep_merge(cfg, {"controller": cfg_ctrl})
    if variant:
        var_path = ROOT / "config" / "controllers" / f"{variant}.yaml"
        if var_path.exists():
            cfg_var = load_yaml(var_path)
            cfg = deep_merge(cfg, {"controller": cfg_var})
        else:
            raise FileNotFoundError(f"Variant not found: {var_path}")
    return cfg
