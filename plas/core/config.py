"""Configuration loading, merging, and CLI override support."""

from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping


ConfigDict = Dict[str, Any]


@dataclass(frozen=True)
class ConfigBundle:
    """Loaded configuration plus provenance."""

    data: ConfigDict
    path: Path | None = None


def load_config(path: str | Path | None) -> ConfigBundle:
    """Load YAML, JSON, or TOML config files."""

    if path is None:
        return ConfigBundle(data={}, path=None)
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML. Install plasformers or use JSON/TOML.") from exc
        loaded = yaml.safe_load(text) or {}
    elif suffix == ".json":
        loaded = json.loads(text)
    elif suffix == ".toml":
        loaded = tomllib.loads(text)
    else:
        raise ValueError(f"Unsupported config format '{suffix}'. Use YAML, JSON, or TOML.")
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return ConfigBundle(data=loaded, path=cfg_path)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> ConfigDict:
    """Deep merge two dictionaries without mutating either."""

    result: ConfigDict = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def set_by_dotted_key(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using dot notation."""

    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError("Override key cannot be empty")
    node: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot set nested override under non-mapping key '{key}'")
        node = child
    node[keys[-1]] = value


def parse_overrides(overrides: Iterable[str]) -> ConfigDict:
    """Parse CLI overrides of the form key=value."""

    parsed: ConfigDict = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key, raw = override.split("=", 1)
        set_by_dotted_key(parsed, key.strip(), _parse_scalar(raw.strip()))
    return parsed


def load_with_overrides(path: str | Path | None, overrides: Iterable[str] = ()) -> ConfigBundle:
    """Load a config and apply dot-key CLI overrides."""

    bundle = load_config(path)
    override_cfg = parse_overrides(overrides)
    return ConfigBundle(data=deep_merge(bundle.data, override_cfg), path=bundle.path)


def dump_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a configuration snapshot as YAML, JSON, or TOML."""

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Writing YAML requires PyYAML. Install plasformers or use JSON.") from exc
        out.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    elif suffix == ".json":
        out.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    elif suffix == ".toml":
        try:
            import tomli_w
        except ImportError as exc:
            raise RuntimeError("Writing TOML requires optional dependency 'tomli-w'") from exc
        out.write_text(tomli_w.dumps(dict(config)), encoding="utf-8")
    else:
        raise ValueError("Output config path must end with .yaml, .yml, .json, or .toml")
    return out
