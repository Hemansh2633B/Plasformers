"""Simple plugin registry for third-party extensions."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable


Factory = Callable[..., Any]


@dataclass(frozen=True)
class PluginRecord:
    """A registered extension factory."""

    kind: str
    name: str
    factory: Factory
    source: str = "runtime"


class PluginRegistry:
    """Registry for custom backbones, losses, augmentations, heads, and exporters."""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, PluginRecord]] = {}

    def register(self, kind: str, name: str, factory: Factory, source: str = "runtime") -> None:
        self._items.setdefault(kind, {})[name] = PluginRecord(kind, name, factory, source)

    def get(self, kind: str, name: str) -> Factory:
        try:
            return self._items[kind][name].factory
        except KeyError as exc:
            raise KeyError(f"No plugin registered for kind='{kind}', name='{name}'") from exc

    def list(self, kind: str | None = None) -> Iterable[PluginRecord]:
        if kind is not None:
            return tuple(self._items.get(kind, {}).values())
        records = []
        for values in self._items.values():
            records.extend(values.values())
        return tuple(records)

    def load_module(self, dotted_path: str) -> None:
        """Import a module that registers plugins as a side effect."""

        importlib.import_module(dotted_path)


GLOBAL_PLUGINS = PluginRegistry()


def register_plugin(kind: str, name: str) -> Callable[[Factory], Factory]:
    """Decorator for registering plugin factories."""

    def decorator(factory: Factory) -> Factory:
        GLOBAL_PLUGINS.register(kind, name, factory, source=f"{factory.__module__}.{factory.__name__}")
        return factory

    return decorator
