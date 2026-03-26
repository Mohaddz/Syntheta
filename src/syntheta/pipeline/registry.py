"""Component registry for generators, transformers, and filters."""

from __future__ import annotations

_generators: dict[str, type] = {}
_transformers: dict[str, type] = {}
_filters: dict[str, type] = {}


def register_generator(name: str):
    """Decorator to register a generator class."""

    def decorator(cls: type) -> type:
        _generators[name] = cls
        return cls

    return decorator


def register_transformer(name: str):
    """Decorator to register a transformer class."""

    def decorator(cls: type) -> type:
        _transformers[name] = cls
        return cls

    return decorator


def register_filter(name: str):
    """Decorator to register a filter class."""

    def decorator(cls: type) -> type:
        _filters[name] = cls
        return cls

    return decorator


def get_generator(name: str) -> type:
    """Look up a registered generator by name."""
    if name not in _generators:
        raise KeyError(f"Generator '{name}' not registered. Available: {list(_generators)}")
    return _generators[name]


def get_transformer(name: str) -> type:
    """Look up a registered transformer by name."""
    if name not in _transformers:
        raise KeyError(f"Transformer '{name}' not registered. Available: {list(_transformers)}")
    return _transformers[name]


def get_filter(name: str) -> type:
    """Look up a registered filter by name."""
    if name not in _filters:
        raise KeyError(f"Filter '{name}' not registered. Available: {list(_filters)}")
    return _filters[name]
