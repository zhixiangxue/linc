"""Adapter registry — single source of truth for which IM platforms are loaded.

Both the linc-gateway (which instantiates adapters from `linc.yaml`) and the
client-side `linc.<platform>()` accessor consult this registry. Adapters
register themselves on import:

    from linc.adapters import register
    @register
    class SlackAdapter(Adapter):
        name = "slack"
        ...

Or imperatively:

    register(SlackAdapter)

For tests, `unregister(name)` and `clear()` allow temporary registrations
without polluting the global table across test cases (use a fixture).
"""

from __future__ import annotations

from typing import TypeVar

from ..core.adapter import Adapter
from ..core.errors import UnknownPlatform

_REGISTRY: dict[str, type[Adapter]] = {}

A = TypeVar("A", bound=type[Adapter])


def register(adapter_cls: A) -> A:
    """Register an Adapter subclass. Usable as a decorator.

    Raises ValueError if `adapter_cls.name` is unset or empty, or if a different
    class has already claimed that name. Re-registering the *same* class is a
    no-op (idempotent under module reload).
    """
    name = getattr(adapter_cls, "name", None)
    if not name:
        raise ValueError(
            f"{adapter_cls.__qualname__} must define a non-empty class attribute `name`"
        )
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not adapter_cls:
        raise ValueError(
            f"platform {name!r} already registered by {existing.__qualname__}"
        )
    _REGISTRY[name] = adapter_cls
    return adapter_cls


def unregister(name: str) -> None:
    """Remove a platform from the registry. No-op if absent."""
    _REGISTRY.pop(name, None)


def clear() -> None:
    """Wipe the registry. Test-only."""
    _REGISTRY.clear()


def get(name: str) -> type[Adapter]:
    """Look up an adapter class by platform name.

    Raises UnknownPlatform if `name` is not registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise UnknownPlatform(
            f"platform {name!r} is not registered; known: {sorted(_REGISTRY)}"
        ) from e


def supported() -> frozenset[str]:
    """Snapshot the currently registered platform names."""
    return frozenset(_REGISTRY)


def is_supported(name: str) -> bool:
    return name in _REGISTRY


# Eager-load built-in adapters. Each import is wrapped so that partial
# installs (e.g. running core unit tests without a particular SDK) do not
# break the registry for other adapters.
def _load_builtins() -> None:
    try:
        from . import slack  # noqa: F401
    except ImportError:
        pass
    try:
        from . import feishu  # noqa: F401
    except ImportError:
        pass
    try:
        from . import dingtalk  # noqa: F401
    except ImportError:
        pass
    try:
        from . import wecom  # noqa: F401
    except ImportError:
        pass


_load_builtins()
