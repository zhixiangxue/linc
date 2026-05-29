"""Tests for linc.adapters registry."""

from __future__ import annotations

import pytest

from linc.adapters import (
    clear,
    get,
    is_supported,
    register,
    supported,
    unregister,
)
from linc.core.adapter import Adapter
from linc.core.errors import UnknownPlatform

from _fakes.fake_adapter import FakeAdapter


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot/restore the registry around each test so registrations don't leak."""
    from linc.adapters import _REGISTRY

    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_register_and_get_roundtrip():
    register(FakeAdapter)
    assert get("fake") is FakeAdapter
    assert is_supported("fake")
    assert "fake" in supported()


def test_register_as_decorator_returns_class():
    class MyAdapter(Adapter):
        name = "tmp"

        async def start(self): ...
        async def stop(self): ...
        async def send(self, conv_id, content): return "x", {}
        def parse_inbound(self, raw): return None

    returned = register(MyAdapter)
    assert returned is MyAdapter
    assert get("tmp") is MyAdapter


def test_register_rejects_missing_name():
    class NoName(Adapter):
        # name intentionally unset
        async def start(self): ...
        async def stop(self): ...
        async def send(self, conv_id, content): return "x", {}
        def parse_inbound(self, raw): return None

    with pytest.raises(ValueError, match="must define a non-empty class attribute `name`"):
        register(NoName)


def test_register_rejects_conflicting_class_for_same_name():
    register(FakeAdapter)

    class Imposter(Adapter):
        name = "fake"

        async def start(self): ...
        async def stop(self): ...
        async def send(self, conv_id, content): return "x", {}
        def parse_inbound(self, raw): return None

    with pytest.raises(ValueError, match="already registered"):
        register(Imposter)


def test_register_same_class_twice_is_idempotent():
    register(FakeAdapter)
    register(FakeAdapter)  # must not raise
    assert get("fake") is FakeAdapter


def test_get_unknown_raises_unknown_platform():
    with pytest.raises(UnknownPlatform, match="not registered"):
        get("nope")


def test_unregister_is_noop_for_unknown():
    unregister("never-existed")  # must not raise


def test_clear_wipes_registry():
    register(FakeAdapter)
    assert is_supported("fake")
    clear()
    assert not is_supported("fake")
    assert supported() == frozenset()


def test_supported_returns_snapshot_not_live_view():
    register(FakeAdapter)
    snap = supported()
    unregister("fake")
    # snap was a frozenset taken at call time -> must still contain "fake"
    assert "fake" in snap
    # but live state has changed
    assert not is_supported("fake")
