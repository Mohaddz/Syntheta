"""Tests for component registry."""

import pytest

from syntheta.pipeline.registry import (
    _filters,
    _generators,
    _transformers,
    get_generator,
    register_filter,
    register_generator,
    register_transformer,
)


class TestRegistry:
    def test_register_and_get_generator(self):
        @register_generator("test_gen")
        class TestGen:
            pass

        assert get_generator("test_gen") is TestGen
        del _generators["test_gen"]

    def test_register_and_get_transformer(self):
        from syntheta.pipeline.registry import get_transformer

        @register_transformer("test_trans")
        class TestTrans:
            pass

        assert get_transformer("test_trans") is TestTrans
        del _transformers["test_trans"]

    def test_register_and_get_filter(self):
        from syntheta.pipeline.registry import get_filter

        @register_filter("test_filt")
        class TestFilt:
            pass

        assert get_filter("test_filt") is TestFilt
        del _filters["test_filt"]

    def test_unknown_generator_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_generator("nonexistent")
