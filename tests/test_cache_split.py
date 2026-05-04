"""Tests for the KV-cache prefix-extension helper.

Pure-function logic; runs without torch / transformers.
"""

import pytest

from graft.llm.transformers import _cache_split


class TestCacheSplit:
    def test_none_cache_is_miss(self):
        extends, to_fwd = _cache_split([1, 2, 3], None)
        assert extends is False
        assert to_fwd == [1, 2, 3]

    def test_empty_cache_is_miss(self):
        # Empty list cached_context counts as no useful prefix.
        extends, to_fwd = _cache_split([1, 2, 3], [])
        assert extends is False
        assert to_fwd == [1, 2, 3]

    def test_strict_prefix_extends(self):
        extends, to_fwd = _cache_split([1, 2, 3, 4, 5], [1, 2, 3])
        assert extends is True
        assert to_fwd == [4, 5]

    def test_extension_by_one(self):
        extends, to_fwd = _cache_split([1, 2, 3, 4], [1, 2, 3])
        assert extends is True
        assert to_fwd == [4]

    def test_equal_context_is_miss(self):
        # No new tokens to forward; signal cache miss so caller re-forwards
        # rather than attempting a 0-length input_ids.
        extends, to_fwd = _cache_split([1, 2, 3], [1, 2, 3])
        assert extends is False
        assert to_fwd == [1, 2, 3]

    def test_divergence_is_miss(self):
        extends, to_fwd = _cache_split([1, 2, 9, 4], [1, 2, 3])
        assert extends is False
        assert to_fwd == [1, 2, 9, 4]

    def test_cache_longer_than_context_is_miss(self):
        extends, to_fwd = _cache_split([1, 2], [1, 2, 3, 4])
        assert extends is False
        assert to_fwd == [1, 2]

    def test_disjoint_context_is_miss(self):
        extends, to_fwd = _cache_split([7, 8, 9], [1, 2, 3])
        assert extends is False
        assert to_fwd == [7, 8, 9]
