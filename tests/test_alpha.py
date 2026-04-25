"""Tests for graft.alpha strategies."""

import pytest

from graft.alpha import constant, sigmoid_on_length, step


class TestConstant:
    def test_returns_value_regardless_of_length(self):
        fn = constant(0.5)
        assert fn(0) == 0.5
        assert fn(10) == 0.5
        assert fn(1000) == 0.5

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            constant(-0.1)
        with pytest.raises(ValueError):
            constant(1.1)


class TestSigmoid:
    def test_zero_match_returns_zero(self):
        fn = sigmoid_on_length()
        assert fn(0) == 0.0
        assert fn(-1) == 0.0

    def test_monotonic_in_match_length(self):
        fn = sigmoid_on_length(midpoint=4, steepness=1, max_alpha=0.7)
        assert fn(1) < fn(4) < fn(8) < fn(20)

    def test_asymptotes_to_max_alpha(self):
        fn = sigmoid_on_length(midpoint=4, steepness=1, max_alpha=0.7)
        # At very large match length, alpha should approach max_alpha.
        assert fn(100) == pytest.approx(0.7, abs=0.001)

    def test_at_midpoint_is_half_max(self):
        fn = sigmoid_on_length(midpoint=5, steepness=1, max_alpha=0.6)
        assert fn(5) == pytest.approx(0.3)

    def test_rejects_invalid_args(self):
        with pytest.raises(ValueError):
            sigmoid_on_length(steepness=0)
        with pytest.raises(ValueError):
            sigmoid_on_length(max_alpha=1.5)


class TestStep:
    def test_basic_steps(self):
        fn = step([(0, 0.0), (3, 0.3), (6, 0.5), (10, 0.7)])
        assert fn(0) == 0.0
        assert fn(2) == 0.0
        assert fn(3) == 0.3
        assert fn(5) == 0.3
        assert fn(6) == 0.5
        assert fn(9) == 0.5
        assert fn(10) == 0.7
        assert fn(100) == 0.7

    def test_unsorted_input(self):
        fn = step([(10, 0.7), (0, 0.0), (3, 0.3)])
        # Should sort internally.
        assert fn(5) == 0.3
        assert fn(15) == 0.7

    def test_empty_thresholds_raises(self):
        with pytest.raises(ValueError):
            step([])

    def test_below_smallest_threshold(self):
        # Below the smallest threshold, alpha is 0 (default).
        fn = step([(5, 0.5), (10, 0.7)])
        assert fn(3) == 0.0
