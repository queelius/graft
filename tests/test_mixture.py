"""Tests for graft.mixture."""

import math

import pytest

from graft.mixture import geometric_mix, linear_mix


class TestLinearMix:
    def test_alpha_zero_returns_llm_only(self):
        p_llm = {0: 0.5, 1: 0.5}
        p_inf = {2: 1.0}
        result = linear_mix(p_llm, p_inf, alpha=0.0)
        assert result[0] == 0.5
        assert result[1] == 0.5
        assert result[2] == 0.0

    def test_alpha_one_returns_inf_only(self):
        p_llm = {0: 1.0}
        p_inf = {1: 0.4, 2: 0.6}
        result = linear_mix(p_llm, p_inf, alpha=1.0)
        assert result[0] == 0.0
        assert result[1] == 0.4
        assert result[2] == 0.6

    def test_sums_to_one_when_inputs_do(self):
        p_llm = {0: 0.5, 1: 0.4, 2: 0.1}
        p_inf = {1: 0.3, 2: 0.7}
        result = linear_mix(p_llm, p_inf, alpha=0.5)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_disjoint_supports_keep_both(self):
        # LLM has 0, inf has 1. Linear mixture preserves both.
        p_llm = {0: 1.0}
        p_inf = {1: 1.0}
        result = linear_mix(p_llm, p_inf, alpha=0.5)
        assert result[0] == 0.5
        assert result[1] == 0.5

    def test_inf_token_not_in_llm_survives(self):
        # Key win for grounding: a token from infinigram (e.g. a rare name)
        # not in the LLM's vocab still gets weight.
        p_llm = {0: 0.99, 1: 0.01}
        p_inf = {99: 1.0}  # token only inf knows
        result = linear_mix(p_llm, p_inf, alpha=0.3)
        assert result[99] == pytest.approx(0.30)
        assert result[0] == pytest.approx(0.7 * 0.99)


class TestGeometricMix:
    def test_alpha_zero_approximately_equals_llm(self):
        p_llm = {0: 0.6, 1: 0.4}
        p_inf = {2: 1.0}
        result = geometric_mix(p_llm, p_inf, alpha=0.0, smoothing=1e-8)
        # alpha=0: only p_llm contributes (each token's p_inf factor is x^0 = 1).
        # Smoothing inflates token 2 a tiny bit, but tokens 0 and 1 dominate.
        assert result[0] > 0.5
        assert result[1] > 0.3
        assert result[2] < 1e-6

    def test_normalizes_to_one(self):
        p_llm = {0: 0.3, 1: 0.7}
        p_inf = {0: 0.6, 1: 0.4}
        result = geometric_mix(p_llm, p_inf, alpha=0.5)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_disjoint_supports_with_smoothing(self):
        # Without smoothing, the product is 0 everywhere; degenerate.
        # With smoothing, both supports survive but each is weak.
        p_llm = {0: 1.0}
        p_inf = {1: 1.0}
        result = geometric_mix(p_llm, p_inf, alpha=0.5, smoothing=1e-4)
        assert 0 in result
        assert 1 in result
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_disjoint_supports_zero_smoothing_uniform_fallback(self):
        # With smoothing=0 and disjoint supports, every raw value is 0.
        # Implementation falls back to uniform.
        p_llm = {0: 1.0}
        p_inf = {1: 1.0}
        result = geometric_mix(p_llm, p_inf, alpha=0.5, smoothing=0.0)
        assert result[0] == 0.5
        assert result[1] == 0.5

    def test_requires_both_to_endorse(self):
        # The defining behavior: if EITHER side puts very low mass on a token,
        # the geometric mix downweights it heavily relative to linear mix.
        p_llm = {0: 0.5, 1: 0.5}
        p_inf = {0: 0.99, 1: 0.01}
        geo = geometric_mix(p_llm, p_inf, alpha=0.5, smoothing=1e-12)
        # Token 0 (both endorse) should dominate; token 1 (only LLM) should be small.
        assert geo[0] > 0.8
        assert geo[1] < 0.2
