"""Integration tests for TransformersClient KV-cache reuse.

Loads a tiny HF model and verifies that cached and reset-and-recomputed
calls return numerically identical logits. Skipped when torch /
transformers are not installed.

Uses ``sshleifer/tiny-gpt2`` (4-layer toy model, ~hundreds of KB) so the
download cost is negligible after the first run.
"""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from graft.llm.transformers import TransformersClient

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def client():
    return TransformersClient(TINY_MODEL, device="cpu")


def _max_abs_diff(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


class TestKvCacheCorrectness:
    def test_extension_matches_reset_recompute(self, client):
        """Extending the context via cache produces the same logits as
        recomputing from scratch with the cache reset."""
        client.reset_cache()
        # Prime cache with [10, 20, 30].
        _ = client.next_token_logprobs([10, 20, 30])
        # Extend by one token; this hits the cache path.
        cached = client.next_token_logprobs([10, 20, 30, 40])

        # Now reset and recompute from scratch.
        client.reset_cache()
        fresh = client.next_token_logprobs([10, 20, 30, 40])

        assert _max_abs_diff(cached, fresh) < 1e-5

    def test_long_extension_matches_reset_recompute(self, client):
        """Several consecutive extensions all stay numerically equivalent."""
        client.reset_cache()
        ctx = [5, 7, 11, 13]
        _ = client.next_token_logprobs(ctx)
        for tok in [17, 19, 23, 29]:
            ctx = ctx + [tok]
            cached = client.next_token_logprobs(ctx)

            client_reset = TransformersClient(TINY_MODEL, device="cpu")
            fresh = client_reset.next_token_logprobs(ctx)
            assert _max_abs_diff(cached, fresh) < 1e-5

    def test_divergence_triggers_full_recompute(self, client):
        """A diverged context must re-forward and produce correct logits."""
        client.reset_cache()
        _ = client.next_token_logprobs([10, 20, 30])
        # Diverge: prefix doesn't match.
        diverged = client.next_token_logprobs([99, 88, 77])

        client.reset_cache()
        fresh = client.next_token_logprobs([99, 88, 77])

        assert _max_abs_diff(diverged, fresh) < 1e-5

    def test_reset_cache_isolates_state(self, client):
        """After reset_cache, internal cache state is None."""
        client.reset_cache()
        _ = client.next_token_logprobs([1, 2, 3])
        assert client._cached_context is not None
        assert client._past_key_values is not None
        client.reset_cache()
        assert client._cached_context is None
        assert client._past_key_values is None

    def test_logprobs_are_valid_distribution(self, client):
        client.reset_cache()
        lp = client.next_token_logprobs([10, 20, 30])
        # Probabilities sum to ~1.
        total = sum(math.exp(v) for v in lp.values())
        assert abs(total - 1.0) < 1e-3

    def test_empty_context_uses_bos_or_zero(self, client):
        client.reset_cache()
        # Should not raise; uses bos_token_id (or 0) under the hood.
        lp = client.next_token_logprobs([])
        assert len(lp) == client.vocab_size
