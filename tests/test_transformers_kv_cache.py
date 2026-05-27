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
    @pytest.fixture(autouse=True)
    def _reset(self, client):
        """Reset the module-scoped client's cache before every test so order
        is irrelevant and new tests can't accidentally inherit prior state."""
        client.reset_cache()

    def test_extension_matches_reset_recompute(self, client):
        """Extending the context via cache produces the same logits as
        recomputing from scratch with the cache reset."""
        _ = client.next_token_logprobs([10, 20, 30])
        cached = client.next_token_logprobs([10, 20, 30, 40])

        client.reset_cache()
        fresh = client.next_token_logprobs([10, 20, 30, 40])

        assert _max_abs_diff(cached, fresh) < 1e-5

    def test_long_extension_matches_reset_recompute(self, client):
        """Several consecutive extensions all stay numerically equivalent.

        Two-pass: first build the cached chain so each step extends from
        the prior cache (the property under test); then reset between
        each fresh recompute. Functionally identical to instantiating a
        new client per step, but avoids reloading the model four times.
        """
        cached_results = {}
        ctx = [5, 7, 11, 13]
        _ = client.next_token_logprobs(ctx)
        for tok in [17, 19, 23, 29]:
            ctx = ctx + [tok]
            cached_results[tuple(ctx)] = client.next_token_logprobs(ctx)

        for ctx_tuple, cached in cached_results.items():
            client.reset_cache()
            fresh = client.next_token_logprobs(list(ctx_tuple))
            assert _max_abs_diff(cached, fresh) < 1e-5

    def test_divergence_triggers_full_recompute(self, client):
        """A diverged context must re-forward and produce correct logits."""
        _ = client.next_token_logprobs([10, 20, 30])
        diverged = client.next_token_logprobs([99, 88, 77])

        client.reset_cache()
        fresh = client.next_token_logprobs([99, 88, 77])

        assert _max_abs_diff(diverged, fresh) < 1e-5

    def test_reset_cache_isolates_state(self, client):
        """After reset_cache, internal cache state is None."""
        _ = client.next_token_logprobs([1, 2, 3])
        assert client._cached_context is not None
        assert client._past_key_values is not None
        client.reset_cache()
        assert client._cached_context is None
        assert client._past_key_values is None

    def test_logprobs_are_valid_distribution(self, client):
        lp = client.next_token_logprobs([10, 20, 30])
        total = sum(math.exp(v) for v in lp.values())
        assert abs(total - 1.0) < 1e-3

    def test_empty_context_uses_bos_or_zero(self, client):
        lp = client.next_token_logprobs([])
        assert len(lp) == client.vocab_size

    def test_forward_exception_clears_cache(self, client, monkeypatch):
        """If the forward pass raises after potentially mutating the in-place
        DynamicCache, internal state must be reset so the next call cannot
        silently read stale KV against a context length that no longer matches.
        """
        _ = client.next_token_logprobs([1, 2, 3])
        assert client._cached_context is not None
        assert client._past_key_values is not None

        def boom(*args, **kwargs):
            raise RuntimeError("simulated forward failure")

        monkeypatch.setattr(client.model, "forward", boom)

        with pytest.raises(RuntimeError, match="simulated forward failure"):
            client.next_token_logprobs([1, 2, 3, 4])

        assert client._cached_context is None
        assert client._past_key_values is None

        monkeypatch.undo()
        lp = client.next_token_logprobs([1, 2, 3, 4])
        assert len(lp) == client.vocab_size
