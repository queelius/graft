"""Tests for the REST server's request resolution and end-to-end wiring.

These exercise:
    1. ``_resolve_alpha`` and ``_resolve_mixture`` (unit-level)
    2. The HTTP surface end-to-end via :class:`fastapi.testclient.TestClient`,
       using a deterministic FakeLLM and a byte-level Infinigram so no torch /
       HF model is required.
"""

import math
from typing import Dict, List

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from infinigram import Infinigram

from graft.mixture import geometric_mix, linear_mix
from graft.server.api import _resolve_alpha, _resolve_mixture, make_app
from graft.server.api import CompletionRequest


class FakeLLM:
    """Same shape as tests/test_pipeline.py's FakeLLM (byte-vocab, deterministic)."""

    def __init__(self, vocab_size: int = 256, distribution: Dict[int, float] = None):
        self._vocab_size = vocab_size
        if distribution is None:
            distribution = {i: 1.0 / vocab_size for i in range(vocab_size)}
        z = sum(distribution.values())
        self._logprobs = {v: math.log(p / z) for v, p in distribution.items() if p > 0}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenizer_id(self) -> str:
        return "fake"

    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]:
        return dict(self._logprobs)


class FakeTokenizer:
    """Byte-level tokenizer matching FakeLLM's 256-byte vocab."""

    def encode(self, s: str) -> List[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


# --- _resolve_alpha ----------------------------------------------------------


class TestResolveAlpha:
    def test_constant(self):
        req = CompletionRequest(prompt="x", alpha=0.4, alpha_strategy="constant")
        fn = _resolve_alpha(req)
        assert fn(0) == 0.4
        assert fn(100) == 0.4

    def test_sigmoid(self):
        req = CompletionRequest(
            prompt="x",
            alpha_strategy="sigmoid",
            sigmoid_midpoint=4,
            sigmoid_steepness=1,
            sigmoid_max_alpha=0.7,
        )
        fn = _resolve_alpha(req)
        # Monotonic and asymptotes to max_alpha.
        assert fn(0) == 0.0
        assert fn(1) < fn(4) < fn(20)
        assert fn(100) == pytest.approx(0.7, abs=1e-3)

    def test_step(self):
        req = CompletionRequest(
            prompt="x",
            alpha_strategy="step",
            step_thresholds=[(0, 0.0), (3, 0.3), (6, 0.5)],
        )
        fn = _resolve_alpha(req)
        assert fn(2) == 0.0
        assert fn(3) == 0.3
        assert fn(7) == 0.5

    def test_step_requires_thresholds(self):
        req = CompletionRequest(prompt="x", alpha_strategy="step")
        with pytest.raises(HTTPException) as exc:
            _resolve_alpha(req)
        assert exc.value.status_code == 400
        assert "step_thresholds" in exc.value.detail

    def test_unknown_strategy(self):
        req = CompletionRequest(prompt="x", alpha_strategy="bogus")
        with pytest.raises(HTTPException) as exc:
            _resolve_alpha(req)
        assert exc.value.status_code == 400


# --- _resolve_mixture --------------------------------------------------------


class TestResolveMixture:
    def test_linear_default(self):
        req = CompletionRequest(prompt="x")  # mixture_strategy default = "linear"
        fn = _resolve_mixture(req)
        assert fn is linear_mix

    def test_linear_explicit(self):
        req = CompletionRequest(prompt="x", mixture_strategy="linear")
        fn = _resolve_mixture(req)
        assert fn is linear_mix

    def test_geometric(self):
        req = CompletionRequest(
            prompt="x",
            mixture_strategy="geometric",
            geometric_smoothing=1e-6,
        )
        fn = _resolve_mixture(req)
        # Functionally equivalent to geometric_mix with the configured smoothing.
        p_llm = {0: 0.5, 1: 0.5}
        p_inf = {0: 0.9, 1: 0.1}
        expected = geometric_mix(p_llm, p_inf, 0.5, smoothing=1e-6)
        actual = fn(p_llm, p_inf, 0.5)
        for k in expected:
            assert actual[k] == pytest.approx(expected[k])

    def test_unknown_strategy(self):
        req = CompletionRequest(prompt="x", mixture_strategy="bogus")
        with pytest.raises(HTTPException) as exc:
            _resolve_mixture(req)
        assert exc.value.status_code == 400


# --- end-to-end through TestClient ------------------------------------------


@pytest.fixture()
def client():
    llm = FakeLLM()
    inf = Infinigram(b"the cat sat on the mat the cat sat on the mat")
    tok = FakeTokenizer()
    app = make_app(llm, inf, tok)
    return TestClient(app)


class TestHttpSurface:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["llm"] == "fake"
        assert body["vocab_size"] == 256

    def test_completion_constant(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 5,
                "temperature": 0.0,
                "alpha_strategy": "constant",
                "alpha": 0.3,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["tokens"]) == 5
        assert body["metadata"]["alpha_strategy"] == "constant"

    def test_completion_sigmoid(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 3,
                "temperature": 0.0,
                "alpha_strategy": "sigmoid",
            },
        )
        assert r.status_code == 200
        assert len(r.json()["tokens"]) == 3

    def test_completion_step(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 4,
                "temperature": 0.0,
                "alpha_strategy": "step",
                "step_thresholds": [[0, 0.0], [3, 0.4], [6, 0.6]],
            },
        )
        assert r.status_code == 200
        assert len(r.json()["tokens"]) == 4

    def test_completion_step_missing_thresholds_400(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 1,
                "alpha_strategy": "step",
            },
        )
        assert r.status_code == 400
        assert "step_thresholds" in r.json()["detail"]

    def test_completion_geometric_mixture(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 3,
                "temperature": 0.0,
                "alpha_strategy": "constant",
                "alpha": 0.5,
                "mixture_strategy": "geometric",
                "geometric_smoothing": 1e-6,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["tokens"]) == 3
        assert body["metadata"]["mixture_strategy"] == "geometric"

    def test_completion_unknown_strategy_400(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 1,
                "alpha_strategy": "bogus",
            },
        )
        assert r.status_code == 400

    def test_completion_unknown_mixture_400(self, client):
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "the ",
                "max_tokens": 1,
                "mixture_strategy": "bogus",
            },
        )
        assert r.status_code == 400
