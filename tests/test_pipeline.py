"""Tests for graft.pipeline.generate_grounded.

These tests use a deterministic FakeLLM rather than loading any HF model so
the test suite stays fast and CPU-only.
"""

import math
from typing import Dict, List

import pytest
from infinigram import Infinigram

from graft.alpha import constant
from graft.pipeline import generate_grounded


class FakeLLM:
    """Deterministic LLM for tests: returns a configurable distribution.

    Vocab size matches infinigram's byte vocab (256) by default so the alignment
    check passes.
    """

    def __init__(self, vocab_size: int = 256, distribution: Dict[int, float] = None):
        self._vocab_size = vocab_size
        if distribution is None:
            # Uniform over the whole vocab.
            distribution = {i: 1.0 / vocab_size for i in range(vocab_size)}
        # Normalize and store as logprobs for the protocol.
        z = sum(distribution.values())
        self._logprobs = {v: math.log(p / z) for v, p in distribution.items() if p > 0}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenizer_id(self) -> str:
        return "fake"

    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]:
        return dict(self._logprobs)


class TestGenerateGrounded:
    def test_generates_max_tokens(self):
        llm = FakeLLM()
        inf = Infinigram(b"the cat sat on the mat the cat sat on the mat")
        tokens = generate_grounded(
            prompt=list(b"the "),
            llm=llm,
            inf=inf,
            max_tokens=5,
            temperature=0,  # greedy → deterministic
        )
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < 256 for t in tokens)

    def test_alpha_zero_uses_llm_only(self):
        # If alpha = 0 always, the infinigram is irrelevant. Result should
        # be driven entirely by the LLM. Use an LLM that always produces
        # token 'X' deterministically.
        x = ord("X")
        llm = FakeLLM(distribution={x: 1.0})
        inf = Infinigram(b"hello world")
        tokens = generate_grounded(
            prompt=list(b"h"),
            llm=llm,
            inf=inf,
            max_tokens=5,
            temperature=0,
            alpha_fn=constant(0.0),
        )
        assert tokens == [x] * 5

    def test_grounds_when_inf_is_strong(self):
        # When infinigram has a clear continuation and alpha = 1.0, output
        # should follow the corpus. After "ab" in "ababab..." comes "a".
        llm = FakeLLM(distribution={ord("Z"): 1.0})  # LLM wants 'Z'
        inf = Infinigram(b"ababab" * 50)
        tokens = generate_grounded(
            prompt=list(b"ab"),
            llm=llm,
            inf=inf,
            max_tokens=4,
            temperature=0,
            alpha_fn=constant(1.0),
        )
        # alpha=1.0 means inf-only. Corpus pattern: ab → a → b → a → b → ...
        assert tokens[0] == ord("a")
        assert tokens[1] == ord("b")

    def test_falls_back_to_llm_when_inf_has_no_match(self):
        # Build a tiny corpus that doesn't contain the prompt context.
        # Infinigram should return None for continuations; pipeline should
        # fall through to LLM-only.
        x = ord("X")
        llm = FakeLLM(distribution={x: 1.0})
        inf = Infinigram(b"abc")  # tiny corpus
        # Prompt with token IDs not in the corpus at all.
        tokens = generate_grounded(
            prompt=[250, 251, 252],
            llm=llm,
            inf=inf,
            max_tokens=3,
            temperature=0,
            alpha_fn=constant(0.5),
        )
        assert tokens == [x, x, x]

    def test_vocab_mismatch_raises(self):
        llm = FakeLLM(vocab_size=1000)
        inf = Infinigram(b"hello")  # vocab=256
        with pytest.raises(ValueError, match="Tokenizer mismatch"):
            generate_grounded(prompt=[1, 2], llm=llm, inf=inf, max_tokens=1)

    def test_stop_tokens_truncate(self):
        # Greedy LLM produces 'A' forever, but we stop at sequence [A, A].
        a = ord("A")
        llm = FakeLLM(distribution={a: 1.0})
        inf = Infinigram(b"hello")
        tokens = generate_grounded(
            prompt=[a],
            llm=llm,
            inf=inf,
            max_tokens=10,
            temperature=0,
            alpha_fn=constant(0.0),
            stop_tokens=[[a, a]],
        )
        # Should stop and strip the [a, a] suffix immediately.
        assert tokens == []
