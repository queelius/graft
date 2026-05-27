"""End-to-end integration: real HF LLM + real Infinigram + generate_grounded.

Until now, the only "real model" coverage was manual smoke-testing via
``graft-serve``. This test exercises the full stack against
``sshleifer/tiny-gpt2`` and a small on-disk infinigram index built with
the same tokenizer, so a future tokenizer-alignment regression or a
breaking change in ``Infinigram.build`` / ``continuations`` is caught
in CI rather than in production.

Skipped when torch / transformers are not installed.
"""

import math
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from infinigram import Infinigram

from graft.alpha import constant, sigmoid_on_length
from graft.llm.transformers import TransformersClient
from graft.pipeline import generate_grounded

TINY_MODEL = "sshleifer/tiny-gpt2"

# Small but non-trivial: enough repetition for the infinigram to find suffix
# matches, but small enough that the index builds in milliseconds.
CORPUS_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "the rain in spain stays mainly in the plain. "
    "the rain in spain stays mainly in the plain. "
    "to be or not to be, that is the question. "
    "to be or not to be, that is the question. "
)


@pytest.fixture(scope="module")
def stack():
    """A loaded LLM + a real infinigram index built with the same tokenizer.

    Module-scoped: model load + index build are both done once.
    """
    llm = TransformersClient(TINY_MODEL, device="cpu")
    with tempfile.TemporaryDirectory(prefix="graft-e2e-") as tmpdir:
        index_path = Path(tmpdir) / "index"
        Infinigram.build(
            CORPUS_TEXT,
            str(index_path),
            tokenizer=llm.tokenizer,
            verbose=False,
        )
        inf = Infinigram.load(str(index_path), tokenizer=llm.tokenizer)
        yield llm, inf


class TestEndToEnd:
    def test_vocab_sizes_align(self, stack):
        """The whole project rests on this; assert it directly."""
        llm, inf = stack
        assert llm.vocab_size == inf.vocab_size

    def test_generates_requested_token_count(self, stack):
        llm, inf = stack
        prompt = llm.tokenizer.encode("the quick brown")
        tokens = generate_grounded(
            prompt=prompt,
            llm=llm,
            inf=inf,
            max_tokens=8,
            temperature=0.0,
            alpha_fn=constant(0.3),
        )
        assert len(tokens) == 8
        assert all(0 <= t < llm.vocab_size for t in tokens)

    def test_decodes_to_string(self, stack):
        """Output tokens must round-trip through the tokenizer."""
        llm, inf = stack
        prompt = llm.tokenizer.encode("the rain in")
        tokens = generate_grounded(
            prompt=prompt,
            llm=llm,
            inf=inf,
            max_tokens=6,
            temperature=0.0,
            alpha_fn=sigmoid_on_length(midpoint=3, max_alpha=0.7),
        )
        text = llm.tokenizer.decode(tokens)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_corpus_dominates_at_alpha_one(self, stack):
        """alpha=1.0 + greedy must reduce to argmax of the infinigram's
        continuation counts at the longest matching suffix.

        Computes that argmax independently from inf.continuations(prompt)
        and compares against the pipeline output, so the assertion does
        not depend on how the LLM's tokenizer happens to encode any
        particular surface form.
        """
        llm, inf = stack
        prompt = llm.tokenizer.encode("the quick brown")

        counts = inf.continuations(prompt)
        assert counts is not None, "in-corpus prefix should have continuations"
        expected = max(counts.items(), key=lambda kv: kv[1])[0]

        tokens = generate_grounded(
            prompt=prompt,
            llm=llm,
            inf=inf,
            max_tokens=1,
            temperature=0.0,
            alpha_fn=constant(1.0),
        )
        assert tokens[0] == expected

    def test_alpha_zero_matches_bare_llm_argmax(self, stack):
        """With alpha=0, the pipeline must reduce to LLM-only sampling.

        Greedy decoding with alpha=0 should yield exactly the LLM's argmax
        token at each step.
        """
        llm, inf = stack
        prompt = llm.tokenizer.encode("hello world")

        # Reset the cache so the LLM-direct call below sees the same state
        # as the pipeline call.
        llm.reset_cache()
        pipeline_tokens = generate_grounded(
            prompt=prompt,
            llm=llm,
            inf=inf,
            max_tokens=3,
            temperature=0.0,
            alpha_fn=constant(0.0),
        )

        # Independently roll forward the LLM's greedy argmax.
        llm.reset_cache()
        ctx = list(prompt)
        expected = []
        for _ in range(3):
            lp = llm.next_token_logprobs(ctx)
            argmax = max(lp.items(), key=lambda kv: kv[1])[0]
            expected.append(argmax)
            ctx.append(argmax)

        assert pipeline_tokens == expected

    def test_off_corpus_prompt_falls_back_to_llm(self, stack):
        """A prompt whose tail isn't in the corpus exercises the
        ``inf.continuations(...) is None`` early-exit path. Must not raise."""
        llm, inf = stack
        # Tokens unlikely to appear contiguously in the toy corpus.
        prompt = llm.tokenizer.encode("xylophone quaternion supercalifragilistic")
        tokens = generate_grounded(
            prompt=prompt,
            llm=llm,
            inf=inf,
            max_tokens=3,
            temperature=0.0,
            alpha_fn=constant(0.5),
        )
        assert len(tokens) == 3

    def test_logprobs_form_valid_distribution(self, stack):
        """Sanity: the LLM's per-step logprobs sum to ~1 in probability space."""
        llm, _ = stack
        llm.reset_cache()
        lp = llm.next_token_logprobs(llm.tokenizer.encode("the"))
        total = sum(math.exp(v) for v in lp.values())
        assert abs(total - 1.0) < 1e-3
